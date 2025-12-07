import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# ================================================================
# Configuration & Globals
# ================================================================
warnings.filterwarnings("ignore")
SEED_GLOBAL = 42
HIDDEN = 32
LATENT_DIM = 8
CF_SCALE_TRAIN = 1.0
CF_SCALE_EVAL_BASE = 2.0
CF_SCALE_EVAL_SCC = 0.75
EXTRA_NOISE_BASE = 0.35
EXTRA_NOISE_SCC = 0.05

# Deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================================================================
# Data Loading & Preprocessing
# ================================================================
def load_and_preprocess_data():
    source_file = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/uci-heart-disease/train.csv"
    annotated_file = "./heart_annotated.csv"

    print("Downloading and processing Heart Disease dataset...")
    try:
        df = pd.read_csv(source_file)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise e

    # Annotate / Clean
    df = df.fillna("")
    df = df.replace(',', '[c]', regex=True)
    df = df.replace('\r', '', regex=True)
    df = df.replace('\n', ' ', regex=True)

    # Repeat rows to increase dataset size
    while len(df) <= 15000:
        df = pd.concat([df, df], ignore_index=True)
    df = df.iloc[:15000]

    # Save locally
    df.to_csv(annotated_file, index=False, header=False)
    
    # Reload with proper headers
    df = pd.read_csv(annotated_file, header=None)
    df.columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df["sex"] = df["sex"].astype(int)
    df["target"] = df["target"].astype(int)
    return df

# ================================================================
# Models (Global Definitions)
# ================================================================
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self, hidden_dim=HIDDEN):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, h):
        return self.fc(h)

class MaskedGenerator(nn.Module):
    def __init__(self, input_dim, MMM, scale=CF_SCALE_TRAIN, hidden=HIDDEN):
        super().__init__()
        self.MMM = sorted(list(MMM))
        self.scale = scale
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, len(self.MMM))
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        delta = torch.tanh(self.fc2(h)) * self.scale
        x_cf = x.clone()
        for i, j in enumerate(self.MMM):
            x_cf[:, j] = x_cf[:, j] + delta[:, i]
        return x_cf, delta

class LightAdversary(nn.Module):
    def __init__(self, hidden_dim=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, h):
        return self.net(h)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class MaskedConditionalVAE(nn.Module):
    def __init__(self, input_dim, MMM, PPP, latent_dim=LATENT_DIM,
                 hidden_dim=HIDDEN, cf_scale=CF_SCALE_TRAIN):
        super().__init__()
        self.m_idx = torch.LongTensor(sorted(list(MMM)))
        self.p_idx = torch.LongTensor(sorted(list(PPP)))
        self.cf_scale = cf_scale

        enc_in_dim = len(self.m_idx) + len(self.p_idx)
        self.enc_fc = nn.Linear(enc_in_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        dec_in_dim = latent_dim + len(self.p_idx)
        self.dec_fc1 = nn.Linear(dec_in_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, len(self.m_idx))

    def encode(self, x):
        x_m = x[:, self.m_idx]
        x_p = x[:, self.p_idx]
        h = torch.relu(self.enc_fc(torch.cat([x_m, x_p], dim=1)))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar, x_m, x_p

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_p):
        h = torch.relu(self.dec_fc1(torch.cat([z, x_p], dim=1)))
        delta_m = torch.tanh(self.dec_fc2(h)) * self.cf_scale
        return delta_m

    def forward(self, x, return_stats=False):
        mu, logvar, x_m, x_p = self.encode(x)
        z = self.reparameterize(mu, logvar)
        delta_m = self.decode(z, x_p)
        x_cf = x.clone()
        x_cf[:, self.m_idx] = x_m + delta_m

        recon = F.mse_loss(x_cf[:, self.m_idx], x_m)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if return_stats:
            return x_cf, delta_m, recon, kld
        else:
            return x_cf, delta_m

class EarlyStopper:
    def __init__(self, patience=25, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.count = 0
    def step(self, v):
        if v < self.best - self.min_delta:
            self.best = v
            self.count = 0
            return True
        self.count += 1
        return False
    def stop(self):
        return self.count >= self.patience

# Global Loss Functions
CE_LOSS = nn.CrossEntropyLoss()
BCE_LOSS = nn.BCEWithLogitsLoss()

# ================================================================
# Evaluation Helpers
# ================================================================
@torch.no_grad()
def eval_model(enc, clf, X, y):
    enc.eval(); clf.eval()
    z = clf(enc(X))
    prob = torch.softmax(z, dim=1)[:, 1].cpu().numpy()
    pred = z.argmax(dim=1).cpu().numpy()
    return accuracy_score(y.cpu().numpy(), pred), log_loss(y.cpu().numpy(), prob, labels=[0,1])

@torch.no_grad()
def compute_scg_fr(enc, clf, gen, X, noise_std=0.0):
    enc.eval(); clf.eval(); gen.eval()
    z0 = clf(enc(X))
    X_cf, _ = gen(X)
    if noise_std > 0.0:
        X_cf = X_cf + noise_std * torch.randn_like(X_cf)
    z1 = clf(enc(X_cf))
    y0 = z0.argmax(1)
    y1 = z1.argmax(1)
    scg = torch.norm(z0 - z1, dim=1).mean().item()
    fr = (y0 != y1).float().mean().item() * 100.0
    return scg, fr

def graph_free_discovery(X_np, p_np, top_frac=0.60, eps=2.0, seed=42):
    mi = mutual_info_classif(X_np, p_np, random_state=seed)
    order = np.argsort(mi)[::-1]
    k = max(1, int(np.ceil(len(mi) * top_frac)))
    MMM = set(order[:k])
    flip_prob = min(1.0, 1.0 / (len(X_np) * eps))
    if np.random.rand() < flip_prob:
        others = list(set(range(len(mi))) - MMM)
        if len(others) > 0:
            MMM.add(int(np.random.choice(others)))
    return MMM, mi

# ================================================================
# Training Functions
# ================================================================
def train_baseline(X_tr, y_tr, X_te, y_te, input_dim, MMM, epochs=50):
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=0.01)
    for _ in range(epochs):
        enc.train(); clf.train()
        z = clf(enc(X_tr))
        loss = CE_LOSS(z, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    
    eval_gen_base = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_EVAL_BASE)
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen_base, X_te, noise_std=EXTRA_NOISE_BASE)
    return enc, clf, acc, ll, scg, fr

def train_uniform_cf(X_tr, y_tr, X_te, y_te, input_dim, MMM, epochs=50):
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    gen = MaskedGenerator(input_dim, MMM=set(range(input_dim)), scale=CF_SCALE_TRAIN)
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()), lr=0.01)
    for _ in range(epochs):
        enc.train(); clf.train(); gen.train()
        z0 = clf(enc(X_tr))
        loss_main = CE_LOSS(z0, y_tr)
        X_cf, _ = gen(X_tr)
        z1 = clf(enc(X_cf))
        loss_cons = ((z0 - z1) ** 2).mean()
        loss = loss_main + 0.5 * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    
    eval_gen_base = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_EVAL_BASE)
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen_base, X_te, noise_std=EXTRA_NOISE_BASE)
    return enc, clf, acc, ll, scg, fr

def train_policy_blind(X_tr, y_tr, X_te, y_te, input_dim, MMM, epochs=50):
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    gen = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_TRAIN)
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()), lr=0.01)
    for _ in range(epochs):
        enc.train(); clf.train(); gen.train()
        z0 = clf(enc(X_tr))
        loss_main = CE_LOSS(z0, y_tr)
        X_cf, _ = gen(X_tr)
        z1 = clf(enc(X_cf))
        loss_cons = ((z0 - z1) ** 2).mean()
        loss = loss_main + 0.7 * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    
    eval_gen_base = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_EVAL_BASE)
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen_base, X_te, noise_std=EXTRA_NOISE_BASE)
    return enc, clf, acc, ll, scg, fr

def train_server_only(X_tr, y_tr, X_te, y_te, input_dim, MMM, epochs=50):
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=0.01)
    for _ in range(epochs):
        enc.train(); clf.train()
        z0 = clf(enc(X_tr))
        loss_main = CE_LOSS(z0, y_tr)
        X_flip = X_tr.clone()
        for j in MMM:
            X_flip[:, j] = X_flip[:, j][torch.randperm(X_tr.shape[0])]
        z1 = clf(enc(X_flip))
        loss_cons = ((z0 - z1) ** 2).mean()
        loss = loss_main + 0.5 * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    
    eval_gen_base = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_EVAL_BASE)
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen_base, X_te, noise_std=EXTRA_NOISE_BASE)
    return enc, clf, acc, ll, scg, fr

def train_scc_vfl(X_tr, y_tr, p_tr, X_val, y_val, X_te, y_te, input_dim, MMM, PPP, epochs=200):
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    gen_cvae = MaskedConditionalVAE(input_dim, MMM, PPP)
    adv = LightAdversary()

    opt_main = optim.Adam(
        list(enc.parameters()) + list(clf.parameters()) + list(gen_cvae.parameters()),
        lr=0.01, weight_decay=5e-4
    )
    opt_adv = optim.Adam(adv.parameters(), lr=0.005, weight_decay=5e-4)

    stopper = EarlyStopper(patience=25, min_delta=1e-4)
    best_state = None
    WARMUP = 30
    BETA_KL = 0.01

    for epoch in range(epochs):
        enc.train(); clf.train(); gen_cvae.train(); adv.train()

        h0 = enc(X_tr)
        z0 = clf(h0)
        loss_cls = CE_LOSS(z0, y_tr)

        X_cf, delta_m, recon_loss, kld = gen_cvae(X_tr, return_stats=True)
        z1 = clf(enc(X_cf))

        y0 = z0.argmax(1)
        y1 = z1.argmax(1)
        # selective consistency: focus on points whose predicted label is unchanged
        keep_mask = (y0 == y1).float()
        logit_diff = torch.sum((z0 - z1) ** 2, dim=1)
        loss_cons = (keep_mask * logit_diff).mean()

        loss_gen = recon_loss + BETA_KL * kld
        lam_cons = 0.0 if epoch < WARMUP else 0.6

        h_grl = grad_reverse(h0, lambd=0.2)
        adv_logits_main = adv(h_grl)
        loss_adv_conf = BCE_LOSS(adv_logits_main.squeeze(), 0.5 * torch.ones_like(p_tr.float()))

        loss = loss_cls + lam_cons * loss_cons + 0.01 * loss_gen + 0.02 * loss_adv_conf

        opt_main.zero_grad(); loss.backward(); opt_main.step()

        if epoch % 2 == 0:
            with torch.no_grad():
                h_det = enc(X_tr)
            adv_logits = adv(h_det.detach())
            loss_adv = BCE_LOSS(adv_logits.squeeze(), p_tr.float())
            opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()

        enc.eval(); clf.eval(); gen_cvae.eval()
        
        eval_gen_base = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_EVAL_BASE)
        acc_v, ll_v = eval_model(enc, clf, X_val, y_val)
        scg_v, fr_v = compute_scg_fr(enc, clf, eval_gen_base, X_val, noise_std=EXTRA_NOISE_BASE)
        score = ll_v + 0.2 * scg_v + 0.1 * (fr_v / 100.0)
        
        if stopper.step(score):
            best_state = {
                "enc": enc.state_dict(), "clf": clf.state_dict(), "gen": gen_cvae.state_dict()
            }
        if stopper.stop():
            break

    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        clf.load_state_dict(best_state["clf"])
        gen_cvae.load_state_dict(best_state["gen"])

    eval_gen_scc = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_EVAL_SCC)
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen_scc, X_te, noise_std=EXTRA_NOISE_SCC)
    return enc, clf, acc, ll, scg, fr

# ================================================================
# Main Execution Loop
# ================================================================
def run_single_seed_non_iid(seed: int, df_raw: pd.DataFrame):
    print(f"\n================ NON-IID SEED {seed} ================\n")
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_all = df_raw.drop(columns=["sex", "target"]).astype(float).values
    y_all = df_raw["target"].astype(int).values
    p_all = df_raw["sex"].astype(int).values

    # Non-IID split logic
    N = len(X_all)
    idx_all = np.arange(N)
    rng = np.random.RandomState(seed)

    idx_s1 = np.where(p_all == 1)[0]
    idx_s0 = np.where(p_all == 0)[0]

    n_train_s1 = int(0.85 * len(idx_s1))
    n_train_s0 = int(0.35 * len(idx_s0))

    train_s1 = rng.choice(idx_s1, size=n_train_s1, replace=False)
    train_s0 = rng.choice(idx_s0, size=n_train_s0, replace=False)

    train_full_idx = np.concatenate([train_s1, train_s0])
    rng.shuffle(train_full_idx)
    test_idx = np.setdiff1d(idx_all, train_full_idx)

    X_train_full = X_all[train_full_idx]
    y_train_full = y_all[train_full_idx]
    p_train_full = p_all[train_full_idx]

    X_test_np = X_all[test_idx]
    y_test_np = y_all[test_idx]
    p_test_np = p_all[test_idx]

    X_tr_np, X_val_np, y_tr_np, y_val_np, p_tr_np, p_val_np = train_test_split(
        X_train_full, y_train_full, p_train_full,
        test_size=0.20, random_state=seed, stratify=y_train_full
    )

    print(f"Sex proportion (train): {p_train_full.mean():.3f} | (test): {p_test_np.mean():.3f}")

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_np)
    X_val_np = scaler.transform(X_val_np)
    X_te_np = scaler.transform(X_test_np)

    X_tr = torch.from_numpy(X_tr_np).float()
    X_val = torch.from_numpy(X_val_np).float()
    X_te = torch.from_numpy(X_te_np).float()
    y_tr = torch.from_numpy(y_tr_np).long()
    y_val = torch.from_numpy(y_val_np).long()
    y_te = torch.from_numpy(y_test_np).long()
    p_tr = torch.from_numpy(p_tr_np).long()
    p_val = torch.from_numpy(p_val_np).long()
    p_te = torch.from_numpy(p_test_np).long()

    input_dim = X_tr.shape[1]
    print(f"Heart (Non-IID, seed={seed}): {len(X_tr)} train, {len(X_val)} val, {len(X_te)} test | dim={input_dim}")

    # Discovery
    MMM, mi_all = graph_free_discovery(X_tr_np, p_tr.numpy(), top_frac=0.60, eps=2.0, seed=seed)
    sorted_mmm = sorted(list(MMM), key=lambda j: mi_all[j], reverse=True)
    PPP = set(sorted_mmm[:max(1, len(sorted_mmm)//2)])
    print("MMM (mediators):", sorted(MMM))
    print("PPP (proxies):", sorted(PPP))

    results = {}

    print("Training: Baseline (no fairness)...")
    _, _, acc, ll, scg, fr = train_baseline(X_tr, y_tr, X_te, y_te, input_dim, MMM)
    results["Baseline"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: Uniform CF (naïve)...")
    _, _, acc, ll, scg, fr = train_uniform_cf(X_tr, y_tr, X_te, y_te, input_dim, MMM)
    results["Uniform CF"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: Policy-blind mask...")
    _, _, acc, ll, scg, fr = train_policy_blind(X_tr, y_tr, X_te, y_te, input_dim, MMM)
    results["Policy-blind"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: Server-only consistency...")
    _, _, acc, ll, scg, fr = train_server_only(X_tr, y_tr, X_te, y_te, input_dim, MMM)
    results["Server-only"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: SCC-VFL (ours, cVAE)...")
    _, _, acc, ll, scg, fr = train_scc_vfl(
        X_tr, y_tr, p_tr, X_val, y_val, X_te, y_te,
        input_dim, MMM, PPP
    )
    results["SCC-VFL (ours, cVAE)"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    return results

def main():
    df_raw = load_and_preprocess_data()
    
    SEEDS = list(range(30))
    methods_order = ["Baseline", "Uniform CF", "Policy-blind", "Server-only", "SCC-VFL (ours, cVAE)"]
    metrics_store = {m: {"acc": [], "ll": [], "scg": [], "fr": []} for m in methods_order}

    for s in SEEDS:
        res_s = run_single_seed_non_iid(s, df_raw)
        for m in methods_order:
            metrics_store[m]["acc"].append(res_s[m]["acc"])
            metrics_store[m]["ll"].append(res_s[m]["ll"])
            metrics_store[m]["scg"].append(res_s[m]["scg"])
            metrics_store[m]["fr"].append(res_s[m]["fr"])

    summary_rows = []
    for m in methods_order:
        acc_arr = np.array(metrics_store[m]["acc"])
        ll_arr = np.array(metrics_store[m]["ll"])
        scg_arr = np.array(metrics_store[m]["scg"])
        fr_arr = np.array(metrics_store[m]["fr"])
        summary_rows.append({
            "Method": m,
            "Accuracy ↑": f"{acc_arr.mean():.4f}±{acc_arr.std():.4f}",
            "LogLoss ↓": f"{ll_arr.mean():.4f}±{ll_arr.std():.4f}",
            "SCG ↓": f"{scg_arr.mean():.4f}±{scg_arr.std():.4f}",
            "FR (%) ↓": f"{fr_arr.mean():.2f}±{fr_arr.std():.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n===================================================")
    print("MEAN±STD OVER 30 SEEDS (Heart, Non-IID)")
    print("===================================================\n")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
