import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import mutual_info_classif
from ucimlrepo import fetch_ucirepo

# ================================================================
# Configuration & Globals
# ================================================================
warnings.filterwarnings("ignore")
SEED_GLOBAL = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deterministic settings
np.random.seed(SEED_GLOBAL)
torch.manual_seed(SEED_GLOBAL)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================================================================
# Data Loading
# ================================================================
def load_german_credit_data():
    print("Fetching German Credit dataset from UCI...")
    dataset = fetch_ucirepo(id=144)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    return X, y

# ================================================================
# Models & Helpers
# ================================================================
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, h):
        return self.fc(h)

class MaskedGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, MMM=None, cf_scale=0.25):
        super().__init__()
        self.input_dim = input_dim
        self.MMM = sorted(list(MMM)) if MMM is not None else list(range(input_dim))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(self.MMM))
        self.act = nn.ReLU()
        self.cf_scale = cf_scale
    def forward(self, x):
        h = self.act(self.fc1(x))
        delta = torch.tanh(self.fc2(h)) * self.cf_scale
        x_cf = x.clone()
        for i, j in enumerate(self.MMM):
            x_cf[:, j] = x_cf[:, j] + delta[:, i]
        return x_cf, delta

class LightAdversary(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
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

class SmoothCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, logits, target):
        n_class = logits.size(1)
        logprobs = self.log_softmax(logits)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=1))

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

class EarlyStopper:
    def __init__(self, patience=35, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0
    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            return True
        self.count += 1
        return False
    def stop(self):
        return self.count >= self.patience

# Global Loss Functions
CE_LOSS = SmoothCE(smoothing=0.05)
BCE_LOGITS = nn.BCEWithLogitsLoss()

@torch.no_grad()
def eval_model(encoder, classifier, X_te, y_te, temp_scaler=None):
    encoder.eval(); classifier.eval()
    h = encoder(X_te)
    logits = classifier(h)
    if temp_scaler is not None:
        logits = temp_scaler(logits)
    prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_te.cpu().numpy(), pred)
    ll = log_loss(y_te.cpu().numpy(), prob, labels=[0, 1])
    return acc, ll, logits

@torch.no_grad()
def compute_scg_fr(encoder, classifier, gen, X_eval, temp_scaler=None):
    encoder.eval(); classifier.eval(); gen.eval()
    h0 = encoder(X_eval); z0 = classifier(h0)
    if temp_scaler is not None:
        z0 = temp_scaler(z0)
    y0 = z0.argmax(dim=1)

    X_cf, _ = gen(X_eval)
    h1 = encoder(X_cf); z1 = classifier(h1)
    if temp_scaler is not None:
        z1 = temp_scaler(z1)
    y1 = z1.argmax(dim=1)

    scg = torch.norm(z0 - z1, dim=1).mean().item()
    fr = (y0 != y1).float().mean().item() * 100.0
    return scg, fr

def graph_free_discovery(X_np, p_np, top_frac=0.33, eps=2.0, seed=42):
    n, d = X_np.shape
    mi = mutual_info_classif(X_np, p_np, discrete_features=False, random_state=seed)
    order = np.argsort(mi)[::-1]
    k = max(1, int(np.ceil(d * top_frac)))
    MMM = set(order[:k])
    flip_prob = max(0.0, min(1.0, 1.0 / (n * max(eps, 1e-6))))
    if np.random.rand() < flip_prob:
        others = list(set(range(d)) - MMM)
        if len(others) > 0:
            MMM.add(int(np.random.choice(others)))
    return MMM, mi

# ================================================================
# Training Functions (Ablations)
# ================================================================
def train_scc_vfl_best(
    X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM, PPP,
    epochs=300, lr_main=0.015, lr_adv=0.005, weight_decay=5e-4,
    lam_cons_max=1.2, lam_adv=0.03, lam_gen=0.01, grad_rev_lambda=0.25,
    warmup=40, cosine=True,
):
    enc = SimpleEncoder(X_tr.shape[1], hidden_dim=64, dropout=0.05)
    clf = Classifier(hidden_dim=64)
    gen = MaskedGenerator(X_tr.shape[1], hidden_dim=64, MMM=MMM, cf_scale=0.20)
    adv = LightAdversary(hidden_dim=64)

    opt_main = optim.AdamW(
        list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()),
        lr=lr_main, weight_decay=weight_decay,
    )
    opt_adv = optim.AdamW(adv.parameters(), lr=lr_adv, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=max(1, epochs - warmup)) if cosine else None
    stopper = EarlyStopper(patience=35, min_delta=1e-4)
    best_state = None

    for epoch in range(epochs):
        enc.train(); clf.train(); gen.train(); adv.train()

        if epoch < warmup:
            lam_cons = 0.0
        else:
            t = (epoch - warmup) / max(1, (epochs - warmup))
            lam_cons = lam_cons_max * (0.5 - 0.5 * np.cos(np.pi * t))

        h0 = enc(X_tr); z0 = clf(h0)
        loss_cls = CE_LOSS(z0, y_tr)

        X_cf, delta = gen(X_tr)
        h1 = enc(X_cf); z1 = clf(h1)
        y0 = z0.argmax(dim=1); y1 = z1.argmax(dim=1)
        flip_mask = (y0 != y1).float()
        loss_cons = (flip_mask * torch.sum((z0 - z1) ** 2, dim=1)).mean()
        loss_gen_reg = (delta ** 2).mean()

        h_grl = grad_reverse(h0, grad_rev_lambda)
        adv_logits_main = adv(h_grl)
        loss_adv_confuse = BCE_LOGITS(adv_logits_main.squeeze(), 0.5 * torch.ones_like(p_tr.float()))

        loss = loss_cls + lam_cons * loss_cons + lam_gen * loss_gen_reg + lam_adv * loss_adv_confuse

        opt_main.zero_grad(); loss.backward(); opt_main.step()

        if epoch % 2 == 0:
            with torch.no_grad(): h_det = enc(X_tr)
            adv_logits = adv(h_det.detach())
            loss_adv = BCE_LOGITS(adv_logits.squeeze(), p_tr.float())
            opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()

        if scheduler is not None and epoch >= warmup: scheduler.step()

        enc.eval(); clf.eval(); gen.eval()
        with torch.no_grad():
            acc_val, ll_val, _ = eval_model(enc, clf, X_val, y_val)
            scg_val, fr_val = compute_scg_fr(enc, clf, gen, X_val)
            score = ll_val + 0.1 * scg_val + 0.1 * (fr_val / 100.0)
        if stopper.step(score):
            best_state = {"enc": enc.state_dict(), "clf": clf.state_dict(), "gen": gen.state_dict()}
        if stopper.stop(): break

    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        clf.load_state_dict(best_state["clf"])
        gen.load_state_dict(best_state["gen"])

    temp = TemperatureScaler()
    opt_t = optim.LBFGS(temp.parameters(), lr=0.1, max_iter=50)
    def _closure():
        opt_t.zero_grad()
        logits = temp(clf(enc(X_val)))
        loss = nn.CrossEntropyLoss()(logits, y_val)
        loss.backward()
        return loss
    opt_t.step(_closure)

    acc, ll, _ = eval_model(enc, clf, X_te, y_te, temp_scaler=temp)
    scg, fr = compute_scg_fr(enc, clf, gen, X_te, temp_scaler=temp)
    return acc, ll, scg, fr, enc, clf, gen, temp

def train_scc_no_consistency(
    X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM, PPP,
    epochs=300, lr_main=0.015, lr_adv=0.005, weight_decay=5e-4,
    lam_adv=0.03, lam_gen=0.01, grad_rev_lambda=0.25,
    warmup=40, cosine=True,
):
    enc = SimpleEncoder(X_tr.shape[1], hidden_dim=64, dropout=0.05)
    clf = Classifier(hidden_dim=64)
    gen = MaskedGenerator(X_tr.shape[1], hidden_dim=64, MMM=MMM, cf_scale=0.20)
    adv = LightAdversary(hidden_dim=64)

    opt_main = optim.AdamW(
        list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()),
        lr=lr_main, weight_decay=weight_decay,
    )
    opt_adv = optim.AdamW(adv.parameters(), lr=lr_adv, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=max(1, epochs - warmup)) if cosine else None
    stopper = EarlyStopper(patience=35, min_delta=1e-4)
    best_state = None

    for epoch in range(epochs):
        enc.train(); clf.train(); gen.train(); adv.train()

        h0 = enc(X_tr); z0 = clf(h0)
        loss_cls = CE_LOSS(z0, y_tr)

        X_cf, delta = gen(X_tr)
        loss_gen_reg = (delta ** 2).mean()

        h_grl = grad_reverse(h0, grad_rev_lambda)
        adv_logits_main = adv(h_grl)
        loss_adv_confuse = BCE_LOGITS(adv_logits_main.squeeze(), 0.5 * torch.ones_like(p_tr.float()))

        loss = loss_cls + lam_gen * loss_gen_reg + lam_adv * loss_adv_confuse

        opt_main.zero_grad(); loss.backward(); opt_main.step()

        if epoch % 2 == 0:
            with torch.no_grad(): h_det = enc(X_tr)
            adv_logits = adv(h_det.detach())
            loss_adv = BCE_LOGITS(adv_logits.squeeze(), p_tr.float())
            opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()

        if scheduler is not None and epoch >= warmup: scheduler.step()

        enc.eval(); clf.eval(); gen.eval()
        with torch.no_grad():
            acc_val, ll_val, _ = eval_model(enc, clf, X_val, y_val)
            scg_val, fr_val = compute_scg_fr(enc, clf, gen, X_val)
            score = ll_val + 0.1 * scg_val + 0.1 * (fr_val / 100.0)
        if stopper.step(score):
            best_state = {"enc": enc.state_dict(), "clf": clf.state_dict(), "gen": gen.state_dict()}
        if stopper.stop(): break

    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        clf.load_state_dict(best_state["clf"])
        gen.load_state_dict(best_state["gen"])

    temp = TemperatureScaler()
    opt_t = optim.LBFGS(temp.parameters(), lr=0.1, max_iter=50)
    def _closure():
        opt_t.zero_grad()
        logits = temp(clf(enc(X_val)))
        loss = nn.CrossEntropyLoss()(logits, y_val)
        loss.backward()
        return loss
    opt_t.step(_closure)

    acc, ll, _ = eval_model(enc, clf, X_te, y_te, temp_scaler=temp)
    scg, fr = compute_scg_fr(enc, clf, gen, X_te, temp_scaler=temp)
    return acc, ll, scg, fr, enc, clf, gen, temp

def train_scc_no_generator(
    X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM, PPP,
    epochs=300, lr_main=0.015, lr_adv=0.005, weight_decay=5e-4,
    lam_adv=0.03, grad_rev_lambda=0.25,
    warmup=40, cosine=True,
):
    enc = SimpleEncoder(X_tr.shape[1], hidden_dim=64, dropout=0.05)
    clf = Classifier(hidden_dim=64)
    adv = LightAdversary(hidden_dim=64)

    opt_main = optim.AdamW(list(enc.parameters()) + list(clf.parameters()), lr=lr_main, weight_decay=weight_decay)
    opt_adv = optim.AdamW(adv.parameters(), lr=lr_adv, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=max(1, epochs - warmup)) if cosine else None
    stopper = EarlyStopper(patience=35, min_delta=1e-4)
    best_state = None

    for epoch in range(epochs):
        enc.train(); clf.train(); adv.train()

        h0 = enc(X_tr); z0 = clf(h0)
        loss_cls = CE_LOSS(z0, y_tr)

        h_grl = grad_reverse(h0, grad_rev_lambda)
        adv_logits_main = adv(h_grl)
        loss_adv_confuse = BCE_LOGITS(adv_logits_main.squeeze(), 0.5 * torch.ones_like(p_tr.float()))

        loss = loss_cls + lam_adv * loss_adv_confuse

        opt_main.zero_grad(); loss.backward(); opt_main.step()

        if epoch % 2 == 0:
            with torch.no_grad(): h_det = enc(X_tr)
            adv_logits = adv(h_det.detach())
            loss_adv = BCE_LOGITS(adv_logits.squeeze(), p_tr.float())
            opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()

        if scheduler is not None and epoch >= warmup: scheduler.step()

        enc.eval(); clf.eval()
        with torch.no_grad():
            acc_val, ll_val, _ = eval_model(enc, clf, X_val, y_val)
            eval_gen = MaskedGenerator(X_tr.shape[1], hidden_dim=64, MMM=MMM, cf_scale=0.20)
            scg_val, fr_val = compute_scg_fr(enc, clf, eval_gen, X_val)
            score = ll_val + 0.1 * scg_val + 0.1 * (fr_val / 100.0)
        if stopper.step(score):
            best_state = {"enc": enc.state_dict(), "clf": clf.state_dict()}
        if stopper.stop(): break

    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        clf.load_state_dict(best_state["clf"])

    temp = TemperatureScaler()
    opt_t = optim.LBFGS(temp.parameters(), lr=0.1, max_iter=50)
    def _closure():
        opt_t.zero_grad()
        logits = temp(clf(enc(X_val)))
        loss = nn.CrossEntropyLoss()(logits, y_val)
        loss.backward()
        return loss
    opt_t.step(_closure)

    eval_gen_test = MaskedGenerator(X_tr.shape[1], hidden_dim=64, MMM=MMM, cf_scale=0.20)
    acc, ll, _ = eval_model(enc, clf, X_te, y_te, temp_scaler=temp)
    scg, fr = compute_scg_fr(enc, clf, eval_gen_test, X_te, temp_scaler=temp)
    return acc, ll, scg, fr, enc, clf, eval_gen_test, temp

# ================================================================
# Main Execution Loop
# ================================================================
def run_single_seed(seed: int, X_raw: pd.DataFrame, Y_raw: pd.DataFrame):
    print(f"\n================ German Credit IID SEED {seed} ================\n")
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = X_raw.copy()
    y = (Y_raw.values.ravel() == 2).astype(int)

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    age_col = "Attribute13"
    protected = (X[age_col] >= 25).astype(int)
    X = X.drop(columns=[age_col])

    X_train_full, X_test_df, y_train_full, y_test_np, p_train_full, p_test_s = train_test_split(
        X, y, protected, test_size=0.30, random_state=seed, stratify=y,
    )
    X_tr_df, X_val_df, y_tr_np, y_val_np, p_tr_s, p_val_s = train_test_split(
        X_train_full, y_train_full, p_train_full, test_size=0.20, random_state=seed, stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_df.to_numpy(dtype=float))
    X_val_np = scaler.transform(X_val_df.to_numpy(dtype=float))
    X_te_np = scaler.transform(X_test_df.to_numpy(dtype=float))

    X_tr = torch.from_numpy(X_tr_np).float()
    X_val = torch.from_numpy(X_val_np).float()
    X_te = torch.from_numpy(X_te_np).float()
    y_tr = torch.from_numpy(y_tr_np).long()
    y_val = torch.from_numpy(y_val_np).long()
    y_te = torch.from_numpy(y_test_np).long()
    p_tr = torch.from_numpy(np.asarray(p_tr_s, dtype=int)).long()
    p_val = torch.from_numpy(np.asarray(p_val_s, dtype=int)).long()
    p_te = torch.from_numpy(np.asarray(p_test_s, dtype=int)).long()

    input_dim = X_tr.shape[1]
    print(f"Dataset: {X_tr.shape[0]} train, {X_val.shape[0]} val, {X_te.shape[0]} test | dim={input_dim}")

    MMM, mi_all = graph_free_discovery(X_tr_np, p_tr.numpy(), top_frac=0.33, eps=2.0, seed=seed)
    mmm_sorted = sorted(list(MMM), key=lambda j: mi_all[j], reverse=True)
    PPP = set(mmm_sorted[:max(1, len(mmm_sorted) // 2)])
    MMM_all = set(range(input_dim))
    PPP_all = set(range(input_dim))

    results = {}

    print("Training: SCC-VFL (full)...")
    acc, ll, scg, fr, _, _, _, _ = train_scc_vfl_best(
        X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM, PPP,
    )
    results["SCC-VFL (full)"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: SCC-VFL w/o mask discovery (MMM=all)...")
    acc, ll, scg, fr, _, _, _, _ = train_scc_vfl_best(
        X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM_all, PPP_all,
    )
    results["SCC-VFL w/o mask discovery"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: SCC-VFL w/o generator...")
    acc, ll, scg, fr, _, _, _, _ = train_scc_no_generator(
        X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM, PPP,
    )
    results["SCC-VFL w/o generator"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: SCC-VFL w/o consistency penalty...")
    acc, ll, scg, fr, _, _, _, _ = train_scc_no_consistency(
        X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM, PPP,
    )
    results["SCC-VFL w/o consistency"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    return results

def main():
    X_raw, Y_raw = load_german_credit_data()
    
    SEEDS = list(range(30))
    methods_order = [
        "SCC-VFL (full)",
        "SCC-VFL w/o mask discovery",
        "SCC-VFL w/o generator",
        "SCC-VFL w/o consistency",
    ]

    metrics_store = {m: {"acc": [], "ll": [], "scg": [], "fr": []} for m in methods_order}

    for s in SEEDS:
        res_s = run_single_seed(s, X_raw, Y_raw)
        for m in methods_order:
            metrics_store[m]["acc"].append(res_s[m]["acc"])
            metrics_store[m]["ll"].append(res_s[m]["ll"])
            metrics_store[m]["scg"].append(res_s[m]["scg"])
            metrics_store[m]["fr"].append(res_s[m]["fr"])

    summary_rows = []
    for m in methods_order:
        acc_arr = np.array(metrics_store[m]["acc"])
        ll_arr  = np.array(metrics_store[m]["ll"])
        scg_arr = np.array(metrics_store[m]["scg"])
        fr_arr  = np.array(metrics_store[m]["fr"])
        summary_rows.append({
            "Method": m,
            "Accuracy ↑": f"{acc_arr.mean():.4f}±{acc_arr.std():.4f}",
            "LogLoss ↓":  f"{ll_arr.mean():.4f}±{ll_arr.std():.4f}",
            "SCG ↓":      f"{scg_arr.mean():.4f}±{scg_arr.std():.4f}",
            "FR (%) ↓":   f"{fr_arr.mean():.2f}±{fr_arr.std():.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n===================================================")
    print("ABLATION: MEAN±STD OVER 30 SEEDS (German Credit, IID)")
    print("===================================================\n")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
