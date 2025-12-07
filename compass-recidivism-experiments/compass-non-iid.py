import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# ================================================================
# Configuration & Globals
# ================================================================
warnings.filterwarnings("ignore")
BASE_SEED = 42
HIDDEN = 32
LATENT_DIM = 16

CF_SCALE_TRAIN = 0.5
CF_SCALE_TRAIN_SCC = 0.25
CF_SCALE_EVAL = 2.5
EXTRA_NOISE = 0.2
EPOCHS_BASE = 80

# Deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================================================================
# Data Loading & Preprocessing
# ================================================================
def load_compass_data():
    url = 'https://storage.googleapis.com/what-if-tool-resources/computefest2019/cox-violent-parsed_filt.csv'
    print(f"Downloading COMPAS dataset from {url}...")
    df = pd.read_csv(url)
    
    # Target
    y_all = df["event"].astype(int).values

    # Sensitive attribute: binary (African-American = 1, others = 0)
    race_series = df["race"].fillna("Unknown").astype(str)
    p_all = (race_series == "African-American").astype(int).values

    # Feature matrix: drop race and event
    X_df = df.drop(columns=["race", "event"]).copy()

    # Handle NaNs and encode categorical columns
    for col in X_df.columns:
        if X_df[col].dtype == "object":
            X_df[col] = X_df[col].fillna("__NA__").astype(str)
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col])
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
            if X_df[col].isna().all():
                X_df[col] = 0.0
            else:
                median = X_df[col].median()
                X_df[col] = X_df[col].fillna(median)

    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_all = X_df.astype(float).values
    
    return X_all, y_all, p_all

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

class CvaeGenerator(nn.Module):
    def __init__(self, input_dim, MMM, z_dim=LATENT_DIM, hidden=HIDDEN, scale=CF_SCALE_TRAIN_SCC):
        super().__init__()
        self.MMM = sorted(list(MMM))
        self.d_m = len(self.MMM)
        self.z_dim = z_dim
        self.scale = scale

        self.fc_enc = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)
        self.emb_s = nn.Embedding(2, 4)
        self.fc_dec1 = nn.Linear(z_dim + 4, hidden)
        self.fc_dec2 = nn.Linear(hidden, self.d_m)

    def encode(self, x):
        h = torch.relu(self.fc_enc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, s_cf):
        s_emb = self.emb_s(s_cf)
        h = torch.relu(self.fc_dec1(torch.cat([z, s_emb], dim=1)))
        delta_m = torch.tanh(self.fc_dec2(h)) * self.scale
        return delta_m

    def forward(self, x, s_cf):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        delta_m = self.decode(z, s_cf)
        x_cf = x.clone()
        for i, j in enumerate(self.MMM):
            x_cf[:, j] = x_cf[:, j] + delta_m[:, i]
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return x_cf, delta_m, kld

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

def grl(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

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
def eval_model(enc, clf, X, y, temp_scaler=None):
    enc.eval(); clf.eval()
    z = clf(enc(X))
    if temp_scaler is not None:
        z = temp_scaler(z)
    prob = torch.softmax(z, dim=1)[:, 1].cpu().numpy()
    pred = z.argmax(dim=1).cpu().numpy()
    return accuracy_score(y.cpu().numpy(), pred), log_loss(y.cpu().numpy(), prob, labels=[0, 1])

@torch.no_grad()
def compute_scg_fr(enc, clf, gen, X, p=None, flip_s=False, add_noise=True):
    enc.eval(); clf.eval(); gen.eval()
    z0 = clf(enc(X))
    y0 = z0.argmax(1)

    if p is None:
        X_cf, _ = gen(X)
    else:
        s_cf = 1 - p if flip_s else p
        X_cf, _, _ = gen(X, s_cf)

    if add_noise:
        X_cf = X_cf + EXTRA_NOISE * torch.randn_like(X_cf)

    z1 = clf(enc(X_cf))
    y1 = z1.argmax(1)

    scg = torch.norm(z0 - z1, dim=1).mean().item()
    fr = (y0 != y1).float().mean().item() * 100.0
    return scg, fr

def graph_free_discovery(
    X_np,
    p_np,
    top_frac: float = 0.60,
    eps: float = 2.0,
    seed: int = 42,
    cor_flag: bool = False,
):
    """
    Graph-free mask discovery.

    When cor_flag = False:
        - Behaves like the original implementation: rank by mutual information
          I(x_j; s) and take the top 'top_frac' as mediators (MMM), with a
          small random flip probability.

    When cor_flag = True:
        - Still computes MI(x_j; s) for reporting / compatibility.
        - Additionally computes DP-noisy HISC and ΔP per feature using
          a simple Laplace mechanism.
        - Uses a combined DP score (MI + HISC + ΔP) to rank features
          and select MMM.
    """
    rng = np.random.RandomState(seed)

    # ----------------------------------
    # 1) Mutual information (non-DP)
    # ----------------------------------
    mi = mutual_info_classif(X_np, p_np, random_state=seed)

    n, d = X_np.shape
    k = max(1, int(np.ceil(d * top_frac)))

    if not cor_flag:
        # Original behavior: MI ranking + one random DP-style flip
        order = np.argsort(mi)[::-1]
        MMM = set(order[:k])

        # Very simple "DP-ish" flip as in the original code
        flip_prob = min(1.0, 1.0 / (len(X_np) * eps))
        if rng.rand() < flip_prob:
            others = list(set(range(d)) - MMM)
            if len(others) > 0:
                MMM.add(int(rng.choice(others)))
        return MMM, mi

    # ============================================================
    # cor_flag = True: DP HISC / ΔP-based refinement
    # ============================================================

    # Helper: binarize a feature for ΔP using median threshold
    def _binarize_feature(x_col: np.ndarray) -> np.ndarray:
        x = x_col.astype(float)
        # Guard against degenerate / constant columns
        if np.allclose(x.min(), x.max()):
            return np.zeros_like(x, dtype=int)
        med = np.median(x)
        return (x > med).astype(int)

    # Helper: DP ΔP = |P(x=1 | s=1) - P(x=1 | s=0)| with Laplace noise
    def _dp_delta_p(x_bin: np.ndarray, s: np.ndarray, eps_local: float) -> float:
        # Counts: n_{s,v} for s∈{0,1}, v∈{0,1}
        counts = np.zeros((2, 2), dtype=float)
        for ss in (0, 1):
            mask_s = (s == ss)
            for v in (0, 1):
                counts[ss, v] = np.sum(mask_s & (x_bin == v))

        # Add Laplace noise on counts (sensitivity 1)
        scale = 1.0 / eps_local
        noise = rng.laplace(loc=0.0, scale=scale, size=counts.shape)
        noisy_counts = counts + noise
        noisy_counts = np.clip(noisy_counts, 1e-6, None)

        # P(x=1 | s)
        probs = noisy_counts / noisy_counts.sum(axis=1, keepdims=True)
        p1_s0 = probs[0, 1]
        p1_s1 = probs[1, 1]
        return float(abs(p1_s1 - p1_s0))

    # Helper: DP HISC-like covariance magnitude between x (scaled to [0,1]) and s
    def _dp_hisc(x_col: np.ndarray, s: np.ndarray, eps_local: float) -> float:
        x = x_col.astype(float)
        # Scale x to [0,1] to bound sensitivity
        xmin, xmax = x.min(), x.max()
        if np.allclose(xmin, xmax):
            x_scaled = np.zeros_like(x)
        else:
            x_scaled = (x - xmin) / (xmax - xmin)

        s_float = s.astype(float)
        n_local = len(x_scaled)

        # Sensitivity of each mean is at most 1/n over [0,1]
        scale = 1.0 / (n_local * eps_local)

        Ex = x_scaled.mean() + rng.laplace(0.0, scale)
        Es = s_float.mean() + rng.laplace(0.0, scale)
        Exs = (x_scaled * s_float).mean() + rng.laplace(0.0, scale)

        cov_dp = Exs - Ex * Es
        return float(abs(cov_dp))

    # ----------------------------------
    # 2) DP HISC and ΔP per feature
    # ----------------------------------
    delta_p_dp = np.zeros(d, dtype=float)
    hisc_dp = np.zeros(d, dtype=float)

    # Split budget crudely between HISC and ΔP
    eps_hisc = eps / 2.0
    eps_dp = eps / 2.0

    for j in range(d):
        x_j = X_np[:, j]
        x_bin = _binarize_feature(x_j)

        delta_p_dp[j] = _dp_delta_p(x_bin, p_np, eps_local=eps_dp)
        hisc_dp[j] = _dp_hisc(x_j, p_np, eps_local=eps_hisc)

    # ----------------------------------
    # 3) Combined DP ranking score
    # ----------------------------------
    # Normalize components to comparable scales
    mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)
    dp_norm = (delta_p_dp - delta_p_dp.min()) / (delta_p_dp.max() - delta_p_dp.min() + 1e-8)
    hisc_norm = (hisc_dp - hisc_dp.min()) / (hisc_dp.max() - hisc_dp.min() + 1e-8)

    # Combined score: higher means "stronger descendant / proxy signal"
    # You can tune the weights if you like; 1:1:1 keeps it simple.
    score = mi_norm + dp_norm + hisc_norm

    order = np.argsort(score)[::-1]
    MMM = set(order[:k])

    # Optional small DP-style random flip (as in original)
    flip_prob = min(1.0, 1.0 / (len(X_np) * eps))
    if rng.rand() < flip_prob:
        others = list(set(range(d)) - MMM)
        if len(others) > 0:
            MMM.add(int(rng.choice(others)))
    return MMM, mi

# ================================================================
# Training Functions
# ================================================================
def train_baseline(X_tr, y_tr, X_te, y_te, eval_gen, epochs=EPOCHS_BASE, lr=0.005):
    input_dim = X_tr.shape[1]
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train()
        z = clf(enc(X_tr))
        loss = CE_LOSS(z, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen, X_te, add_noise=True)
    return enc, clf, acc, ll, scg, fr

def train_uniform_cf(X_tr, y_tr, X_te, y_te, eval_gen, epochs=EPOCHS_BASE, lr=0.005):
    input_dim = X_tr.shape[1]
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    gen = MaskedGenerator(input_dim, MMM=set(range(input_dim)), scale=CF_SCALE_TRAIN)
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train(); gen.train()
        z0 = clf(enc(X_tr))
        loss_main = CE_LOSS(z0, y_tr)
        X_cf, _ = gen(X_tr)
        z1 = clf(enc(X_cf))
        loss_cons = ((z0 - z1) ** 2).mean()
        loss = loss_main + 0.1 * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen, X_te, add_noise=True)
    return enc, clf, acc, ll, scg, fr

def train_policy_blind(X_tr, y_tr, X_te, y_te, MMM, eval_gen, epochs=EPOCHS_BASE, lr=0.005):
    input_dim = X_tr.shape[1]
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    gen = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_TRAIN)
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train(); gen.train()
        z0 = clf(enc(X_tr))
        loss_main = CE_LOSS(z0, y_tr)
        X_cf, _ = gen(X_tr)
        z1 = clf(enc(X_cf))
        loss_cons = ((z0 - z1) ** 2).mean()
        loss = loss_main + 0.1 * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen, X_te, add_noise=True)
    return enc, clf, acc, ll, scg, fr

def train_server_only(X_tr, y_tr, X_te, y_te, MMM, eval_gen, epochs=EPOCHS_BASE, lr=0.005):
    input_dim = X_tr.shape[1]
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train()
        z0 = clf(enc(X_tr))
        loss_main = CE_LOSS(z0, y_tr)
        X_flip = X_tr.clone()
        for j in MMM:
            X_flip[:, j] = X_flip[:, j] + 0.3 * torch.randn_like(X_flip[:, j])
        z1 = clf(enc(X_flip))
        loss_cons = ((z0 - z1) ** 2).mean()
        loss = loss_main + 0.1 * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll = eval_model(enc, clf, X_te, y_te)
    scg, fr = compute_scg_fr(enc, clf, eval_gen, X_te, add_noise=True)
    return enc, clf, acc, ll, scg, fr

def train_scc_vfl(X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, p_te, MMM, epochs=150):
    input_dim = X_tr.shape[1]
    enc = SimpleEncoder(input_dim)
    clf = Classifier()
    gen = CvaeGenerator(input_dim, MMM, z_dim=LATENT_DIM, hidden=HIDDEN, scale=CF_SCALE_TRAIN_SCC)
    adv = LightAdversary()

    opt_main = optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()), lr=0.005)
    opt_adv = optim.Adam(adv.parameters(), lr=0.003)
    stopper = EarlyStopper(patience=25, min_delta=1e-4)
    best_state = None
    WARMUP = 25

    for epoch in range(epochs):
        enc.train(); clf.train(); gen.train(); adv.train()

        h = enc(X_tr)
        z = clf(h)
        loss_cls = CE_LOSS(z, y_tr)

        s_cf = 1 - p_tr
        X_cf, d, kld = gen(X_tr, s_cf)
        z_cf = clf(enc(X_cf))
        loss_cons = ((z - z_cf) ** 2).mean()
        loss_reg = (d ** 2).mean()

        lam = 0.0 if epoch < WARMUP else 0.4
        h_rev = grl(h, lambd=0.1)
        adv_logits_main = adv(h_rev).squeeze()
        loss_adv_conf = BCE_LOSS(adv_logits_main, torch.full_like(p_tr.float(), 0.5))

        loss = loss_cls + lam * 0.8 * loss_cons + 0.002 * loss_reg + 0.008 * loss_adv_conf + 0.001 * kld
        opt_main.zero_grad(); loss.backward(); opt_main.step()

        if epoch % 2 == 0:
            with torch.no_grad():
                h_det = enc(X_tr)
            adv_logits2 = adv(h_det).squeeze()
            loss_adv = BCE_LOSS(adv_logits2, p_tr.float())
            opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()

        enc.eval(); clf.eval(); gen.eval()
        acc_v, ll_v = eval_model(enc, clf, X_val, y_val)
        scg_v, fr_v = compute_scg_fr(enc, clf, gen, X_val, p=p_val, flip_s=True, add_noise=False)
        score = ll_v + 0.15 * scg_v + 0.08 * (fr_v / 100.0)
        if stopper.step(score):
            best_state = {"enc": enc.state_dict(), "clf": clf.state_dict(), "gen": gen.state_dict()}
        if stopper.stop():
            break

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

    acc, ll = eval_model(enc, clf, X_te, y_te, temp_scaler=temp)
    scg, fr = compute_scg_fr(enc, clf, gen, X_te, p=p_te, flip_s=True, add_noise=False)
    return enc, clf, acc, ll, scg, fr

# ================================================================
# Main Execution Loop
# ================================================================
def run_single_seed_non_iid(seed: int, X_all, y_all, p_all):
    print(f"\n================ NON-IID SEED {seed} ================\n")
    np.random.seed(seed)
    torch.manual_seed(seed)

    N = len(X_all)
    idx_all = np.arange(N)
    rng = np.random.RandomState(seed)

    aa_idx = np.where(p_all == 1)[0]
    oth_idx = np.where(p_all == 0)[0]

    n_train_aa = int(0.85 * len(aa_idx))
    n_train_oth = int(0.35 * len(oth_idx))

    train_aa = rng.choice(aa_idx, size=n_train_aa, replace=False)
    train_oth = rng.choice(oth_idx, size=n_train_oth, replace=False)

    train_full_idx = np.concatenate([train_aa, train_oth])
    rng.shuffle(train_full_idx)
    test_idx = np.setdiff1d(idx_all, train_full_idx)

    X_train_full = X_all[train_full_idx]
    y_train_full = y_all[train_full_idx]
    p_train_full = p_all[train_full_idx]

    X_test_np = X_all[test_idx]
    y_test_np = y_all[test_idx]
    p_test_np = p_all[test_idx]

    print(f"NON-IID race proportion (train): {p_train_full.mean():.3f} | (test): {p_test_np.mean():.3f}")

    X_tr_np, X_val_np, y_tr_np, y_val_np, p_tr_np, p_val_np = train_test_split(
        X_train_full, y_train_full, p_train_full,
        test_size=0.20, random_state=seed, stratify=y_train_full
    )

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
    print(f"COMPAS Cox (Non-IID, seed={seed}): {len(X_tr)} train, {len(X_val)} val, {len(X_te)} test | dim={input_dim}")

    MMM, mi_all = graph_free_discovery(X_tr_np, p_tr.numpy(), top_frac=0.60, eps=2.0, seed=seed)
    sorted_mmm = sorted(list(MMM), key=lambda j: mi_all[j], reverse=True)
    PPP = set(sorted_mmm[:max(1, len(sorted_mmm)//2)])
    print("MMM (mediators):", sorted(MMM))
    print("PPP (proxies):", sorted(PPP))

    eval_gen = MaskedGenerator(input_dim, MMM, scale=CF_SCALE_EVAL)
    results = {}

    print("Training: Baseline (no fairness)...")
    _, _, acc, ll, scg, fr = train_baseline(X_tr, y_tr, X_te, y_te, eval_gen)
    results["Baseline"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: Uniform CF (naive)...")
    _, _, acc, ll, scg, fr = train_uniform_cf(X_tr, y_tr, X_te, y_te, eval_gen)
    results["Uniform CF"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: Policy blind mask...")
    _, _, acc, ll, scg, fr = train_policy_blind(X_tr, y_tr, X_te, y_te, MMM, eval_gen)
    results["Policy blind"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: Server only consistency...")
    _, _, acc, ll, scg, fr = train_server_only(X_tr, y_tr, X_te, y_te, MMM, eval_gen)
    results["Server only"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    print("Training: SCC VFL (ours)...")
    _, _, acc, ll, scg, fr = train_scc_vfl(X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, p_te, MMM)
    results["SCC VFL (ours)"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    return results

def main():
    X_all, y_all, p_all = load_compass_data()
    
    SEEDS = list(range(30))
    methods_order = ["Baseline", "Uniform CF", "Policy blind", "Server only", "SCC VFL (ours)"]
    metrics_store = {m: {"acc": [], "ll": [], "scg": [], "fr": []} for m in methods_order}

    for s in SEEDS:
        res_s = run_single_seed_non_iid(s, X_all, y_all, p_all)
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
    print("MEAN±STD OVER 30 SEEDS (COMPAS Cox, Non-IID)")
    print("===================================================\n")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
