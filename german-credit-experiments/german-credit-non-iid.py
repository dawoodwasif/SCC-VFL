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

# Set deterministic behavior
np.random.seed(SEED_GLOBAL)
torch.manual_seed(SEED_GLOBAL)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================================================================
# Models and Helpers
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
    """Edits only MMM with bounded magnitude."""
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
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
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

# ================================================================
# Training Routines
# ================================================================
def train_baseline(X_tr, y_tr, X_te, y_te, epochs=120, lr=0.01):
    enc = SimpleEncoder(X_tr.shape[1])
    clf = Classifier()
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train()
        h = enc(X_tr); z = clf(h)
        loss = CE_LOSS(z, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll, _ = eval_model(enc, clf, X_te, y_te)
    return acc, ll, enc, clf

def train_uniform_cf(X_tr, y_tr, X_te, y_te, epochs=120, lr=0.01, lam_cons=0.2):
    enc = SimpleEncoder(X_tr.shape[1])
    clf = Classifier()
    gen = MaskedGenerator(X_tr.shape[1], MMM=set(range(X_tr.shape[1])), cf_scale=0.30)
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train(); gen.train()
        h0 = enc(X_tr); z0 = clf(h0)
        loss_main = CE_LOSS(z0, y_tr)
        X_cf, _ = gen(X_tr); h1 = enc(X_cf); z1 = clf(h1)
        loss_cons = torch.mean((z0 - z1) ** 2)
        loss = loss_main + lam_cons * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll, _ = eval_model(enc, clf, X_te, y_te)
    return acc, ll, enc, clf, gen

def train_policy_blind_mask(X_tr, y_tr, X_te, y_te, MMM, epochs=120, lr=0.01, lam_cons=0.3):
    enc = SimpleEncoder(X_tr.shape[1])
    clf = Classifier()
    gen = MaskedGenerator(X_tr.shape[1], MMM=MMM, cf_scale=0.30)
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train(); gen.train()
        h0 = enc(X_tr); z0 = clf(h0)
        loss_main = CE_LOSS(z0, y_tr)
        X_cf, _ = gen(X_tr); h1 = enc(X_cf); z1 = clf(h1)
        loss_cons = torch.mean((z0 - z1) ** 2)
        loss = loss_main + lam_cons * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll, _ = eval_model(enc, clf, X_te, y_te)
    return acc, ll, enc, clf, gen

def train_server_only_consistency(X_tr, y_tr, X_te, y_te, MMM, epochs=120, lr=0.01, lam_cons=0.1):
    enc = SimpleEncoder(X_tr.shape[1])
    clf = Classifier()
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train()
        h0 = enc(X_tr); z0 = clf(h0)
        loss_main = CE_LOSS(z0, y_tr)
        X_flip = X_tr.clone()
        for j in MMM:
            X_flip[:, j] = X_flip[torch.randperm(X_tr.shape[0]), j]
        h1 = enc(X_flip); z1 = clf(h1)
        loss_cons = torch.mean((z0 - z1) ** 2)
        loss = loss_main + lam_cons * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll, _ = eval_model(enc, clf, X_te, y_te)
    return acc, ll, enc, clf

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
        lr=lr_main,
        weight_decay=weight_decay,
    )
    opt_adv = optim.AdamW(adv.parameters(), lr=lr_adv, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt_main, T_max=max(1, epochs - warmup)
    ) if cosine else None

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
        loss_adv_confuse = BCE_LOGITS(
            adv_logits_main.squeeze(),
            0.5 * torch.ones_like(p_tr.float()),
        )

        loss = (
            loss_cls
            + lam_cons * loss_cons
            + lam_gen * loss_gen_reg
            + lam_adv * loss_adv_confuse
        )

        opt_main.zero_grad()
        loss.backward()
        opt_main.step()

        if epoch % 2 == 0:
            with torch.no_grad():
                h_det = enc(X_tr)
            adv_logits = adv(h_det.detach())
            loss_adv = BCE_LOGITS(adv_logits.squeeze(), p_tr.float())
            opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()

        if scheduler is not None and epoch >= warmup:
            scheduler.step()

        enc.eval(); clf.eval(); gen.eval()
        with torch.no_grad():
            acc_val, ll_val, _ = eval_model(enc, clf, X_val, y_val, temp_scaler=None)
            scg_val, fr_val = compute_scg_fr(enc, clf, gen, X_val, temp_scaler=None)
            score = ll_val + 0.1 * scg_val + 0.1 * (fr_val / 100.0)
        if stopper.step(score):
            best_state = {
                "enc": enc.state_dict(),
                "clf": clf.state_dict(),
                "gen": gen.state_dict(),
            }
        if stopper.stop():
            break

    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        clf.load_state_dict(best_state["clf"])
        gen.load_state_dict(best_state["gen"])

    # Temperature scaling on validation
    temp = TemperatureScaler()
    opt_t = optim.LBFGS(temp.parameters(), lr=0.1, max_iter=50)
    def _closure():
        opt_t.zero_grad()
        h = enc(X_val); logits = clf(h); logits = temp(logits)
        loss = nn.CrossEntropyLoss()(logits, y_val)
        loss.backward()
        return loss
    opt_t.step(_closure)

    acc, ll, _ = eval_model(enc, clf, X_te, y_te, temp_scaler=temp)
    scg, fr = compute_scg_fr(enc, clf, gen, X_te, temp_scaler=temp)
    return acc, ll, scg, fr, enc, clf, gen, temp

# ================================================================
# Discovery & Data Loading
# ================================================================
def graph_free_discovery(
    X_np,
    p_np,
    top_frac: float = 0.33,
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
# Main Execution Loop
# ================================================================
def run_single_seed_non_iid(seed: int, X_raw: pd.DataFrame, Y_raw: pd.DataFrame):
    print(f"\n============ German Credit NON-IID SEED {seed} ============\n")
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = X_raw.copy()
    y = (Y_raw.values.ravel() == 2).astype(int)  # 1 = bad, 0 = good

    # Encode categoricals
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    age_col = "Attribute13"
    protected = (X[age_col] >= 25).astype(int)     # 1 = older, 0 = younger
    X = X.drop(columns=[age_col])

    X_np = X.to_numpy(dtype=float)
    y_np = y
    p_np = protected.to_numpy(dtype=int)

    N = len(X_np)
    idx_all = np.arange(N)
    rng = np.random.RandomState(seed)

    old_idx = np.where(p_np == 1)[0]
    young_idx = np.where(p_np == 0)[0]

    # Train skewed to older group
    n_train_old = int(0.85 * len(old_idx))
    n_train_young = int(0.35 * len(young_idx))

    train_old = rng.choice(old_idx, size=n_train_old, replace=False)
    train_young = rng.choice(young_idx, size=n_train_young, replace=False)

    train_full_idx = np.concatenate([train_old, train_young])
    rng.shuffle(train_full_idx)
    test_idx = np.setdiff1d(idx_all, train_full_idx)

    X_train_full = X_np[train_full_idx]
    y_train_full = y_np[train_full_idx]
    p_train_full = p_np[train_full_idx]

    X_te_np = X_np[test_idx]
    y_te_np = y_np[test_idx]
    p_te_np = p_np[test_idx]

    print(f"NON-IID age proportion (train ≥25): {p_train_full.mean():.3f} | (test ≥25): {p_te_np.mean():.3f}")

    # Train / val split (still stratified by label)
    X_tr_np, X_val_np, y_tr_np, y_val_np, p_tr_np, p_val_np = train_test_split(
        X_train_full, y_train_full, p_train_full,
        test_size=0.20,
        random_state=seed,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr_np)
    X_val_np = scaler.transform(X_val_np)
    X_te_np = scaler.transform(X_te_np)

    X_tr = torch.from_numpy(X_tr_np).float()
    X_val = torch.from_numpy(X_val_np).float()
    X_te = torch.from_numpy(X_te_np).float()
    y_tr = torch.from_numpy(y_tr_np).long()
    y_val = torch.from_numpy(y_val_np).long()
    y_te = torch.from_numpy(y_te_np).long()
    p_tr = torch.from_numpy(p_tr_np).long()
    p_val = torch.from_numpy(p_val_np).long()
    p_te = torch.from_numpy(p_te_np).long()

    input_dim = X_tr.shape[1]
    print(f"Dataset (Non-IID): {X_tr.shape[0]} train, {X_val.shape[0]} val, {X_te.shape[0]} test | dim={input_dim}")

    # Graph-free mediator discovery on skewed train
    MMM, mi_all = graph_free_discovery(X_tr_np, p_tr.numpy(), top_frac=0.33, eps=2.0, seed=seed)
    NNN = set(range(input_dim)) - MMM
    mmm_sorted = sorted(list(MMM), key=lambda j: mi_all[j], reverse=True)
    PPP = set(mmm_sorted[:max(1, len(mmm_sorted) // 2)])
    print(f"NNN (non-desc): {sorted(NNN)}")
    print(f"MMM (mediators): {sorted(MMM)}")
    print(f"PPP (proxies): {sorted(PPP)}")

    results = {}

    # 1) Baseline
    print("Training: Baseline (no fairness)...")
    acc, ll, enc_b, clf_b = train_baseline(X_tr, y_tr, X_te, y_te, epochs=120, lr=0.01)
    eval_gen_base = MaskedGenerator(input_dim, MMM=MMM, cf_scale=0.20)
    scg, fr = compute_scg_fr(enc_b, clf_b, eval_gen_base, X_te)
    results["Baseline (no fairness)"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}
    print(f"  Acc: {acc*100:.1f}% | LL: {ll:.3f} | SCG: {scg:.3f} | FR: {fr:.1f}%")

    # 2) Uniform CF
    print("Training: Uniform CF (naïve)...")
    acc, ll, enc_u, clf_u, gen_u = train_uniform_cf(
        X_tr, y_tr, X_te, y_te, epochs=120, lr=0.01, lam_cons=0.2
    )
    scg, fr = compute_scg_fr(enc_u, clf_u, gen_u, X_te)
    results["Uniform CF (naïve)"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}
    print(f"  Acc: {acc*100:.1f}% | LL: {ll:.3f} | SCG: {scg:.3f} | FR: {fr:.1f}%")

    # 3) Policy-blind mask
    print("Training: Policy-blind mask...")
    acc, ll, enc_pb, clf_pb, gen_pb = train_policy_blind_mask(
        X_tr, y_tr, X_te, y_te, MMM,
        epochs=120, lr=0.01, lam_cons=0.3,
    )
    scg, fr = compute_scg_fr(enc_pb, clf_pb, gen_pb, X_te)
    results["Policy-blind mask"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}
    print(f"  Acc: {acc*100:.1f}% | LL: {ll:.3f} | SCG: {scg:.3f} | FR: {fr:.1f}%")

    # 4) Server-only consistency
    print("Training: Server-only consistency...")
    acc, ll, enc_s, clf_s = train_server_only_consistency(
        X_tr, y_tr, X_te, y_te, MMM,
        epochs=120, lr=0.01, lam_cons=0.1,
    )
    eval_gen_server = MaskedGenerator(input_dim, MMM=MMM, cf_scale=0.20)
    scg, fr = compute_scg_fr(enc_s, clf_s, eval_gen_server, X_te)
    results["Server-only consistency"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}
    print(f"  Acc: {acc*100:.1f}% | LL: {ll:.3f} | SCG: {scg:.3f} | FR: {fr:.1f}%")

    # 5) SCC-VFL (ours)
    print("Training: SCC-VFL (ours)...")
    acc, ll, scg, fr, enc_o, clf_o, gen_o, temp_o = train_scc_vfl_best(
        X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM, PPP,
        epochs=300, lr_main=0.015, lr_adv=0.005, weight_decay=5e-4,
        lam_cons_max=1.2, lam_adv=0.03, lam_gen=0.01, grad_rev_lambda=0.25,
        warmup=40, cosine=True,
    )
    results["SCC-VFL (ours)"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}
    print(f"  Acc: {acc*100:.1f}% | LL: {ll:.3f} | SCG: {scg:.3f} | FR: {fr:.1f}%")

    return results

def main():
    X_raw, Y_raw = load_german_credit_data()
    
    SEEDS = list(range(30))
    methods_order = [
        "Baseline (no fairness)",
        "Uniform CF (naïve)",
        "Policy-blind mask",
        "Server-only consistency",
        "SCC-VFL (ours)",
    ]

    metrics_store = {
        m: {"acc": [], "ll": [], "scg": [], "fr": []}
        for m in methods_order
    }

    for s in SEEDS:
        res_s = run_single_seed_non_iid(s, X_raw, Y_raw)
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
    print("MEAN±STD OVER 30 SEEDS (German Credit, Non-IID)")
    print("===================================================\n")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
