import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from ucimlrepo import fetch_ucirepo

# ================================================================
# Configuration & Globals
# ================================================================
warnings.filterwarnings("ignore")
SEEDS = list(range(30))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, input_dim, hidden_dim=64, dropout=0.10):
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

def predict_labels(enc, clf, X, temp_scaler=None):
    enc.eval(); clf.eval()
    with torch.no_grad():
        h = enc(X)
        z = clf(h)
        if temp_scaler is not None:
            z = temp_scaler(z)
        y = z.argmax(dim=1)
    return y

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
    return MMM, mii

# ================================================================
# Attack Modules
# ================================================================
class AIAttackerH(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)

def make_balanced_subset(X, p, max_samples=None):
    idx0 = torch.where(p == 0)[0]
    idx1 = torch.where(p == 1)[0]
    n = min(len(idx0), len(idx1))
    if max_samples is not None:
        n = min(n, max_samples // 2)
    perm0 = idx0[torch.randperm(len(idx0))]
    perm1 = idx1[torch.randperm(len(idx1))]
    idx = torch.cat([perm0[:n], perm1[:n]], dim=0)
    idx = idx[torch.randperm(len(idx))]
    return X[idx], p[idx]

def train_aia_for_method(enc, clf, X_tr, p_tr, X_te, p_te, total_epochs=80, lr=5e-4, eval_epochs=(10, 30, 80)):
    enc.eval(); clf.eval()
    X_tr_bal, p_tr_bal = make_balanced_subset(X_tr, p_tr, max_samples=800)
    X_te_bal, p_te_bal = make_balanced_subset(X_te, p_te, max_samples=800)

    with torch.no_grad():
        h_tr = enc(X_tr_bal)
        h_te = enc(X_te_bal)

    attacker = AIAttackerH(in_dim=h_tr.shape[1])
    opt = optim.Adam(attacker.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    checkpoints = []
    eval_epochs = sorted(list(set(eval_epochs)))

    for epoch in range(1, total_epochs + 1):
        attacker.train()
        logits = attacker(h_tr)
        loss = criterion(logits, p_tr_bal.float())
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch in eval_epochs:
            attacker.eval()
            with torch.no_grad():
                logits_te = attacker(h_te)
                probs_te = torch.sigmoid(logits_te).cpu().numpy()
                p_true = p_te_bal.cpu().numpy()
                preds = (probs_te >= 0.5).astype(int)
            acc = (preds == p_true).mean()
            auc = roc_auc_score(p_true, probs_te)
            checkpoints.append((epoch, acc, auc))
    return checkpoints

def pgd_subspace_attack(enc, clf, X, y, MMM, eps=0.1, alpha=0.02, steps=20, temp_scaler=None):
    enc.eval(); clf.eval()
    X_orig = X.clone().detach()
    X_adv = X.clone().detach().requires_grad_(True)
    MMM_list = list(MMM)
    non_mmm = list(set(range(X.shape[1])) - set(MMM_list))

    for _ in range(steps):
        h = enc(X_adv)
        z = clf(h)
        if temp_scaler is not None: z = temp_scaler(z)
        loss = CE_LOSS(z, y)

        if X_adv.grad is not None: X_adv.grad.zero_()
        loss.backward()

        with torch.no_grad():
            grad = X_adv.grad
            X_adv_new = X_adv.clone()
            for j in MMM_list:
                X_adv_new[:, j] = X_adv[:, j] + alpha * grad[:, j].sign()

            delta = X_adv_new - X_orig
            for j in non_mmm: delta[:, j] = 0.0
            delta = torch.clamp(delta, min=-eps, max=eps)
            X_adv = (X_orig + delta).detach()
            X_adv.requires_grad_(True)
    return X_adv

# ================================================================
# Training Functions
# ================================================================
def train_baseline(X_tr, y_tr, X_te, y_te, epochs=120, lr=0.01):
    enc = SimpleEncoder(X_tr.shape[1], hidden_dim=64, dropout=0.10)
    clf = Classifier(hidden_dim=64)
    opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=lr)
    for _ in range(epochs):
        enc.train(); clf.train()
        h = enc(X_tr); z = clf(h)
        loss = CE_LOSS(z, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    acc, ll, _ = eval_model(enc, clf, X_te, y_te)
    return acc, ll, enc, clf

def train_uniform_cf(X_tr, y_tr, X_te, y_te, epochs=120, lr=0.01, lam_cons=0.15):
    enc = SimpleEncoder(X_tr.shape[1], hidden_dim=64, dropout=0.10)
    clf = Classifier(hidden_dim=64)
    gen = MaskedGenerator(X_tr.shape[1], hidden_dim=64, MMM=set(range(X_tr.shape[1])), cf_scale=0.30)
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

def train_policy_blind_mask(X_tr, y_tr, X_te, y_te, MMM, epochs=120, lr=0.01, lam_cons=0.25):
    enc = SimpleEncoder(X_tr.shape[1], hidden_dim=64, dropout=0.10)
    clf = Classifier(hidden_dim=64)
    gen = MaskedGenerator(X_tr.shape[1], hidden_dim=64, MMM=MMM, cf_scale=0.30)
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

def train_server_only_consistency(X_tr, y_tr, X_te, y_te, MMM, epochs=120, lr=0.01, lam_cons=0.05):
    enc = SimpleEncoder(X_tr.shape[1], hidden_dim=64, dropout=0.05)
    clf = Classifier(hidden_dim=64)
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
    enc = SimpleEncoder(X_tr.shape[1], hidden_dim=64, dropout=0.10)
    clf = Classifier(hidden_dim=64)
    gen = MaskedGenerator(X_tr.shape[1], hidden_dim=64, MMM=MMM, cf_scale=0.18)
    adv = LightAdversary(hidden_dim=64)
    MMM_list = sorted(list(MMM))

    opt_main = optim.AdamW(
        list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()),
        lr=lr_main, weight_decay=weight_decay,
    )
    opt_adv = optim.AdamW(adv.parameters(), lr=lr_adv, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=max(1, epochs - warmup)) if cosine else None
    stopper = EarlyStopper(patience=35, min_delta=1e-4)
    best_state = None

    lam_group = 0.5; lam_pgd = 1.0; eps_train = 0.20; alpha_fgsm = eps_train

    for epoch in range(epochs):
        enc.train(); clf.train(); gen.train(); adv.train()

        if epoch < warmup: lam_cons = 0.0
        else:
            t = (epoch - warmup) / max(1, (epochs - warmup))
            lam_cons = lam_cons_max * (0.5 - 0.5 * np.cos(np.pi * t))

        # FGSM-style adversarial example in MMM subspace
        X_adv = X_tr.clone().detach().requires_grad_(True)
        h_adv_inner = enc(X_adv); z_adv_inner = clf(h_adv_inner)
        loss_inner = CE_LOSS(z_adv_inner, y_tr)
        opt_main.zero_grad(); loss_inner.backward()

        with torch.no_grad():
            grad = X_adv.grad.detach()
            X_attack = X_tr.clone()
            for j in MMM_list: X_attack[:, j] = X_attack[:, j] + alpha_fgsm * grad[:, j].sign()
            delta = X_attack - X_tr
            for j in range(X_tr.shape[1]):
                if j not in MMM_list: delta[:, j] = 0.0
            delta = torch.clamp(delta, min=-eps_train, max=eps_train)
            X_attack = (X_tr + delta).detach()

        for p in list(enc.parameters()) + list(clf.parameters()) + list(gen.parameters()) + list(adv.parameters()):
            if p.grad is not None: p.grad.zero_()

        h0 = enc(X_tr); z0 = clf(h0)
        loss_cls = CE_LOSS(z0, y_tr)

        X_cf, delta = gen(X_tr)
        h1 = enc(X_cf); z1 = clf(h1)
        y0 = z0.argmax(dim=1); y1 = z1.argmax(dim=1)
        flip_mask = (y0 != y1).float()
        loss_cons = (flip_mask * torch.sum((z0 - z1) ** 2, dim=1)).mean()
        loss_gen_reg = (delta ** 2).mean()

        h_adv = enc(X_attack); z_adv = clf(h_adv)
        loss_pgd_inv = torch.mean(torch.sum((z0 - z_adv) ** 2, dim=1))

        idx0 = (p_tr == 0); idx1 = (p_tr == 1)
        if idx0.any() and idx1.any():
            h0_mean = h0[idx0].mean(dim=0); h1_mean = h0[idx1].mean(dim=0)
            loss_group = torch.mean((h0_mean - h1_mean) ** 2)
        else:
            loss_group = torch.tensor(0.0, device=X_tr.device)

        h_grl = grad_reverse(h0, grad_rev_lambda)
        adv_logits_main = adv(h_grl)
        loss_adv_confuse = BCE_LOGITS(adv_logits_main.squeeze(), 0.5 * torch.ones_like(p_tr.float()))

        loss = (
            loss_cls + lam_cons * loss_cons + lam_gen * loss_gen_reg +
            lam_adv * loss_adv_confuse + lam_pgd * loss_pgd_inv + lam_group * loss_group
        )

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

# ================================================================
# Main Execution Loop
# ================================================================
def run_single_seed(seed: int, X_raw: pd.DataFrame, Y_raw: pd.DataFrame,
                    aia_eval_epochs=(10, 30, 80), pgd_eps_list=(0.02, 0.05, 0.10, 0.20)):
    print(f"\n================ SEED {seed} ================")
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
        X_train_full, y_train_full, p_train_full, test_size=0.20, random_state=seed + 1, stratify=y_train_full,
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
    MMM, mi_all = graph_free_discovery(X_tr.numpy(), p_tr.numpy(), top_frac=0.33, eps=2.0)
    mmm_sorted = sorted(list(MMM), key=lambda j: mi_all[j], reverse=True)
    PPP = set(mmm_sorted[:max(1, len(mmm_sorted) // 2)])

    methods = {}

    print("Training: Adv-NoMask...")
    acc_b, ll_b, enc_b, clf_b = train_baseline(X_tr, y_tr, X_te, y_te)
    methods["Adv-NoMask"] = {"enc": enc_b, "clf": clf_b, "temp": None, "acc": acc_b, "ll": ll_b}

    print("Training: Uniform-CF...")
    acc_u, ll_u, enc_u, clf_u, gen_u = train_uniform_cf(X_tr, y_tr, X_te, y_te)
    methods["Uniform-CF"] = {"enc": enc_u, "clf": clf_u, "temp": None, "acc": acc_u, "ll": ll_u}

    print("Training: Policy-blind Mask...")
    acc_pb, ll_pb, enc_pb, clf_pb, gen_pb = train_policy_blind_mask(X_tr, y_tr, X_te, y_te, MMM)
    methods["Policy-blind Mask"] = {"enc": enc_pb, "clf": clf_pb, "temp": None, "acc": acc_pb, "ll": ll_pb}

    print("Training: Server-Consistency...")
    acc_s, ll_s, enc_s, clf_s = train_server_only_consistency(X_tr, y_tr, X_te, y_te, MMM)
    methods["Server-Consistency"] = {"enc": enc_s, "clf": clf_s, "temp": None, "acc": acc_s, "ll": ll_s}

    print("Training: SCC-VFL (ours)...")
    acc_o, ll_o, scg_o, fr_o, enc_o, clf_o, gen_o, temp_o = train_scc_vfl_best(
        X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, MMM, PPP,
    )
    methods["SCC-VFL (ours)"] = {"enc": enc_o, "clf": clf_o, "temp": temp_o, "acc": acc_o, "ll": ll_o}

    methods_order = ["Adv-NoMask", "Uniform-CF", "Policy-blind Mask", "Server-Consistency", "SCC-VFL (ours)"]
    base_metrics = {}
    aia_metrics = {m: {} for m in methods_order}
    pgd_metrics = {m: {} for m in methods_order}

    for name in methods_order:
        enc = methods[name]["enc"]; clf = methods[name]["clf"]; temp = methods[name]["temp"]
        acc, ll, _ = eval_model(enc, clf, X_te, y_te, temp_scaler=temp)
        base_metrics[name] = {"acc": acc, "ll": ll}

        print(f"\nRunning AIA (balanced) for {name}...")
        checkpoints = train_aia_for_method(
            enc, clf, X_tr, p_tr, X_te, p_te,
            total_epochs=max(aia_eval_epochs), lr=5e-4, eval_epochs=aia_eval_epochs,
        )
        for (epoch, acc_aia, auc_aia) in checkpoints:
            aia_metrics[name][epoch] = {"acc": acc_aia, "auc": auc_aia}

        print(f"Running Subspace PGD for {name}...")
        y_orig = predict_labels(enc, clf, X_te, temp_scaler=temp)
        for eps in pgd_eps_list:
            X_adv = pgd_subspace_attack(
                enc, clf, X_te, y_orig, MMM, eps=eps, alpha=eps/5.0, steps=20, temp_scaler=temp,
            )
            y_adv = predict_labels(enc, clf, X_adv, temp_scaler=temp)
            asr = (y_adv != y_orig).float().mean().item()
            pgd_metrics[name][eps] = {"asr": asr}

    return {"base": base_metrics, "aia": aia_metrics, "pgd": pgd_metrics}

def main():
    X_raw, Y_raw = load_german_credit_data()
    
    methods_order = ["Adv-NoMask", "Uniform-CF", "Policy-blind Mask", "Server-Consistency", "SCC-VFL (ours)"]
    aia_strengths = [10, 20, 40, 80]
    pgd_eps_list = [0.02, 0.05, 0.10, 0.20]

    base_acc_store = {m: [] for m in methods_order}
    base_ll_store = {m: [] for m in methods_order}
    aia_store = {m: {s: [] for s in aia_strengths} for m in methods_order}
    pgd_store = {m: {e: [] for e in pgd_eps_list} for m in methods_order}

    for seed in SEEDS:
        res = run_single_seed(seed, X_raw, Y_raw, aia_eval_epochs=aia_strengths, pgd_eps_list=pgd_eps_list)
        for m in methods_order:
            base_acc_store[m].append(res["base"][m]["acc"])
            base_ll_store[m].append(res["base"][m]["ll"])
            for s in aia_strengths: aia_store[m][s].append(res["aia"][m][s]["acc"])
            for e in pgd_eps_list: pgd_store[m][e].append(res["pgd"][m][e]["asr"])

    print("\n============ AVERAGED BASE METRICS OVER 30 SEEDS ============")
    for m in methods_order:
        acc_mean = np.mean(base_acc_store[m]); acc_std = np.std(base_acc_store[m])
        ll_mean = np.mean(base_ll_store[m]); ll_std = np.std(base_ll_store[m])
        print(f"{m:25s} | Acc: {acc_mean*100:.2f}±{acc_std*100:.2f}% | LL: {ll_mean:.3f}±{ll_std:.3f}")

    print("\n============ AIA ATTACK (BALANCED) OVER 30 SEEDS ============")
    for m in methods_order:
        for s in aia_strengths:
            vals = np.array(aia_store[m][s])
            print(f"{m:25s} | epochs={s:3d} | ASR: {vals.mean()*100:.2f}±{vals.std()*100:.2f}%")

    print("\n============ SUBSPACE PGD ATTACK (ASR) OVER 30 SEEDS =========")
    for m in methods_order:
        for e in pgd_eps_list:
            vals = np.array(pgd_store[m][e])
            print(f"{m:25s} | eps={e:.2f} | ASR: {vals.mean()*100:.2f}±{vals.std()*100:.2f}%")

    # Plotting
    plt.figure(figsize=(8, 5))
    for m in methods_order:
        means = [np.mean(aia_store[m][s]) for s in aia_strengths]
        plt.plot(aia_strengths, means, marker="o", label=m)
    plt.xlabel("AIA attacker epochs"); plt.ylabel("Average Attack Success Rate")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("aia_attack_strength_balanced_avg.png", dpi=200)

    plt.figure(figsize=(8, 5))
    for m in methods_order:
        means = [np.mean(pgd_store[m][e]) for e in pgd_eps_list]
        plt.plot(pgd_eps_list, means, marker="o", label=m)
    plt.xlabel("PGD epsilon"); plt.ylabel("Average Attack Success Rate")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("pgd_attack_strength_avg.png", dpi=200)

if __name__ == "__main__":
    main()
