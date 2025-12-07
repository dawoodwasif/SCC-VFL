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
        # Fallback or exit if critical
        raise e

    # Annotate / Clean
    df = df.fillna("")
    df = df.replace(',', '[c]', regex=True)
    df = df.replace('\r', '', regex=True)
    df = df.replace('\n', ' ', regex=True)

    # Repeat rows to increase dataset size (as per original logic)
    while len(df) <= 15000:
        df = pd.concat([df, df], ignore_index=True)
    df = df.iloc[:15000]

    # Save locally (optional, but keeps original behavior)
    df.to_csv(annotated_file, index=False, header=False)
    
    # Reload with proper headers for processing
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
            nn.Linear(16, 1),
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

# --- VFL Specific Models ---
class LocalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class ConditionalMaskedVAE(nn.Module):
    def __init__(self, in_dim, mediator_idx_local, cond_dim=1,
                 hidden_dim=HIDDEN, latent_dim=8, scale=CF_SCALE_TRAIN):
        super().__init__()
        self.in_dim = in_dim
        self.m_idx = sorted(mediator_idx_local)
        self.cond_dim = cond_dim
        self.scale = scale
        m_dim = len(self.m_idx)

        # Encoder
        self.enc_fc1 = nn.Linear(m_dim + cond_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, m_dim)

    def encode(self, m, cond):
        h = torch.relu(self.enc_fc1(torch.cat([m, cond], dim=1)))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        h = torch.relu(self.dec_fc1(torch.cat([z, cond], dim=1)))
        m_out = self.dec_out(h)
        return m_out

    def forward(self, x_local, s, s_cf=None):
        m = x_local[:, self.m_idx]
        cond = s.unsqueeze(1).float()
        if s_cf is None:
            s_cf = 1 - s
        cond_cf = s_cf.unsqueeze(1).float()

        mu, logvar = self.encode(m, cond)
        z = self.reparameterize(mu, logvar)

        m_rec = self.decode(z, cond)
        m_cf = self.decode(z, cond_cf)

        x_cf = x_local.clone()
        x_cf[:, self.m_idx] = m_cf * self.scale

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return x_cf, m_rec, m, kld

class ProxyAdversary(nn.Module):
    def __init__(self, hidden_dim=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, h):
        return self.net(h)

class TopModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN, num_parties=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_parties, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
    def forward(self, h_list):
        h_cat = torch.cat(h_list, dim=1)
        return self.fc(h_cat)

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
    return accuracy_score(y.cpu().numpy(), pred), log_loss(y.cpu().numpy(), prob, labels=[0, 1])

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

@torch.no_grad()
def eval_scc(p1_enc, p2_enc, top_model, X, y, party1_idx, party2_idx):
    p1_enc.eval(); p2_enc.eval(); top_model.eval()
    x1 = X[:, party1_idx]
    x2 = X[:, party2_idx]
    h1 = p1_enc(x1)
    h2 = p2_enc(x2)
    logits = top_model([h1, h2])
    prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    return accuracy_score(y.cpu().numpy(), pred), log_loss(y.cpu().numpy(), prob, labels=[0, 1])

@torch.no_grad()
def compute_scg_fr_scc(p1_enc, p2_enc, cvae1, cvae2, top_model, X, s_vec, party1_idx, party2_idx, noise_std=EXTRA_NOISE_SCC):
    p1_enc.eval(); p2_enc.eval(); cvae1.eval(); cvae2.eval(); top_model.eval()
    x1 = X[:, party1_idx]
    x2 = X[:, party2_idx]
    s = s_vec

    h1 = p1_enc(x1)
    h2 = p2_enc(x2)
    logits0 = top_model([h1, h2])

    x1_cf, _, _, _ = cvae1(x1, s, s_cf=1 - s)
    x2_cf, _, _, _ = cvae2(x2, s, s_cf=1 - s)
    if noise_std > 0.0:
        x1_cf = x1_cf + noise_std * torch.randn_like(x1_cf)
        x2_cf = x2_cf + noise_std * torch.randn_like(x2_cf)
    h1_cf = p1_enc(x1_cf)
    h2_cf = p2_enc(x2_cf)
    logits1 = top_model([h1_cf, h2_cf])

    y0 = logits0.argmax(1)
    y1 = logits1.argmax(1)
    scg = torch.norm(logits0 - logits1, dim=1).mean().item()
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

def train_scc_vfl(X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, p_te, MMM, party1_idx, party2_idx, epochs=200):
    # Determine local MMM indices
    party1_all = list(party1_idx)
    party2_all = list(party2_idx)
    
    MMM1_global = sorted(list(MMM.intersection(set(party1_all))))
    MMM2_global = sorted(list(MMM.intersection(set(party2_all))))
    if len(MMM1_global) == 0: MMM1_global = [party1_all[0]]
    if len(MMM2_global) == 0: MMM2_global = [party2_all[0]]

    MMM1_local = [party1_all.index(j) for j in MMM1_global]
    MMM2_local = [party2_all.index(j) for j in MMM2_global]

    p1_enc = LocalEncoder(len(party1_all))
    p2_enc = LocalEncoder(len(party2_all))
    cvae1 = ConditionalMaskedVAE(len(party1_all), MMM1_local)
    cvae2 = ConditionalMaskedVAE(len(party2_all), MMM2_local)
    adv1 = ProxyAdversary()
    adv2 = ProxyAdversary()
    top_model = TopModel()

    params_main = (
        list(p1_enc.parameters()) + list(p2_enc.parameters()) +
        list(cvae1.parameters()) + list(cvae2.parameters()) +
        list(top_model.parameters())
    )
    params_adv = list(adv1.parameters()) + list(adv2.parameters())

    opt_main = optim.Adam(params_main, lr=0.01, weight_decay=5e-4)
    opt_adv = optim.Adam(params_adv, lr=0.005, weight_decay=5e-4)
    stopper = EarlyStopper(patience=25, min_delta=1e-4)
    best_state = None
    WARMUP = 30

    x1_tr = X_tr[:, party1_all]; x2_tr = X_tr[:, party2_all]
    s_tr = p_tr

    for epoch in range(epochs):
        p1_enc.train(); p2_enc.train(); cvae1.train(); cvae2.train()
        adv1.train(); adv2.train(); top_model.train()

        # Original branch
        h1 = p1_enc(x1_tr)
        h2 = p2_enc(x2_tr)
        logits = top_model([h1, h2])
        loss_cls = CE_LOSS(logits, y_tr)

        # Mediator cVAEs
        x1_cf, m1_rec, m1_orig, kld1 = cvae1(x1_tr, s_tr, s_cf=1 - s_tr)
        x2_cf, m2_rec, m2_orig, kld2 = cvae2(x2_tr, s_tr, s_cf=1 - s_tr)
        h1_cf = p1_enc(x1_cf)
        h2_cf = p2_enc(x2_cf)
        logits_cf = top_model([h1_cf, h2_cf])

        loss_scc = ((logits - logits_cf) ** 2).mean()
        rec1 = F.mse_loss(m1_rec, m1_orig, reduction="mean")
        rec2 = F.mse_loss(m2_rec, m2_orig, reduction="mean")
        vae_loss = rec1 + rec2 + 0.1 * (kld1 + kld2)

        # Proxy adversaries with GRL
        s_float = s_tr.float()
        h1_grl = grad_reverse(h1, lambd=0.2)
        h2_grl = grad_reverse(h2, lambd=0.2)
        adv_logits1 = adv1(h1_grl).squeeze()
        adv_logits2 = adv2(h2_grl).squeeze()
        loss_adv_conf = (
            BCE_LOSS(adv_logits1, 0.5 * torch.ones_like(s_float)) +
            BCE_LOSS(adv_logits2, 0.5 * torch.ones_like(s_float))
        )

        lam_scc = 0.0 if epoch < WARMUP else 0.6
        loss = loss_cls + lam_scc * loss_scc + 0.5 * vae_loss + 0.05 * loss_adv_conf

        opt_main.zero_grad(); loss.backward(); opt_main.step()

        # Train adversaries
        if epoch % 2 == 0:
            with torch.no_grad():
                h1_det = p1_enc(x1_tr)
                h2_det = p2_enc(x2_tr)
            adv_logits1_true = adv1(h1_det).squeeze()
            adv_logits2_true = adv2(h2_det).squeeze()
            loss_adv = BCE_LOSS(adv_logits1_true, s_float) + BCE_LOSS(adv_logits2_true, s_float)
            opt_adv.zero_grad(); loss_adv.backward(); opt_adv.step()

        # Validation
        acc_v, ll_v = eval_scc(p1_enc, p2_enc, top_model, X_val, y_val, party1_all, party2_all)
        scg_v, fr_v = compute_scg_fr_scc(
            p1_enc, p2_enc, cvae1, cvae2, top_model,
            X_val, p_val, party1_all, party2_all, noise_std=EXTRA_NOISE_SCC
        )
        score = ll_v + 0.2 * scg_v + 0.1 * (fr_v / 100.0)

        if stopper.step(score):
            best_state = {
                "p1_enc": p1_enc.state_dict(), "p2_enc": p2_enc.state_dict(),
                "cvae1": cvae1.state_dict(), "cvae2": cvae2.state_dict(),
                "top": top_model.state_dict(),
            }
        if stopper.stop():
            break

    if best_state is not None:
        p1_enc.load_state_dict(best_state["p1_enc"])
        p2_enc.load_state_dict(best_state["p2_enc"])
        cvae1.load_state_dict(best_state["cvae1"])
        cvae2.load_state_dict(best_state["cvae2"])
        top_model.load_state_dict(best_state["top"])

    acc, ll = eval_scc(p1_enc, p2_enc, top_model, X_te, y_te, party1_all, party2_all)
    scg, fr = compute_scg_fr_scc(
        p1_enc, p2_enc, cvae1, cvae2, top_model,
        X_te, p_te, party1_all, party2_all, noise_std=EXTRA_NOISE_SCC
    )
    return p1_enc, p2_enc, cvae1, cvae2, top_model, acc, ll, scg, fr

# ================================================================
# Main Execution Loop
# ================================================================
def run_single_seed_iid(seed: int, df_raw: pd.DataFrame):
    print(f"\n================ IID SEED {seed} ================\n")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Features, labels, protected
    X_all = df_raw.drop(columns=["sex", "target"]).astype(float).values
    y_all = df_raw["target"].astype(int).values
    p_all = df_raw["sex"].astype(int).values

    # IID split
    X_train_full, X_test_np, y_train_full, y_test_np, p_train_full, p_test_np = train_test_split(
        X_all, y_all, p_all, test_size=0.30, random_state=seed, stratify=y_all
    )
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
    print(f"Heart (IID, seed={seed}): {len(X_tr)} train, {len(X_val)} val, {len(X_te)} test | dim={input_dim}")

    # Discovery
    MMM, mi_all = graph_free_discovery(X_tr_np, p_tr.numpy(), top_frac=0.60, eps=2.0, seed=seed)
    print("MMM (mediators):", sorted(MMM))

    # VFL Partition
    party1_idx = [0, 2, 3, 6, 7, 9]
    party2_idx = [1, 4, 5, 8, 10, 11]

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

    print("Training: SCC-VFL (ours)...")
    _, _, _, _, _, acc, ll, scg, fr = train_scc_vfl(
        X_tr, y_tr, p_tr, X_val, y_val, p_val, X_te, y_te, p_te,
        MMM, party1_idx, party2_idx
    )
    results["SCC-VFL (ours)"] = {"acc": acc, "ll": ll, "scg": scg, "fr": fr}

    return results

def main():
    df_raw = load_and_preprocess_data()
    
    SEEDS = list(range(30))
    methods_order = ["Baseline", "Uniform CF", "Policy-blind", "Server-only", "SCC-VFL (ours)"]
    metrics_store = {m: {"acc": [], "ll": [], "scg": [], "fr": []} for m in methods_order}

    for s in SEEDS:
        res_s = run_single_seed_iid(s, df_raw)
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
    print("MEAN±STD OVER 30 SEEDS (Heart, IID)")
    print("===================================================\n")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
