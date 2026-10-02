"""All quantum OOD detectors: distance, density, reconstruction, and GAN-based families."""
import argparse
import glob
import hashlib
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pennylane as qml
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as tv_datasets
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rq1_feature_extractor"))
from qcl_family_quanforge_ood import (QSVDDDetector, QuantumFeatureExtractor,  # noqa: E402
                                       knn_scores, mean_scores, medoid_scores)
from qcl_family_quanforge_km_tuning import select_k, select_m  # noqa: E402
from run_quanforge_pipeline_mnist import build_run_tag_base, find_resume_checkpoint, set_global_seed  # noqa: E402

from scoring_pipelines import (load_ood_split_scaled, load_ood_split_scaled_raw,  # noqa: E402
                                normalize_images, resize_and_flatten)
from build_adversarial_ood_split import ATTACK_ORDER, load_adversarial_test_set  # noqa: E402


# ============================================================================
# Shared building blocks for the reconstruction/GAN detector family (qae, qvae, gan)
# ============================================================================

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_vqc_params(n_qubits, n_layers, rot_gate, entangle_gate):
    rot_per_q = 2 if rot_gate == "RX+RZ" else 1
    n_rot = n_qubits * rot_per_q * n_layers
    n_ent = n_qubits * n_layers if entangle_gate == "CRY" else 0
    return n_rot + n_ent


ADV_RESULTS_COLUMNS = ["dataset", "norm_cls", "eval_setting", "seed", "detector", "hyperparam", "auc_roc", "auc_pr",
                        "n_id", "n_fgsm", "n_pgd", "n_spsa", "n_salt_pepper"]


class FastVQC(nn.Module):
    """Differentiable statevector VQC: angle encoding, RX/RY/RX+RZ variational,
    CNOT or CRY ring entanglement, PauliZ measurement. Supports CRY trainable angles."""
    def __init__(self, n_qubits, n_layers, enc_gate="ry", rot_gate="RX+RZ",
                 entangle_gate="CRY", return_all_z=True):
        super().__init__()
        self.nq = n_qubits
        self.nl = n_layers
        self.dim = 2 ** n_qubits
        self.enc_gate = enc_gate
        self.rot_gate = rot_gate
        self.entangle_gate = entangle_gate
        self.return_all_z = return_all_z
        self.n_weights = count_vqc_params(n_qubits, n_layers, rot_gate, entangle_gate)
        self.weights = nn.Parameter(0.1 * torch.randn(self.n_weights))
        if entangle_gate == "CNOT":
            for c in range(n_qubits):
                t = (c + 1) % n_qubits
                self.register_buffer(f"_cnot_{c}_{t}", self._make_cnot_perm(c, t))
        for q in range(n_qubits):
            idx = torch.arange(self.dim)
            bit = (idx >> (n_qubits - 1 - q)) & 1
            self.register_buffer(f"_z0_{q}", (bit == 0).nonzero(as_tuple=True)[0])
            self.register_buffer(f"_z1_{q}", (bit == 1).nonzero(as_tuple=True)[0])
    def _make_cnot_perm(self, control, target):
        idx = torch.arange(self.dim)
        c_bit = (idx >> (self.nq - 1 - control)) & 1
        t_bit = (idx >> (self.nq - 1 - target)) & 1
        new_t = t_bit ^ c_bit
        return idx + (new_t - t_bit) * (1 << (self.nq - 1 - target))
    @staticmethod
    def _rx(theta):
        c = torch.cos(theta * 0.5); s = torch.sin(theta * 0.5)
        return torch.stack([torch.stack([c, -1j*s], -1), torch.stack([-1j*s, c], -1)], -2).to(torch.complex64)
    @staticmethod
    def _ry(theta):
        c = torch.cos(theta * 0.5); s = torch.sin(theta * 0.5)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)
    @staticmethod
    def _rz(theta):
        en = torch.exp(-1j * theta * 0.5); ep = torch.exp(1j * theta * 0.5); z = torch.zeros_like(en)
        return torch.stack([torch.stack([en, z], -1), torch.stack([z, ep], -1)], -2).to(torch.complex64)
    def _apply_1q(self, state, gate, q):
        B = state.shape[0]
        if q > 0: state = state.transpose(1, q + 1)
        state = state.reshape(B, 2, -1)
        state = torch.bmm(gate, state) if gate.dim() == 3 else torch.matmul(gate, state)
        state = state.reshape(B, *([2] * self.nq))
        if q > 0: state = state.transpose(1, q + 1)
        return state
    def _apply_cnot(self, state, c, t):
        B = state.shape[0]
        flat = state.reshape(B, -1)
        perm = getattr(self, f"_cnot_{c}_{t}")
        return flat.index_select(1, perm).reshape(B, *([2] * self.nq))
    def _apply_cry(self, state, control, target, theta):
        B = state.shape[0]; nq = self.nq
        c_val = torch.cos(theta * 0.5); s_val = torch.sin(theta * 0.5)
        ry = torch.stack([torch.stack([c_val, -s_val], -1),
                          torch.stack([s_val, c_val], -1)], -2).to(torch.complex64)
        perm = [0, control + 1, target + 1] + [i for i in range(1, nq + 1) if i not in [control + 1, target + 1]]
        state = state.permute(*perm).reshape(B, 2, 2, -1)
        s1 = state[:, 1:2, :, :].reshape(B, 2, -1)
        s1 = torch.matmul(ry, s1).reshape(B, 1, 2, -1)
        state = torch.cat([state[:, 0:1, :, :], s1], dim=1).reshape(B, *([2] * nq))
        inv_perm = [0] * (nq + 1)
        for i, p in enumerate(perm): inv_perm[p] = i
        return state.permute(*inv_perm)
    def forward(self, x):
        B = x.shape[0]; device = x.device
        state = torch.zeros(B, self.dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0
        state = state.reshape(B, *([2] * self.nq))
        enc_fn = self._rx if self.enc_gate == "rx" else self._ry
        for i in range(self.nq):
            state = self._apply_1q(state, enc_fn(x[:, i]), i)
        idx = 0
        for _ in range(self.nl):
            if self.rot_gate == "RX":
                for i in range(self.nq):
                    state = self._apply_1q(state, self._rx(self.weights[idx]), i); idx += 1
            elif self.rot_gate == "RY":
                for i in range(self.nq):
                    state = self._apply_1q(state, self._ry(self.weights[idx]), i); idx += 1
            else:  # RX+RZ
                for i in range(self.nq):
                    state = self._apply_1q(state, self._rx(self.weights[idx]), i); idx += 1
                    state = self._apply_1q(state, self._rz(self.weights[idx]), i); idx += 1
            for i in range(self.nq):
                t = (i + 1) % self.nq
                if self.entangle_gate == "CNOT":
                    state = self._apply_cnot(state, i, t)
                else:
                    state = self._apply_cry(state, i, t, self.weights[idx]); idx += 1
        flat = state.reshape(B, -1)
        probs = flat.abs().pow(2)
        if self.return_all_z:
            zs = []
            for i in range(self.nq):
                zs.append(probs.index_select(1, getattr(self, f"_z0_{i}")).sum(1)
                          - probs.index_select(1, getattr(self, f"_z1_{i}")).sum(1))
            return torch.stack(zs, dim=1)
        else:
            return (probs.index_select(1, getattr(self, "_z0_0")).sum(1)
                    - probs.index_select(1, getattr(self, "_z1_0")).sum(1)).unsqueeze(1)
def load_real_dataset(name, data_dir, target_class, scale, test_scale, img_shape):
    if name == "mnist":
        train_set = tv_datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = tv_datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        train_set = tv_datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = tv_datasets.FashionMNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    tr_idx_all = torch.where(train_set.targets == target_class)[0]
    te_idx_all = torch.where(test_set.targets == target_class)[0]
    pool = tr_idx_all[:int(scale * len(tr_idx_all))]
    n_tr = int(0.8 * len(pool))
    train_idx, val_idx = pool[:n_tr], pool[n_tr:]
    test_idx = te_idx_all[:int(test_scale * len(te_idx_all))]
    print(f"  [{name}] class {target_class}: train={len(train_idx)}, val={len(val_idx)}, test_id={len(test_idx)}")
    train_np = normalize_images(resize_and_flatten(train_set, train_idx, img_shape))
    val_np = normalize_images(resize_and_flatten(train_set, val_idx, img_shape))
    test_id_np = normalize_images(resize_and_flatten(test_set, test_idx, img_shape))
    return train_np, val_np, test_id_np
def load_ood_test(name, data_dir, target_class, test_scale, n_id_test, setting,
                  ood_class, img_shape, seed=42):
    """OOD count exactly equals ID count (50/50), random & evenly across non-ID classes."""
    rng = np.random.RandomState(seed)
    if name == "mnist":
        test_set = tv_datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        test_set = tv_datasets.FashionMNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    ood_classes = [ood_class] if setting == 2 else [c for c in range(10) if c != target_class]
    k = len(ood_classes)
    base, rem = divmod(n_id_test, k)
    extra = set(rng.choice(ood_classes, size=rem, replace=False).tolist())
    parts = []
    for c in ood_classes:
        c_idx = torch.where(test_set.targets == c)[0]
        c_idx = c_idx[:int(test_scale * len(c_idx))]
        n_c = min(base + (1 if c in extra else 0), len(c_idx))
        chosen = rng.choice(len(c_idx), size=n_c, replace=False)
        if n_c > 0:
            parts.append(normalize_images(resize_and_flatten(test_set, c_idx[chosen], img_shape)))
    ood = np.concatenate(parts, axis=0)
    if len(ood) > n_id_test:
        ood = ood[:n_id_test]
    elif len(ood) < n_id_test:
        pad = rng.choice(len(ood), size=n_id_test - len(ood), replace=True)
        ood = np.concatenate([ood, ood[pad]], axis=0)
    print(f"  test_ood={len(ood)} (~{base}/class, ID/OOD=50/50)")
    return ood
def evaluate(scores, labels):
    res = {"auc_roc": roc_auc_score(labels, scores),
           "auc_pr": average_precision_score(labels, scores)}
    id_scores = scores[labels == 0]
    for q in [0.70, 0.80, 0.90, 0.95, 0.99]:
        res[f"f1_{int(q*100)}"] = f1_score(labels, (scores > np.quantile(id_scores, q)).astype(int))
    return res
def compute_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)

# ============================================================================
# QAE: quantum autoencoder reconstruction-based OOD detector
# ============================================================================

class QuantumAutoencoder(nn.Module):
    def __init__(self, n_qubits, n_layers, latent_dim, feature_dim,
                 rot_gate="RX+RZ", entangle_gate="CRY"):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_compress = nn.Sequential(nn.Linear(feature_dim, n_qubits), nn.Tanh())
        self.enc_vqc = FastVQC(n_qubits, n_layers, enc_gate="ry", rot_gate=rot_gate,
                                entangle_gate=entangle_gate, return_all_z=True)
        self.dec_expand = nn.Sequential(nn.Linear(latent_dim, n_qubits), nn.Tanh())
        self.dec_vqc = FastVQC(n_qubits, n_layers, enc_gate="ry", rot_gate=rot_gate,
                                entangle_gate=entangle_gate, return_all_z=True)
        self.dec_upscale = nn.Sequential(
            nn.Linear(n_qubits, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, feature_dim), nn.Sigmoid())
    def encode(self, x):
        compressed = self.enc_compress(x)
        angles = torch.arccos(torch.clamp(compressed, -1.0, 1.0))
        return self.enc_vqc(angles)[:, :self.latent_dim]
    def decode(self, latent):
        expanded = self.dec_expand(latent)
        angles = torch.arccos(torch.clamp(expanded, -1.0, 1.0))
        q_out = self.dec_vqc(angles)
        return F.normalize(self.dec_upscale(q_out), p=2, dim=1)
    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent), latent
class ClassicalAutoencoder(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(),
                                     nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(),
                                     nn.Linear(128, feature_dim), nn.Sigmoid())
    def encode(self, x):
        return self.encoder(x)
    def decode(self, latent):
        return F.normalize(self.decoder(latent), p=2, dim=1)
    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent), latent
def train_ae(model, train_loader, val_loader, args, device, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        ep_loss, n_b = 0.0, 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ep_loss += loss.item(); n_b += 1
        if n_b == 0:
            print(f"  Epoch {epoch+1}: all batches skipped, stopping."); break
        history["train_loss"].append(ep_loss / n_b)
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                vl += criterion(recon, x).item() * x.shape[0]; nv += x.shape[0]
        avg_val = vl / max(nv, 1)
        history["val_loss"].append(avg_val)
        if avg_val < best_val:
            best_val, best_epoch = avg_val, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{model_name}] Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train_MSE={ep_loss/n_b:.6f} val_MSE={avg_val:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{model_name}] Best epoch {best_epoch}, val_MSE={best_val:.6f}")
    return history, best_val, best_epoch
def reconstruction_scores(model, x_np, device, batch_size=128):
    """Per-sample reconstruction MSE = anomaly score (higher = more anomalous)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.tensor(x_np[i:i+batch_size], dtype=torch.float32).to(device)
            recon = model(xb)[0]
            out.append(((xb - recon) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)
def plot_reconstructions(originals, recons_q, recons_c, save_path, n_samples=8):
    rows = [("Original", originals)]
    if recons_q is not None: rows.append(("QAE", recons_q))
    if recons_c is not None: rows.append(("Classical", recons_c))
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(2*n_samples, 2*n_rows))
    if n_samples == 1: axes = axes.reshape(n_rows, 1)
    for r, (label, data) in enumerate(rows):
        for j in range(n_samples):
            axes[r, j].imshow(data[j].reshape(16, 16), cmap="gray"); axes[r, j].axis("off")
            if j == 0: axes[r, j].set_ylabel(label, fontsize=10, rotation=0, labelpad=40, va="center")
    plt.suptitle("Autoencoder Reconstruction", fontsize=12)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
def plot_score_dist(scores, labels, save_path, title):
    plt.figure(figsize=(8, 4))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, density=True, label="ID")
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, density=True, label="OOD")
    plt.xlabel("Reconstruction error (anomaly score)"); plt.ylabel("Density")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
def plot_loss_curves(h_q, h_c, save_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    if h_q is not None:
        e = range(1, len(h_q["train_loss"]) + 1)
        a1.plot(e, h_q["train_loss"], label="QAE train"); a2.plot(e, h_q["val_loss"], label="QAE val")
    if h_c is not None:
        e = range(1, len(h_c["train_loss"]) + 1)
        a1.plot(e, h_c["train_loss"], label="Classical train"); a2.plot(e, h_c["val_loss"], label="Classical val")
    for ax, t in [(a1, "Training Loss"), (a2, "Validation Loss")]:
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title(t); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
def extract_latent(model, x_np, device, batch_size=128):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.tensor(x_np[i:i+batch_size], dtype=torch.float32).to(device)
            out.append(model.encode(xb).cpu().numpy())
    return np.concatenate(out)
def find_qae_checkpoint(checkpoint_dir, norm_cls, seed):
    pattern = os.path.join(checkpoint_dir, f"*_id{norm_cls}_nq*_seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 run dir matching '{pattern}', found {len(candidates)}: {candidates}. "
            f"Train it first via the qae subcommand."
        )
    model_path = os.path.join(candidates[0], "qae_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Run dir '{candidates[0]}' found but '{model_path}' is missing.")
    return model_path



def _run_qae_natural(args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.fast_test:
            args.epochs = 5; args.seeds = [0]; args.scale = 0.05; args.test_scale = 0.1; args.batch_size = 32
        os.makedirs(args.save_dir, exist_ok=True)
        feature_dim = args.img_size ** 2
        print(f"PyTorch: {torch.__version__}, Device: {device}")
        print(f"QAE n_qubits={args.n_qubits}, latent={args.latent_dim}, rot={args.rot_gate}, ent={args.entangle_gate}; "
              f"Classical latent={args.classical_latent_dim}")
        print(f"VQC trainable params: {count_vqc_params(args.n_qubits, args.n_layers, args.rot_gate, args.entangle_gate)}")
        print(f"seeds={args.seeds}, epochs={args.epochs}, setting={args.setting}")
        csv_columns = [
            "dataset", "setting", "model", "seed", "id_class", "ood_class",
            "n_qubits", "n_layers", "rot_gate", "entangle_gate",
            "latent_dim", "feature_dim",
            "epochs", "batch_size", "lr", "grad_clip", "scale", "test_scale",
            "n_train", "n_val", "n_test_id", "n_test_ood",
            "best_epoch", "best_val_mse", "id_test_mse", "id_test_psnr",
            "auc_roc", "auc_pr", "f1_70", "f1_80", "f1_90", "f1_95", "f1_99", "train_time",
        ]
        csv_path = os.path.join(args.save_dir, "qae_results.csv")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)
        target_classes = list(range(10)) if args.run_all_id else [args.target_class]
        for cls in target_classes:
            print(f"\n{'#'*70}\n# ID class: {cls}\n{'#'*70}")
            img_shape = (args.img_size, args.img_size)
            train_np, val_np, test_id_np = load_real_dataset(
                args.dataset, args.data_dir, cls, args.scale, args.test_scale, img_shape)
            test_ood_np = load_ood_test(
                args.dataset, args.data_dir, cls, args.test_scale, len(test_id_np),
                args.setting, args.ood_class, img_shape)
            test_y = np.concatenate([np.zeros(len(test_id_np)), np.ones(len(test_ood_np))]).astype(int)
            train_loader = DataLoader(TensorDataset(torch.tensor(train_np, dtype=torch.float32)),
                                      batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.tensor(val_np, dtype=torch.float32)),
                                    batch_size=args.batch_size)
            ood_str = f"all_except_{cls}" if args.setting == 1 else str(args.ood_class)
            for seed in args.seeds:
                set_seed(seed)
                print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")
                run_tag = (f"{args.dataset}_id{cls}_nq{args.n_qubits}_nl{args.n_layers}"
                           f"_rot{args.rot_gate.replace('+', '')}_ent{args.entangle_gate}"
                           f"_qlat{args.latent_dim}_clat{args.classical_latent_dim}_seed{seed}")
                run_dir = os.path.join(args.save_dir, run_tag)
                os.makedirs(run_dir, exist_ok=True)
                h_q, h_c, recons_q, recons_c = None, None, None, None
                def run_one(model_kind, model, latent_dim):
                    nonlocal h_q, h_c, recons_q, recons_c
                    name = "QAE" if model_kind == "qae" else "Classical"
                    print(f"\n--- Training {name} ---")
                    print(f"  {name} parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
                    t0 = time.time()
                    hist, best_val, best_epoch = train_ae(model, train_loader, val_loader, args, device, name)
                    train_time = time.time() - t0
                    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                                "history": hist, "best_epoch": best_epoch},
                               os.path.join(run_dir, f"{model_kind}_model.pt"))
                    model.eval()
                    with torch.no_grad():
                        vb = torch.tensor(val_np[:8], dtype=torch.float32).to(device)
                        preview = model(vb)[0].cpu().numpy()
                    for split, data in [("train", train_np), ("val", val_np),
                                        ("test_id", test_id_np), ("test_ood", test_ood_np)]:
                        np.savez(os.path.join(run_dir, f"{model_kind}_latent_{split}.npz"),
                                 latent=extract_latent(model, data, device))
                    id_score = reconstruction_scores(model, test_id_np, device)
                    id_mse = float(id_score.mean())
                    with torch.no_grad():
                        full_recon = model(torch.tensor(test_id_np, dtype=torch.float32).to(device))[0].cpu().numpy()
                    id_psnr = compute_psnr(test_id_np, full_recon)
                    ood_score = reconstruction_scores(model, test_ood_np, device)
                    scores = np.concatenate([id_score, ood_score])
                    metrics = evaluate(scores, test_y)
                    np.savez(os.path.join(run_dir, f"{model_kind}_scores.npz"), scores=scores, labels=test_y)
                    plot_score_dist(scores, test_y, os.path.join(run_dir, f"{model_kind}_score_dist.png"), name)
                    row = {
                        "dataset": args.dataset, "setting": args.setting, "model": name, "seed": seed,
                        "id_class": cls, "ood_class": ood_str,
                        "n_qubits": args.n_qubits, "n_layers": args.n_layers,
                        "rot_gate": args.rot_gate, "entangle_gate": args.entangle_gate,
                        "latent_dim": latent_dim, "feature_dim": feature_dim,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "grad_clip": args.grad_clip,
                        "scale": args.scale, "test_scale": args.test_scale,
                        "n_train": len(train_np), "n_val": len(val_np),
                        "n_test_id": len(test_id_np), "n_test_ood": len(test_ood_np),
                        "best_epoch": best_epoch, "best_val_mse": round(best_val, 6),
                        "id_test_mse": round(id_mse, 6), "id_test_psnr": round(id_psnr, 4),
                        "auc_roc": round(metrics["auc_roc"], 4), "auc_pr": round(metrics["auc_pr"], 4),
                        "f1_70": round(metrics["f1_70"], 4), "f1_80": round(metrics["f1_80"], 4),
                        "f1_90": round(metrics["f1_90"], 4), "f1_95": round(metrics["f1_95"], 4),
                        "f1_99": round(metrics["f1_99"], 4), "train_time": round(train_time, 2),
                    }
                    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)
                    print(f"  {name}: AUROC={metrics['auc_roc']:.4f} AUPR={metrics['auc_pr']:.4f} "
                          f"| ID_MSE={id_mse:.6f} PSNR={id_psnr:.2f} | {train_time:.1f}s")
                    if model_kind == "qae": h_q, recons_q = hist, preview
                    else: h_c, recons_c = hist, preview
                if args.model in ["qae", "both"]:
                    run_one("qae", QuantumAutoencoder(args.n_qubits, args.n_layers,
                                                      args.latent_dim, feature_dim,
                                                      rot_gate=args.rot_gate, entangle_gate=args.entangle_gate).to(device),
                            args.latent_dim)
                if args.model in ["classical", "both"]:
                    run_one("classical", ClassicalAutoencoder(args.classical_latent_dim,
                                                              feature_dim).to(device),
                            args.classical_latent_dim)
                plot_reconstructions(val_np[:8], recons_q, recons_c,
                                     os.path.join(run_dir, "reconstructions.png"))
                plot_loss_curves(h_q, h_c, os.path.join(run_dir, "loss_curves.png"))
                print(f"  Saved to: {run_dir}/")
        print(f"\n{'='*60}\nDone. CSV: {csv_path}\n{'='*60}")
        df = pd.read_csv(csv_path)
        cur = df[(df["dataset"] == args.dataset) & (df["setting"] == args.setting)
                 & (df["scale"] == args.scale) & (df["test_scale"] == args.test_scale)
                 & (df["n_qubits"] == args.n_qubits) & (df["n_layers"] == args.n_layers)
                 & (df["rot_gate"] == args.rot_gate) & (df["entangle_gate"] == args.entangle_gate)]
        per_class = cur.groupby(["model", "id_class"]).agg(
            auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
            auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
            n_seeds=("seed", "nunique")).round(4).reset_index()
        ov = []
        for m, g in per_class.groupby("model"):
            ov.append({"model": m, "id_class": "Overall",
                       "auc_roc_mean": round(g["auc_roc_mean"].mean(), 4),
                       "auc_roc_std": round(g["auc_roc_mean"].std(), 4),
                       "auc_pr_mean": round(g["auc_pr_mean"].mean(), 4),
                       "auc_pr_std": round(g["auc_pr_mean"].std(), 4), "n_seeds": len(g)})
        summary = pd.concat([per_class, pd.DataFrame(ov)], ignore_index=True)
        summary_path = os.path.join(args.save_dir, "summary_per_class.csv")
        summary.to_csv(summary_path, index=False)
        print("\n=== AUC-ROC per-class mean/std (across seeds) + Overall, Scheme 2 ===")
        print(summary.to_string(index=False))
        print(f"Summary saved: {summary_path}")


def _run_qae_adversarial(args):

        assert args.attack is not None, "--attack is required when using --adversarial_dir"
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
            device = torch.device("cuda")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        os.makedirs(args.output_dir, exist_ok=True)
        setting_tag = args.attack
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"qae_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                           args.attack)
        print(f"test={len(test_imgs)} (ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        # ---- this family's own preprocessing convention: L2-normalize + flatten ----
        test_imgs_norm = normalize_images(test_imgs).reshape(len(test_imgs), -1)
        feature_dim = args.img_size * args.img_size

        all_rows = []
        for seed in args.seeds:
            checkpoint_path = find_qae_checkpoint(args.checkpoint_dir, args.norm_cls, seed)
            print(f"Loading FIXED QAE checkpoint (seed={seed}) from '{checkpoint_path}' -- never retrained")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_args = ckpt["args"]

            model = QuantumAutoencoder(ckpt_args["n_qubits"], ckpt_args["n_layers"], ckpt_args["latent_dim"],
                                        feature_dim, ckpt_args.get("rot_gate", "RX+RZ"),
                                        ckpt_args.get("entangle_gate", "CRY")).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            scores = reconstruction_scores(model, test_imgs_norm, device)
            auc_roc = roc_auc_score(test_labels, scores)
            auc_pr = average_precision_score(test_labels, scores)
            print(f"QAE-Recon (seed={seed})  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
            all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                             "seed": seed, "detector": "QAE-Recon", "hyperparam": "",
                             "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

        df = pd.DataFrame(all_rows, columns=ADV_RESULTS_COLUMNS)
        df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "eval_setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def run_qae(args):
    if getattr(args, "adversarial_dir", None):
        _run_qae_adversarial(args)
    else:
        _run_qae_natural(args)


# ============================================================================
# QVAE: quantum variational autoencoder reconstruction-based OOD detector
# ============================================================================

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
class QuantumVAE(nn.Module):
    def __init__(self, n_qubits, n_layers, latent_dim, feature_dim,
                 rot_gate="RX+RZ", entangle_gate="CRY"):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_compress = nn.Sequential(nn.Linear(feature_dim, n_qubits), nn.Tanh())
        self.enc_vqc_mu = FastVQC(n_qubits, n_layers, enc_gate="ry", rot_gate=rot_gate,
                                  entangle_gate=entangle_gate, return_all_z=True)
        self.enc_logvar = nn.Linear(feature_dim, latent_dim)
        self.dec_expand = nn.Sequential(nn.Linear(latent_dim, n_qubits), nn.Tanh())
        self.dec_vqc = FastVQC(n_qubits, n_layers, enc_gate="ry", rot_gate=rot_gate,
                               entangle_gate=entangle_gate, return_all_z=True)
        self.dec_upscale = nn.Sequential(
            nn.Linear(n_qubits, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, feature_dim), nn.Sigmoid())
    def encode(self, x):
        compressed = self.enc_compress(x)
        angles = torch.arccos(torch.clamp(compressed, -1.0, 1.0))
        mu = self.enc_vqc_mu(angles)[:, :self.latent_dim]
        logvar = torch.clamp(self.enc_logvar(x), -10.0, 10.0)
        return mu, logvar
    def decode(self, z):
        expanded = self.dec_expand(z)
        angles = torch.arccos(torch.clamp(expanded, -1.0, 1.0))
        q_out = self.dec_vqc(angles)
        return F.normalize(self.dec_upscale(q_out), p=2, dim=1)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar
class ClassicalVAE(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU())
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(),
                                     nn.Linear(128, feature_dim), nn.Sigmoid())
    def encode(self, x):
        h = self.backbone(x)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10.0, 10.0)
    def decode(self, z):
        return F.normalize(self.decoder(z), p=2, dim=1)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar) if self.training else mu
        return self.decode(z), mu, logvar
def train_vae(model, train_loader, val_loader, args, device, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    recon_crit = nn.MSELoss()
    history = {"train_loss": [], "recon_loss": [], "kl_loss": [], "val_loss": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        ep_t, ep_r, ep_k, n_b = 0.0, 0.0, 0.0, 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            rec = recon_crit(recon, x); kl = kl_divergence(mu, logvar)
            loss = rec + args.beta * kl
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ep_t += loss.item(); ep_r += rec.item(); ep_k += kl.item(); n_b += 1
        if n_b == 0:
            print(f"  Epoch {epoch+1}: all batches skipped, stopping."); break
        history["train_loss"].append(ep_t/n_b); history["recon_loss"].append(ep_r/n_b); history["kl_loss"].append(ep_k/n_b)
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                v = recon_crit(recon, x) + args.beta * kl_divergence(mu, logvar)
                vl += v.item() * x.shape[0]; nv += x.shape[0]
        avg_val = vl / max(nv, 1)
        history["val_loss"].append(avg_val)
        if avg_val < best_val:
            best_val, best_epoch = avg_val, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{model_name}] Epoch {epoch+1:3d}/{args.epochs} | total={ep_t/n_b:.6f} "
                  f"recon={ep_r/n_b:.6f} KL={ep_k/n_b:.6f} val={avg_val:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{model_name}] Best epoch {best_epoch}, val_loss={best_val:.6f}")
    return history, best_val, best_epoch
def qvae_plot_reconstructions(originals, recons_q, recons_c, save_path, n_samples=8):
    rows = [("Original", originals)]
    if recons_q is not None: rows.append(("QVAE", recons_q))
    if recons_c is not None: rows.append(("Classical", recons_c))
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(2*n_samples, 2*n_rows))
    if n_samples == 1: axes = axes.reshape(n_rows, 1)
    for r, (label, data) in enumerate(rows):
        for j in range(n_samples):
            axes[r, j].imshow(data[j].reshape(16, 16), cmap="gray"); axes[r, j].axis("off")
            if j == 0: axes[r, j].set_ylabel(label, fontsize=10, rotation=0, labelpad=45, va="center")
    plt.suptitle("VAE Reconstruction", fontsize=12)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
def qvae_plot_loss_curves(h_q, h_c, save_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    if h_q is not None:
        e = range(1, len(h_q["train_loss"]) + 1)
        a1.plot(e, h_q["train_loss"], label="QVAE train"); a2.plot(e, h_q["val_loss"], label="QVAE val")
    if h_c is not None:
        e = range(1, len(h_c["train_loss"]) + 1)
        a1.plot(e, h_c["train_loss"], label="Classical train"); a2.plot(e, h_c["val_loss"], label="Classical val")
    for ax, t in [(a1, "Training Loss"), (a2, "Validation Loss")]:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(t); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
def qvae_extract_latent(model, x_np, device, batch_size=128):
    """Extract deterministic latent mean mu."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.tensor(x_np[i:i+batch_size], dtype=torch.float32).to(device)
            out.append(model.encode(xb)[0].cpu().numpy())  # [0] = mu
    return np.concatenate(out)
def find_qvae_checkpoint(checkpoint_dir, norm_cls, seed):
    pattern = os.path.join(checkpoint_dir, f"*_id{norm_cls}_nq*_seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 run dir matching '{pattern}', found {len(candidates)}: {candidates}. "
            f"Train it first via the qvae subcommand."
        )
    model_path = os.path.join(candidates[0], "qvae_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Run dir '{candidates[0]}' found but '{model_path}' is missing.")
    return model_path



def _run_qvae_natural(args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.fast_test:
            args.epochs = 5; args.seeds = [0]; args.scale = 0.05; args.test_scale = 0.1; args.batch_size = 32
        os.makedirs(args.save_dir, exist_ok=True)
        feature_dim = args.img_size ** 2
        print(f"PyTorch: {torch.__version__}, Device: {device}")
        print(f"QVAE n_qubits={args.n_qubits}, latent={args.latent_dim}, rot={args.rot_gate}, "
              f"ent={args.entangle_gate}, beta={args.beta}; Classical latent={args.classical_latent_dim}")
        print(f"VQC trainable params: {count_vqc_params(args.n_qubits, args.n_layers, args.rot_gate, args.entangle_gate)}")
        print(f"seeds={args.seeds}, epochs={args.epochs}, setting={args.setting}")
        csv_columns = [
            "dataset", "setting", "model", "seed", "id_class", "ood_class",
            "n_qubits", "n_layers", "rot_gate", "entangle_gate",
            "latent_dim", "feature_dim", "beta",
            "epochs", "batch_size", "lr", "grad_clip", "scale", "test_scale",
            "n_train", "n_val", "n_test_id", "n_test_ood",
            "best_epoch", "best_val_loss", "id_test_mse", "id_test_psnr",
            "auc_roc", "auc_pr", "f1_70", "f1_80", "f1_90", "f1_95", "f1_99", "train_time",
        ]
        csv_path = os.path.join(args.save_dir, "qvae_results.csv")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)
        target_classes = list(range(10)) if args.run_all_id else [args.target_class]
        for cls in target_classes:
            print(f"\n{'#'*70}\n# ID class: {cls}\n{'#'*70}")
            img_shape = (args.img_size, args.img_size)
            train_np, val_np, test_id_np = load_real_dataset(
                args.dataset, args.data_dir, cls, args.scale, args.test_scale, img_shape)
            test_ood_np = load_ood_test(
                args.dataset, args.data_dir, cls, args.test_scale, len(test_id_np),
                args.setting, args.ood_class, img_shape)
            test_y = np.concatenate([np.zeros(len(test_id_np)), np.ones(len(test_ood_np))]).astype(int)
            train_loader = DataLoader(TensorDataset(torch.tensor(train_np, dtype=torch.float32)),
                                      batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.tensor(val_np, dtype=torch.float32)),
                                    batch_size=args.batch_size)
            ood_str = f"all_except_{cls}" if args.setting == 1 else str(args.ood_class)
            for seed in args.seeds:
                set_seed(seed)
                print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")
                run_tag = (f"{args.dataset}_id{cls}_nq{args.n_qubits}_nl{args.n_layers}"
                           f"_rot{args.rot_gate.replace('+', '')}_ent{args.entangle_gate}"
                           f"_qlat{args.latent_dim}_clat{args.classical_latent_dim}_seed{seed}")
                run_dir = os.path.join(args.save_dir, run_tag)
                os.makedirs(run_dir, exist_ok=True)
                h_q, h_c, recons_q, recons_c = None, None, None, None
                def run_one(model_kind, model, latent_dim):
                    nonlocal h_q, h_c, recons_q, recons_c
                    name = "QVAE" if model_kind == "qvae" else "ClassicalVAE"
                    print(f"\n--- Training {name} ---")
                    print(f"  {name} parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
                    t0 = time.time()
                    hist, best_val, best_epoch = train_vae(model, train_loader, val_loader, args, device, name)
                    train_time = time.time() - t0
                    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                                "history": hist, "best_epoch": best_epoch},
                               os.path.join(run_dir, f"{model_kind}_model.pt"))
                    model.eval()
                    with torch.no_grad():
                        vb = torch.tensor(val_np[:8], dtype=torch.float32).to(device)
                        preview = model(vb)[0].cpu().numpy()
                    for split, data in [("train", train_np), ("val", val_np),
                                        ("test_id", test_id_np), ("test_ood", test_ood_np)]:
                        np.savez(os.path.join(run_dir, f"{model_kind}_latent_{split}.npz"),
                                 latent=qvae_extract_latent(model, data, device))
                    id_score = reconstruction_scores(model, test_id_np, device)
                    id_mse = float(id_score.mean())
                    with torch.no_grad():
                        full_recon = model(torch.tensor(test_id_np, dtype=torch.float32).to(device))[0].cpu().numpy()
                    id_psnr = compute_psnr(test_id_np, full_recon)
                    ood_score = reconstruction_scores(model, test_ood_np, device)
                    scores = np.concatenate([id_score, ood_score])
                    metrics = evaluate(scores, test_y)
                    np.savez(os.path.join(run_dir, f"{model_kind}_scores.npz"), scores=scores, labels=test_y)
                    plot_score_dist(scores, test_y, os.path.join(run_dir, f"{model_kind}_score_dist.png"), name)
                    row = {
                        "dataset": args.dataset, "setting": args.setting, "model": name, "seed": seed,
                        "id_class": cls, "ood_class": ood_str,
                        "n_qubits": args.n_qubits, "n_layers": args.n_layers,
                        "rot_gate": args.rot_gate, "entangle_gate": args.entangle_gate,
                        "latent_dim": latent_dim, "feature_dim": feature_dim, "beta": args.beta,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "grad_clip": args.grad_clip,
                        "scale": args.scale, "test_scale": args.test_scale,
                        "n_train": len(train_np), "n_val": len(val_np),
                        "n_test_id": len(test_id_np), "n_test_ood": len(test_ood_np),
                        "best_epoch": best_epoch, "best_val_loss": round(best_val, 6),
                        "id_test_mse": round(id_mse, 6), "id_test_psnr": round(id_psnr, 4),
                        "auc_roc": round(metrics["auc_roc"], 4), "auc_pr": round(metrics["auc_pr"], 4),
                        "f1_70": round(metrics["f1_70"], 4), "f1_80": round(metrics["f1_80"], 4),
                        "f1_90": round(metrics["f1_90"], 4), "f1_95": round(metrics["f1_95"], 4),
                        "f1_99": round(metrics["f1_99"], 4), "train_time": round(train_time, 2),
                    }
                    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)
                    print(f"  {name}: AUROC={metrics['auc_roc']:.4f} AUPR={metrics['auc_pr']:.4f} "
                          f"| ID_MSE={id_mse:.6f} PSNR={id_psnr:.2f} | {train_time:.1f}s")
                    if model_kind == "qvae": h_q, recons_q = hist, preview
                    else: h_c, recons_c = hist, preview
                if args.model in ["qvae", "both"]:
                    run_one("qvae", QuantumVAE(args.n_qubits, args.n_layers,
                                               args.latent_dim, feature_dim,
                                               rot_gate=args.rot_gate, entangle_gate=args.entangle_gate).to(device),
                            args.latent_dim)
                if args.model in ["classical", "both"]:
                    run_one("classical", ClassicalVAE(args.classical_latent_dim, feature_dim).to(device),
                            args.classical_latent_dim)
                qvae_plot_reconstructions(val_np[:8], recons_q, recons_c,
                                     os.path.join(run_dir, "reconstructions.png"))
                qvae_plot_loss_curves(h_q, h_c, os.path.join(run_dir, "loss_curves.png"))
                print(f"  Saved to: {run_dir}/")
        print(f"\n{'='*60}\nDone. CSV: {csv_path}\n{'='*60}")
        df = pd.read_csv(csv_path)
        cur = df[(df["dataset"] == args.dataset) & (df["setting"] == args.setting)
                 & (df["scale"] == args.scale) & (df["test_scale"] == args.test_scale)
                 & (df["n_qubits"] == args.n_qubits) & (df["n_layers"] == args.n_layers)
                 & (df["rot_gate"] == args.rot_gate) & (df["entangle_gate"] == args.entangle_gate)
                 & (df["beta"] == args.beta)]
        per_class = cur.groupby(["model", "id_class"]).agg(
            auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
            auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
            n_seeds=("seed", "nunique")).round(4).reset_index()
        ov = []
        for m, g in per_class.groupby("model"):
            ov.append({"model": m, "id_class": "Overall",
                       "auc_roc_mean": round(g["auc_roc_mean"].mean(), 4),
                       "auc_roc_std": round(g["auc_roc_mean"].std(), 4),
                       "auc_pr_mean": round(g["auc_pr_mean"].mean(), 4),
                       "auc_pr_std": round(g["auc_pr_mean"].std(), 4), "n_seeds": len(g)})
        summary = pd.concat([per_class, pd.DataFrame(ov)], ignore_index=True)
        summary_path = os.path.join(args.save_dir, "summary_per_class.csv")
        summary.to_csv(summary_path, index=False)
        print("\n=== AUC-ROC per-class mean/std (across seeds) + Overall, Scheme 2 ===")
        print(summary.to_string(index=False))
        print(f"Summary saved: {summary_path}")


def _run_qvae_adversarial(args):

        assert args.attack is not None, "--attack is required when using --adversarial_dir"
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
            device = torch.device("cuda")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        os.makedirs(args.output_dir, exist_ok=True)
        setting_tag = args.attack
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"qvae_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                           args.attack)
        print(f"test={len(test_imgs)} (ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        # ---- this family's own preprocessing convention: L2-normalize + flatten ----
        test_imgs_norm = normalize_images(test_imgs).reshape(len(test_imgs), -1)
        feature_dim = args.img_size * args.img_size

        all_rows = []
        for seed in args.seeds:
            checkpoint_path = find_qvae_checkpoint(args.checkpoint_dir, args.norm_cls, seed)
            print(f"Loading FIXED QVAE checkpoint (seed={seed}) from '{checkpoint_path}' -- never retrained")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_args = ckpt["args"]

            model = QuantumVAE(ckpt_args["n_qubits"], ckpt_args["n_layers"], ckpt_args["latent_dim"],
                                feature_dim, ckpt_args.get("rot_gate", "RX+RZ"),
                                ckpt_args.get("entangle_gate", "CRY")).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            scores = reconstruction_scores(model, test_imgs_norm, device)
            auc_roc = roc_auc_score(test_labels, scores)
            auc_pr = average_precision_score(test_labels, scores)
            print(f"QVAE-Recon (seed={seed})  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
            all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                             "seed": seed, "detector": "QVAE-Recon", "hyperparam": "",
                             "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

        df = pd.DataFrame(all_rows, columns=ADV_RESULTS_COLUMNS)
        df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "eval_setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def run_qvae(args):
    if getattr(args, "adversarial_dir", None):
        _run_qvae_adversarial(args)
    else:
        _run_qvae_natural(args)


# ============================================================================
# GAN: Q-AnoGAN / QWGAN-GP quantum GAN-based OOD detectors
# ============================================================================

def build_qiskit_circuit(n_qubits, n_layers, rot_gate="RX+RZ", entangle_gate="CRY", input_mode="noise"):
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
    except ImportError:
        return None, None, None
    rot_per_q = 2 if rot_gate == "RX+RZ" else 1
    n_rot = n_qubits * rot_per_q * n_layers
    n_ent = n_qubits * n_layers if entangle_gate == "CRY" else 0
    n_weight = n_rot + n_ent
    weight_params = ParameterVector("w", n_weight)
    input_params = ParameterVector("z", n_qubits) if input_mode == "noise" else ParameterVector("x", 2 ** n_qubits)
    qc = QuantumCircuit(n_qubits)
    if input_mode == "noise":
        for i in range(n_qubits):
            qc.rx(input_params[i], i)
    else:
        qc.prepare_state(input_params, range(n_qubits))
    w_idx = 0
    for _ in range(n_layers):
        if rot_gate == "RX":
            for i in range(n_qubits):
                qc.rx(weight_params[w_idx], i); w_idx += 1
        elif rot_gate == "RY":
            for i in range(n_qubits):
                qc.ry(weight_params[w_idx], i); w_idx += 1
        else:
            for i in range(n_qubits):
                qc.rx(weight_params[w_idx], i); w_idx += 1
                qc.rz(weight_params[w_idx], i); w_idx += 1
        for i in range(n_qubits):
            t = (i + 1) % n_qubits
            if entangle_gate == "CNOT":
                qc.cx(i, t)
            else:
                qc.cry(weight_params[w_idx], i, t); w_idx += 1
    return qc, input_params, weight_params

class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits, n_layers, rot_gate, entangle_gate, feature_dim):
        super().__init__()
        self.n_qubits = n_qubits
        self.vqc = FastVQC(n_qubits, n_layers, enc_gate="rx", rot_gate=rot_gate,
                           entangle_gate=entangle_gate, return_all_z=True)
        self.upscale = nn.Sequential(
            nn.Linear(n_qubits, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, feature_dim), nn.Sigmoid())
    def forward(self, z):
        q_out = self.vqc(z)
        x = self.upscale(q_out)
        return F.normalize(x, p=2, dim=1)

class QuantumDiscriminator(nn.Module):
    def __init__(self, n_qubits, n_layers, rot_gate, entangle_gate, feature_dim, bce_eps=0.0):
        super().__init__()
        self.bce_eps = bce_eps
        self.compress = nn.Sequential(nn.Linear(feature_dim, n_qubits), nn.Tanh())
        self.vqc = FastVQC(n_qubits, n_layers, enc_gate="ry", rot_gate=rot_gate,
                           entangle_gate=entangle_gate, return_all_z=False)
    def forward(self, x):
        compressed = self.compress(x)
        angles = torch.arccos(torch.clamp(compressed, -1.0, 1.0))
        score = self.vqc(angles)
        prob = torch.sigmoid(score)
        if self.bce_eps > 0:
            prob = torch.clamp(prob, self.bce_eps, 1.0 - self.bce_eps)
        return prob

class ClassicalCritic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 16), nn.LeakyReLU(0.2),
            nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x)

def build_dataloaders(train_np, val_np, test_id_np, test_ood_np, batch_size):
    train_x = torch.tensor(train_np, dtype=torch.float32)
    val_x = torch.tensor(val_np, dtype=torch.float32)
    test_x = torch.tensor(np.concatenate([test_id_np, test_ood_np], axis=0), dtype=torch.float32)
    test_y = torch.cat([torch.zeros(len(test_id_np), dtype=torch.long),
                        torch.ones(len(test_ood_np), dtype=torch.long)])
    train_loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, test_y.numpy()

def compute_val_residual(G, val_loader, args, device):
    G.eval(); total_res, n = 0.0, 0
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)
            z = torch.zeros(x.shape[0], args.latent_dim, device=device)
            total_res += torch.mean(torch.abs(x - G(z))).item() * x.shape[0]
            n += x.shape[0]
    return total_res / max(n, 1)

def train_q_anogan(G, D, train_loader, val_loader, args, device):
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    history = {"g_loss": [], "d_loss": [], "val_residual": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    total_skipped = 0
    for epoch in range(args.epochs):
        G.train(); D.train()
        epoch_g, epoch_d, n_batches, skip_count = 0.0, 0.0, 0, 0
        for (real_data,) in train_loader:
            bs = real_data.shape[0]; real_data = real_data.to(device)
            valid = torch.ones(bs, 1, device=device); fake = torch.zeros(bs, 1, device=device)
            opt_d.zero_grad()
            z = torch.empty(bs, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
            fake_data = G(z).detach()
            d_loss = (criterion(D(real_data), valid) + criterion(D(fake_data), fake)) / 2.0
            if torch.isnan(d_loss) or torch.isinf(d_loss):
                skip_count += 1; continue
            d_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=args.grad_clip)
            opt_d.step()
            for p in D.parameters(): p.requires_grad = False
            opt_g.zero_grad()
            z = torch.empty(bs, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
            g_loss = criterion(D(G(z)), valid)
            if torch.isnan(g_loss) or torch.isinf(g_loss):
                for p in D.parameters(): p.requires_grad = True
                skip_count += 1; continue
            g_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=args.grad_clip)
            opt_g.step()
            for p in D.parameters(): p.requires_grad = True
            epoch_g += g_loss.item(); epoch_d += d_loss.item(); n_batches += 1
        if n_batches == 0:
            print(f"  Epoch {epoch+1}: all batches skipped, stopping."); break
        avg_g, avg_d = epoch_g / n_batches, epoch_d / n_batches
        history["g_loss"].append(avg_g); history["d_loss"].append(avg_d)
        val_res = compute_val_residual(G, val_loader, args, device)
        history["val_residual"].append(val_res)
        if val_res < best_val:
            best_val, best_epoch = val_res, epoch + 1
            best_state = {"G": {k: v.clone() for k, v in G.state_dict().items()},
                          "D": {k: v.clone() for k, v in D.state_dict().items()}}
        total_skipped += skip_count
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | D={avg_d:.4f} G={avg_g:.4f} "
                  f"Val_L1={val_res:.6f} skip={skip_count}")
    if best_state is not None:
        G.load_state_dict(best_state["G"]); D.load_state_dict(best_state["D"])
    print(f"  Best epoch {best_epoch}, val_residual={best_val:.6f}, skipped={total_skipped}")
    return history, best_val, best_epoch

def compute_gradient_penalty(D, real_data, fake_data, device):
    bs = real_data.shape[0]
    alpha = torch.rand(bs, 1, device=device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interp = D(interpolates)
    gradients = torch.autograd.grad(outputs=d_interp, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interp),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(bs, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def train_qwgan_gp(G, D, train_loader, val_loader, args, device):
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    history = {"g_loss": [], "d_loss": [], "val_residual": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs):
        G.train(); D.train()
        epoch_g, epoch_d, n_g_steps = 0.0, 0.0, 0
        data_iter = iter(train_loader); n_batches = len(train_loader)
        for _ in range(n_batches):
            for _ in range(args.n_critic):
                try:
                    (real_data,) = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader); (real_data,) = next(data_iter)
                bs = real_data.shape[0]; real_data = real_data.to(device)
                opt_d.zero_grad()
                z = torch.empty(bs, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
                fake_data = G(z).detach()
                gp = compute_gradient_penalty(D, real_data, fake_data, device)
                d_loss = D(fake_data).mean() - D(real_data).mean() + args.lambda_gp * gp
                if not (torch.isnan(d_loss) or torch.isinf(d_loss)):
                    d_loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=args.grad_clip)
                    opt_d.step()
                epoch_d += d_loss.item()
            for p in D.parameters(): p.requires_grad = False
            opt_g.zero_grad()
            z = torch.empty(bs, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
            g_loss = -D(G(z)).mean()
            if not (torch.isnan(g_loss) or torch.isinf(g_loss)):
                g_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=args.grad_clip)
                opt_g.step()
            for p in D.parameters(): p.requires_grad = True
            epoch_g += g_loss.item(); n_g_steps += 1
        avg_g = epoch_g / max(n_g_steps, 1)
        avg_d = epoch_d / max(n_batches * args.n_critic, 1)
        history["g_loss"].append(avg_g); history["d_loss"].append(avg_d)
        val_res = compute_val_residual(G, val_loader, args, device)
        history["val_residual"].append(val_res)
        if val_res < best_val:
            best_val, best_epoch = val_res, epoch + 1
            best_state = {"G": {k: v.clone() for k, v in G.state_dict().items()},
                          "D": {k: v.clone() for k, v in D.state_dict().items()}}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | C={avg_d:.4f} G={avg_g:.4f} Val_L1={val_res:.6f}")
    if best_state is not None:
        G.load_state_dict(best_state["G"]); D.load_state_dict(best_state["D"])
    print(f"  Best epoch {best_epoch}, val_residual={best_val:.6f}")
    return history, best_val, best_epoch

def find_optimal_z(G, D, x, args, device):
    G.eval(); D.eval(); bs = x.shape[0]
    for p in G.parameters(): p.requires_grad = False
    for p in D.parameters(): p.requires_grad = False
    z = torch.empty(bs, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
    z.requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=args.z_lr)
    with torch.no_grad():
        d_real = D(x)
    for _ in range(args.z_iter):
        optimizer.zero_grad()
        fake = G(z); d_fake = D(fake)
        residual = torch.sum(torch.abs(x - fake), dim=1, keepdim=True)
        discrimination = torch.abs(d_real - d_fake)
        score = (1.0 / args.alpha) * residual + args.alpha * discrimination
        score.mean().backward(); optimizer.step()
    with torch.no_grad():
        fake = G(z); d_fake = D(fake)
        residual = torch.sum(torch.abs(x - fake), dim=1)
        discrimination = torch.abs(d_real.squeeze() - d_fake.squeeze())
        scores = (1.0 / args.alpha) * residual + args.alpha * discrimination
    for p in G.parameters(): p.requires_grad = True
    for p in D.parameters(): p.requires_grad = True
    return scores.detach().cpu().numpy()

def compute_anomaly_scores(G, D, test_loader, args, device):
    all_scores, all_times = [], []
    for x_batch, _ in tqdm(test_loader, desc="  Anomaly detection", leave=False):
        x_batch = x_batch.to(device)
        t0 = time.time()
        scores = find_optimal_z(G, D, x_batch, args, device)
        all_times.append(time.time() - t0)
        all_scores.append(scores)
    return np.concatenate(all_scores), sum(all_times)

def plot_losses(history, save_path, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["g_loss"]) + 1)
    ax1.plot(epochs, history["g_loss"], label="Generator"); ax1.plot(epochs, history["d_loss"], label="Discriminator/Critic")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title(f"{title} - Training Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_residual"], color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean L1 Residual"); ax2.set_title(f"{title} - Validation Residual"); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_generated_samples(G, args, device, save_path, n_samples=16):
    G.eval()
    with torch.no_grad():
        z = torch.empty(n_samples, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
        fake = G(z).cpu().numpy()
    fig, axes = plt.subplots(1, n_samples, figsize=(2*n_samples, 2))
    if n_samples == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(fake[i].reshape(args.img_size, args.img_size), cmap="gray"); ax.axis("off")
    plt.suptitle("Generated samples"); plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()

def plot_score_distribution(scores, labels, save_path, title):
    plt.figure(figsize=(8, 4))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="ID", density=True)
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="OOD", density=True)
    plt.xlabel("Anomaly score"); plt.ylabel("Density"); plt.title(f"{title} - Score Distribution")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_gan_checkpoint(checkpoint_dir, norm_cls, gan_name, seed):
    pattern = os.path.join(checkpoint_dir, f"*_id{norm_cls}_ood*_{gan_name}_seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 run dir matching '{pattern}', found {len(candidates)}: {candidates}. "
            f"Train it first via the gan subcommand."
        )
    model_path = os.path.join(candidates[0], f"{gan_name}_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Run dir '{candidates[0]}' found but '{model_path}' is missing.")
    return model_path



def _run_gan_natural(args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.fast_test:
            args.epochs = 3; args.z_iter = 10; args.seeds = [0]; args.scale = 0.05; args.batch_size = 32
        os.makedirs(args.save_dir, exist_ok=True)
        feature_dim = args.img_size * args.img_size
        print(f"PyTorch {torch.__version__}, Device: {device}")
        print(f"Feature dim: {feature_dim} ({args.img_size}x{args.img_size})")
        print(f"VQC: n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
              f"rot_gate={args.rot_gate}, entangle_gate={args.entangle_gate}")
        print(f"VQC params: {count_vqc_params(args.n_qubits, args.n_layers, args.rot_gate, args.entangle_gate)}")
        print(f"GAN: {args.gan_type}, epochs={args.epochs}, z_iter={args.z_iter}, seeds={args.seeds}")
        qc, _, _ = build_qiskit_circuit(args.n_qubits, args.n_layers, args.rot_gate, args.entangle_gate, "noise")
        if qc is not None:
            print(f"Qiskit circuit depth={qc.depth()}, gates={qc.size()}")

        csv_columns = [
            "dataset","setting","gan_type","circuit","detector","seed","id_class","ood_class",
            "scale","test_scale","n_qubits","n_layers","rot_gate","entangle_gate","latent_dim",
            "feature_dim","epochs","batch_size","lr_g","lr_d","n_critic","lambda_gp","z_iter",
            "z_lr","alpha","grad_clip","bce_eps","n_train","n_val","n_test_id","n_test_ood",
            "g_params","d_params","train_time","best_epoch","best_val_residual","final_g_loss",
            "final_d_loss","infer_time","auc_roc","auc_pr","f1_70","f1_80","f1_90","f1_95","f1_99",
        ]
        csv_path = os.path.join(args.save_dir, "gan_results.csv")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)
            print(f"Created new CSV: {csv_path}")
        else:
            print(f"Appending to existing CSV: {csv_path}")

        id_classes = list(range(10)) if args.run_all_id else [args.target_class]
        for id_cls in id_classes:
            print(f"\n{'#'*70}\n# ID class: {id_cls}\n{'#'*70}")
            img_shape = (args.img_size, args.img_size)
            train_np, val_np, test_id_np = load_real_dataset(
                args.dataset, args.data_dir, id_cls, args.scale, args.test_scale, img_shape)
            test_ood_np = load_ood_test(
                args.dataset, args.data_dir, id_cls, args.test_scale, len(test_id_np),
                args.setting, args.ood_class, img_shape, seed=args.seeds[0])
            train_loader, val_loader, test_loader, test_labels = build_dataloaders(
                train_np, val_np, test_id_np, test_ood_np, args.batch_size)
            ood_str = f"all_except_{id_cls}" if args.setting == 1 else str(args.ood_class)

            for seed in args.seeds:
                set_seed(seed)
                print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")
                gan_types = ["q_anogan","qwgan_gp"] if args.gan_type == "both" else [args.gan_type]
                for gan_name in gan_types:
                    print(f"\n--- {gan_name} ---"); set_seed(seed)
                    run_tag = (f"{args.dataset}_set{args.setting}_id{id_cls}_ood{ood_str}"
                               f"_nq{args.n_qubits}_nl{args.n_layers}"
                               f"_rot{args.rot_gate.replace('+','')}_ent{args.entangle_gate}"
                               f"_{gan_name}_seed{seed}")
                    run_dir = os.path.join(args.save_dir, run_tag)
                    os.makedirs(run_dir, exist_ok=True)
                    print(f"  Run dir: {run_dir}")

                    G = QuantumGenerator(args.n_qubits, args.n_layers, args.rot_gate,
                                         args.entangle_gate, feature_dim).to(device)
                    if gan_name == "q_anogan":
                        D = QuantumDiscriminator(args.n_qubits, args.n_layers, args.rot_gate,
                                                 args.entangle_gate, feature_dim, bce_eps=args.bce_eps).to(device)
                        detector_name = "Q-AnoGAN"
                    else:
                        D = ClassicalCritic(feature_dim).to(device)
                        detector_name = "QWGAN-GP"
                    g_params, d_params = count_parameters(G), count_parameters(D)
                    print(f"  G params={g_params}, D params={d_params}")

                    t0 = time.time()
                    if gan_name == "q_anogan":
                        history, best_val, best_epoch = train_q_anogan(G, D, train_loader, val_loader, args, device)
                    else:
                        history, best_val, best_epoch = train_qwgan_gp(G, D, train_loader, val_loader, args, device)
                    train_time = time.time() - t0

                    torch.save({"G_state_dict": G.state_dict(), "D_state_dict": D.state_dict(),
                                "args": vars(args), "history": history, "best_epoch": best_epoch,
                                "best_val": best_val}, os.path.join(run_dir, f"{gan_name}_model.pt"))
                    np.savez(os.path.join(run_dir, f"{gan_name}_history.npz"),
                             g_loss=history["g_loss"], d_loss=history["d_loss"], val_residual=history["val_residual"])
                    plot_losses(history, os.path.join(run_dir, f"{gan_name}_loss.png"), detector_name)
                    plot_generated_samples(G, args, device, os.path.join(run_dir, f"{gan_name}_samples.png"))

                    t1 = time.time()
                    scores, _ = compute_anomaly_scores(G, D, test_loader, args, device)
                    infer_time = time.time() - t1
                    np.savez(os.path.join(run_dir, f"{gan_name}_scores.npz"), scores=scores, labels=test_labels)
                    metrics = evaluate(scores, test_labels)
                    plot_score_distribution(scores, test_labels, os.path.join(run_dir, f"{gan_name}_scores.png"), detector_name)

                    row = {
                        "dataset": args.dataset, "setting": args.setting, "gan_type": gan_name,
                        "circuit": "VQC", "detector": detector_name, "seed": seed, "id_class": id_cls,
                        "ood_class": ood_str, "scale": args.scale, "test_scale": args.test_scale,
                        "n_qubits": args.n_qubits, "n_layers": args.n_layers, "rot_gate": args.rot_gate,
                        "entangle_gate": args.entangle_gate, "latent_dim": args.latent_dim,
                        "feature_dim": feature_dim, "epochs": args.epochs, "batch_size": args.batch_size,
                        "lr_g": args.lr_g, "lr_d": args.lr_d, "n_critic": args.n_critic,
                        "lambda_gp": args.lambda_gp, "z_iter": args.z_iter, "z_lr": args.z_lr,
                        "alpha": args.alpha, "grad_clip": args.grad_clip, "bce_eps": args.bce_eps,
                        "n_train": len(train_np), "n_val": len(val_np), "n_test_id": len(test_id_np),
                        "n_test_ood": len(test_ood_np), "g_params": g_params, "d_params": d_params,
                        "train_time": round(train_time, 2), "best_epoch": best_epoch,
                        "best_val_residual": round(best_val, 6), "final_g_loss": round(history["g_loss"][-1], 6),
                        "final_d_loss": round(history["d_loss"][-1], 6), "infer_time": round(infer_time, 2),
                        "auc_roc": round(metrics["auc_roc"], 4), "auc_pr": round(metrics["auc_pr"], 4),
                        "f1_70": round(metrics["f1_70"], 4), "f1_80": round(metrics["f1_80"], 4),
                        "f1_90": round(metrics["f1_90"], 4), "f1_95": round(metrics["f1_95"], 4),
                        "f1_99": round(metrics["f1_99"], 4),
                    }
                    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)
                    print(f"\n  {gan_name} (seed={seed}): AUROC={metrics['auc_roc']:.4f} "
                          f"AUPR={metrics['auc_pr']:.4f} | train={train_time:.1f}s infer={infer_time:.1f}s")
                    print(f"  Saved: {run_dir}/")

        df = pd.read_csv(csv_path)
        summary = df.groupby(["dataset","setting","gan_type","id_class","n_qubits","n_layers",
                              "rot_gate","entangle_gate"]).agg(
            auc_roc_mean=("auc_roc","mean"), auc_roc_std=("auc_roc","std"),
            auc_pr_mean=("auc_pr","mean"), f1_95_mean=("f1_95","mean"),
            f1_99_mean=("f1_99","mean"), train_time_mean=("train_time","mean"),
            infer_time_mean=("infer_time","mean")).round(4)
        summary_path = os.path.join(args.save_dir, "summary.csv")
        summary.to_csv(summary_path)
        print(f"\n{'='*70}\nDone. CSV: {csv_path}\nSummary: {summary_path}\n{'='*70}")
        print(summary.to_string())


def _run_gan_adversarial(args):

        assert args.attack is not None, "--attack is required when using --adversarial_dir"
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
            device = torch.device("cuda")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        os.makedirs(args.output_dir, exist_ok=True)
        setting_tag = args.attack
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"quantum_gan_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                           args.attack)
        print(f"test={len(test_imgs)} (ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        # ---- this family's own preprocessing convention: L2-normalize + flatten ----
        test_imgs_norm = normalize_images(test_imgs)
        test_x = torch.tensor(test_imgs_norm, dtype=torch.float32)
        test_y = torch.tensor(test_labels, dtype=torch.long)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

        all_rows = []
        for gan_name in args.gan_types:
            detector = GAN_NAME_TO_DETECTOR[gan_name]
            for seed in args.seeds:
                checkpoint_path = find_gan_checkpoint(args.checkpoint_dir, args.norm_cls, gan_name, seed)
                print(f"Loading FIXED {detector} checkpoint (seed={seed}) from '{checkpoint_path}' -- never retrained")
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                ckpt_args = ckpt["args"]

                feature_dim = args.img_size * args.img_size
                G = QuantumGenerator(ckpt_args["n_qubits"], ckpt_args["n_layers"], ckpt_args["rot_gate"],
                                      ckpt_args["entangle_gate"], feature_dim).to(device)
                G.load_state_dict(ckpt["G_state_dict"])
                G.eval()
                if gan_name == "q_anogan":
                    D = QuantumDiscriminator(ckpt_args["n_qubits"], ckpt_args["n_layers"], ckpt_args["rot_gate"],
                                              ckpt_args["entangle_gate"], feature_dim,
                                              ckpt_args.get("bce_eps", 0.0)).to(device)
                else:
                    D = ClassicalCritic(feature_dim).to(device)
                D.load_state_dict(ckpt["D_state_dict"])
                D.eval()

                score_args = argparse.Namespace(latent_dim=ckpt_args["latent_dim"], z_iter=ckpt_args["z_iter"],
                                                 z_lr=ckpt_args["z_lr"], alpha=ckpt_args["alpha"])
                scores, infer_time = compute_anomaly_scores(G, D, test_loader, score_args, device)
                auc_roc = roc_auc_score(test_labels, scores)
                auc_pr = average_precision_score(test_labels, scores)
                print(f"{detector} (seed={seed})  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f} (infer={infer_time:.1f}s)")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": detector, "hyperparam": "",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

        df = pd.DataFrame(all_rows, columns=ADV_RESULTS_COLUMNS)
        df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "eval_setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def run_gan(args):
    if getattr(args, "adversarial_dir", None):
        _run_gan_adversarial(args)
    else:
        _run_gan_natural(args)


# ============================================================================
# GANomaly: quantum/classical GANomaly OOD detector
# ============================================================================

class GanomalyFastVQC(nn.Module):
    def __init__(self, n_qubits, n_layers, enc_gate="ry", entangle_gate="CRY", return_all_z=True):
        super().__init__()
        self.nq = n_qubits
        self.nl = n_layers
        self.dim = 2 ** n_qubits
        self.enc_gate = enc_gate
        self.entangle_gate = entangle_gate
        self.return_all_z = return_all_z
        # weights: RX+RZ (2 per qubit per layer) + CRY angles (1 per edge per layer)
        n_rot = n_qubits * 2 * n_layers
        n_ent = n_qubits * n_layers if entangle_gate == "CRY" else 0
        self.n_weights = n_rot + n_ent
        self.weights = nn.Parameter(0.1 * torch.randn(self.n_weights))
        if entangle_gate == "CNOT":
            for c in range(n_qubits):
                t = (c + 1) % n_qubits
                self.register_buffer(f"_cnot_{c}_{t}", self._make_cnot_perm(c, t))
        for q in range(n_qubits):
            idx = torch.arange(self.dim)
            bit = (idx >> (n_qubits - 1 - q)) & 1
            self.register_buffer(f"_z0_{q}", (bit == 0).nonzero(as_tuple=True)[0])
            self.register_buffer(f"_z1_{q}", (bit == 1).nonzero(as_tuple=True)[0])

    def _make_cnot_perm(self, control, target):
        idx = torch.arange(self.dim)
        c_bit = (idx >> (self.nq - 1 - control)) & 1
        t_bit = (idx >> (self.nq - 1 - target)) & 1
        new_t = t_bit ^ c_bit
        return idx + (new_t - t_bit) * (1 << (self.nq - 1 - target))

    @staticmethod
    def _rx(theta):
        c = torch.cos(theta * 0.5); s = torch.sin(theta * 0.5)
        return torch.stack([torch.stack([c, -1j*s], -1), torch.stack([-1j*s, c], -1)], -2).to(torch.complex64)

    @staticmethod
    def _ry(theta):
        c = torch.cos(theta * 0.5); s = torch.sin(theta * 0.5)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)

    @staticmethod
    def _rz(theta):
        en = torch.exp(-1j * theta * 0.5); ep = torch.exp(1j * theta * 0.5); z = torch.zeros_like(en)
        return torch.stack([torch.stack([en, z], -1), torch.stack([z, ep], -1)], -2).to(torch.complex64)

    def _apply_1q(self, state, gate, q):
        B = state.shape[0]
        if q > 0: state = state.transpose(1, q + 1)
        state = state.reshape(B, 2, -1)
        state = torch.bmm(gate, state) if gate.dim() == 3 else torch.matmul(gate, state)
        state = state.reshape(B, *([2] * self.nq))
        if q > 0: state = state.transpose(1, q + 1)
        return state

    def _apply_cnot(self, state, c, t):
        B = state.shape[0]
        flat = state.reshape(B, -1)
        perm = getattr(self, f"_cnot_{c}_{t}")
        return flat.index_select(1, perm).reshape(B, *([2] * self.nq))

    def _apply_cry(self, state, control, target, theta):
        B = state.shape[0]; nq = self.nq
        c_val = torch.cos(theta * 0.5); s_val = torch.sin(theta * 0.5)
        ry = torch.stack([torch.stack([c_val, -s_val], -1), torch.stack([s_val, c_val], -1)], -2).to(torch.complex64)
        perm = [0, control + 1, target + 1] + [i for i in range(1, nq + 1) if i not in [control + 1, target + 1]]
        state = state.permute(*perm).reshape(B, 2, 2, -1)
        s1 = state[:, 1:2, :, :].reshape(B, 2, -1)
        s1 = torch.matmul(ry, s1).reshape(B, 1, 2, -1)
        state = torch.cat([state[:, 0:1, :, :], s1], dim=1).reshape(B, *([2] * nq))
        inv_perm = [0] * (nq + 1)
        for i, p in enumerate(perm): inv_perm[p] = i
        return state.permute(*inv_perm)

    def forward(self, x):
        B = x.shape[0]; device = x.device
        state = torch.zeros(B, self.dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0
        state = state.reshape(B, *([2] * self.nq))
        enc_fn = self._rx if self.enc_gate == "rx" else self._ry
        for i in range(self.nq):
            state = self._apply_1q(state, enc_fn(x[:, i]), i)
        idx = 0
        for _ in range(self.nl):
            for i in range(self.nq):
                state = self._apply_1q(state, self._rx(self.weights[idx]), i); idx += 1
                state = self._apply_1q(state, self._rz(self.weights[idx]), i); idx += 1
            for i in range(self.nq):
                t = (i + 1) % self.nq
                if self.entangle_gate == "CNOT":
                    state = self._apply_cnot(state, i, t)
                else:
                    state = self._apply_cry(state, i, t, self.weights[idx]); idx += 1
        flat = state.reshape(B, -1)
        probs = flat.abs().pow(2)
        if self.return_all_z:
            zs = []
            for i in range(self.nq):
                zs.append(probs.index_select(1, getattr(self, f"_z0_{i}")).sum(1)
                          - probs.index_select(1, getattr(self, f"_z1_{i}")).sum(1))
            return torch.stack(zs, dim=1)
        else:
            return (probs.index_select(1, getattr(self, "_z0_0")).sum(1)
                    - probs.index_select(1, getattr(self, "_z1_0")).sum(1)).unsqueeze(1)

# ===============================================================
# Models
# ===============================================================
class QuantumEncoder(nn.Module):
    def __init__(self, n_qubits, n_layers, feature_dim, entangle_gate="CRY"):
        super().__init__()
        self.compress = nn.Sequential(nn.Linear(feature_dim, n_qubits), nn.Tanh())
        self.vqc = GanomalyFastVQC(n_qubits, n_layers, enc_gate="ry", entangle_gate=entangle_gate, return_all_z=True)
    def forward(self, x):
        c = self.compress(x)
        angles = torch.arccos(torch.clamp(c, -1.0, 1.0))
        return self.vqc(angles)

class QuantumDecoder(nn.Module):
    def __init__(self, n_qubits, n_layers, feature_dim, entangle_gate="CRY"):
        super().__init__()
        self.vqc = GanomalyFastVQC(n_qubits, n_layers, enc_gate="rx", entangle_gate=entangle_gate, return_all_z=True)
        self.upscale = nn.Sequential(
            nn.Linear(n_qubits, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, feature_dim), nn.Sigmoid())
    def forward(self, z):
        q = self.vqc(z)
        return F.normalize(self.upscale(q), p=2, dim=1)

class QuantumGANomalyG(nn.Module):
    def __init__(self, n_qubits, n_layers, feature_dim, entangle_gate="CRY"):
        super().__init__()
        self.enc1 = QuantumEncoder(n_qubits, n_layers, feature_dim, entangle_gate)
        self.dec = QuantumDecoder(n_qubits, n_layers, feature_dim, entangle_gate)
        self.enc2 = QuantumEncoder(n_qubits, n_layers, feature_dim, entangle_gate)
    def forward(self, x):
        z1 = self.enc1(x)
        x_hat = self.dec(z1)
        z2 = self.enc2(x_hat)
        return x_hat, z1, z2

class ClassicalGANomalyG(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(),
                                 nn.Linear(128, feature_dim), nn.Sigmoid())
        self.enc2 = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
    def forward(self, x):
        z1 = self.enc1(x)
        x_hat = F.normalize(self.dec(z1), p=2, dim=1)
        z2 = self.enc2(x_hat)
        return x_hat, z1, z2

class ClassicalDiscriminator(nn.Module):
    def __init__(self, feature_dim, bce_eps=0.0):
        super().__init__()
        self.bce_eps = bce_eps
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        p = self.net(x)
        if self.bce_eps > 0:
            p = torch.clamp(p, self.bce_eps, 1.0 - self.bce_eps)
        return p

# ===============================================================
# Data
# ===============================================================
def train_ganomaly(G, D, train_loader, val_loader, args, device, name):
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    l1 = nn.L1Loss()
    history = {"g_loss": [], "d_loss": [], "l_rec": [], "l_lat": [], "val_gap": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs):
        G.train(); D.train()
        eg, ed, er, el, nb = 0.0, 0.0, 0.0, 0.0, 0
        for (x,) in train_loader:
            x = x.to(device); bs = x.shape[0]
            ones = torch.ones(bs, 1, device=device)
            zeros = torch.zeros(bs, 1, device=device)
            opt_d.zero_grad()
            with torch.no_grad():
                x_hat_det, _, _ = G(x)
            d_real = D(x); d_fake = D(x_hat_det)
            d_loss = (bce(d_real, ones) + bce(d_fake, zeros)) / 2.0
            if not (torch.isnan(d_loss) or torch.isinf(d_loss)):
                d_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                opt_d.step()
            for p in D.parameters(): p.requires_grad = False
            opt_g.zero_grad()
            x_hat, z1, z2 = G(x)
            loss_rec = l1(x_hat, x)
            loss_lat = F.l1_loss(z2, z1)
            loss_adv = bce(D(x_hat), ones)
            g_loss = args.w_rec * loss_rec + args.w_lat * loss_lat + args.w_adv * loss_adv
            if not (torch.isnan(g_loss) or torch.isinf(g_loss)):
                g_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
                opt_g.step()
            for p in D.parameters(): p.requires_grad = True
            eg += g_loss.item(); ed += d_loss.item()
            er += loss_rec.item(); el += loss_lat.item(); nb += 1
        if nb == 0:
            print(f"  Epoch {epoch+1}: all batches skipped, stopping."); break
        history["g_loss"].append(eg/nb); history["d_loss"].append(ed/nb)
        history["l_rec"].append(er/nb); history["l_lat"].append(el/nb)
        val_gap = val_latent_gap(G, val_loader, device)
        history["val_gap"].append(val_gap)
        if val_gap < best_val:
            best_val, best_epoch = val_gap, epoch + 1
            best_state = {"G": {k: v.clone() for k, v in G.state_dict().items()},
                          "D": {k: v.clone() for k, v in D.state_dict().items()}}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{name}] Epoch {epoch+1:3d}/{args.epochs} | G={eg/nb:.4f} D={ed/nb:.4f} "
                  f"rec={er/nb:.6f} lat={el/nb:.6f} val_gap={val_gap:.6f}")
    if best_state is not None:
        G.load_state_dict(best_state["G"]); D.load_state_dict(best_state["D"])
    print(f"  [{name}] Best epoch {best_epoch}, val_gap={best_val:.6f}")
    return history, best_val, best_epoch

def val_latent_gap(G, val_loader, device):
    G.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)
            _, z1, z2 = G(x)
            tot += torch.abs(z1 - z2).sum(dim=1).mean().item() * x.shape[0]
            n += x.shape[0]
    return tot / max(n, 1)

# ===============================================================
# Inference: one-pass latent-gap anomaly score
# ===============================================================
def ganomaly_scores(G, x_np, device, batch_size=128):
    G.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.tensor(x_np[i:i+batch_size], dtype=torch.float32).to(device)
            _, z1, z2 = G(xb)
            out.append(torch.abs(z1 - z2).sum(dim=1).cpu().numpy())
    return np.concatenate(out)

def ganomaly_plot_losses(history, save_path, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    e = range(1, len(history["g_loss"]) + 1)
    ax[0].plot(e, history["g_loss"], label="Generator(E-D-E)"); ax[0].plot(e, history["d_loss"], label="Discriminator")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].set_title(f"{title} - G/D Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(e, history["l_rec"], label="reconstruction"); ax[1].plot(e, history["l_lat"], label="latent gap")
    ax[1].plot(e, history["val_gap"], label="val latent gap")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss"); ax[1].set_title(f"{title} - Component Loss"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_recon(G, id_np, ood_np, device, save_path, n=6):
    G.eval()
    with torch.no_grad():
        xi = torch.tensor(id_np[:n], dtype=torch.float32).to(device)
        xo = torch.tensor(ood_np[:n], dtype=torch.float32).to(device)
        ri, _, _ = G(xi); ro, _, _ = G(xo)
        ri, ro = ri.cpu().numpy(), ro.cpu().numpy()
    fig, axes = plt.subplots(4, n, figsize=(2*n, 8))
    for j in range(n):
        for r, data in [(0, id_np), (1, ri), (2, ood_np[:n]), (3, ro)]:
            axes[r, j].imshow(data[j].reshape(16, 16), cmap="gray"); axes[r, j].axis("off")
        if j == 0:
            for r, lab in [(0, "ID orig"), (1, "ID recon"), (2, "OOD orig"), (3, "OOD recon")]:
                axes[r, j].set_ylabel(lab, fontsize=9, rotation=0, labelpad=48, va="center")
    plt.suptitle("GANomaly: ID vs OOD reconstruction"); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

def ganomaly_plot_score_dist(scores, labels, save_path, title):
    plt.figure(figsize=(8, 4))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, density=True, label="ID")
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, density=True, label="OOD")
    plt.xlabel("Latent-gap anomaly score"); plt.ylabel("Density")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

def ganomaly_count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ===============================================================
# Main
# ===============================================================
def find_ganomaly_checkpoint(checkpoint_dir, norm_cls, variant, seed):
    pattern = os.path.join(checkpoint_dir, f"*_id{norm_cls}_ood*_{variant}_*_seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 run dir matching '{pattern}', found {len(candidates)}: {candidates}. "
            f"Train it first via the ganomaly subcommand."
        )
    model_path = os.path.join(candidates[0], f"{variant}_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Run dir '{candidates[0]}' found but '{model_path}' is missing.")
    return model_path



def _run_ganomaly_natural(args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.fast_test:
            args.epochs = 3; args.seeds = [0]; args.scale = 0.05; args.test_scale = 0.1; args.batch_size = 32
        os.makedirs(args.save_dir, exist_ok=True)
        feature_dim = args.img_size ** 2
        print(f"PyTorch {torch.__version__}, Device: {device}")
        print(f"Dataset={args.dataset}, setting={args.setting}, seeds={args.seeds}, epochs={args.epochs}")
        print(f"Quantum: n_qubits={args.n_qubits}, n_layers={args.n_layers}, latent={args.latent_dim}, "
              f"entangle={args.entangle_gate}; Classical latent={args.classical_latent_dim}")
        print(f"Loss weights: w_rec={args.w_rec}, w_lat={args.w_lat}, w_adv={args.w_adv}")
        csv_columns = [
            "dataset", "setting", "model", "seed", "id_class", "ood_class",
            "n_qubits", "n_layers", "latent_dim", "entangle_gate", "feature_dim",
            "epochs", "batch_size", "lr_g", "lr_d", "w_rec", "w_lat", "w_adv",
            "grad_clip", "bce_eps", "scale", "test_scale",
            "n_train", "n_val", "n_test_id", "n_test_ood",
            "g_params", "d_params", "best_epoch", "best_val_gap",
            "final_g_loss", "final_d_loss",
            "auc_roc", "auc_pr", "f1_70", "f1_80", "f1_90", "f1_95", "f1_99", "train_time",
        ]
        csv_path = os.path.join(args.save_dir, "ganomaly_results.csv")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)
        target_classes = list(range(10)) if args.run_all_id else [args.target_class]
        for cls in target_classes:
            print(f"\n{'#'*70}\n# ID class: {cls}\n{'#'*70}")
            img_shape = (args.img_size, args.img_size)
            train_np, val_np, test_id_np = load_real_dataset(
                args.dataset, args.data_dir, cls, args.scale, args.test_scale, img_shape)
            test_ood_np = load_ood_test(
                args.dataset, args.data_dir, cls, args.test_scale, len(test_id_np),
                args.setting, args.ood_class, img_shape)
            test_y = np.concatenate([np.zeros(len(test_id_np)), np.ones(len(test_ood_np))]).astype(int)
            train_loader = DataLoader(TensorDataset(torch.tensor(train_np, dtype=torch.float32)),
                                      batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.tensor(val_np, dtype=torch.float32)),
                                    batch_size=args.batch_size)
            ood_str = f"all_except_{cls}" if args.setting == 1 else str(args.ood_class)
            for seed in args.seeds:
                set_seed(seed)
                print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")
                variants = (["qganomaly", "classical"] if args.model == "both" else [args.model])
                for variant in variants:
                    set_seed(seed)
                    model_name = "Q-GANomaly" if variant == "qganomaly" else "C-GANomaly"
                    latent_dim = args.latent_dim if variant == "qganomaly" else args.classical_latent_dim
                    print(f"\n--- {model_name} ---")
                    run_tag = (f"{args.dataset}_set{args.setting}_id{cls}_ood{ood_str}"
                               f"_{variant}_nq{args.n_qubits}_nl{args.n_layers}"
                               f"_ent{args.entangle_gate}_lat{latent_dim}_seed{seed}")
                    run_dir = os.path.join(args.save_dir, run_tag)
                    os.makedirs(run_dir, exist_ok=True)
                    if variant == "qganomaly":
                        G = QuantumGANomalyG(args.n_qubits, args.n_layers, feature_dim,
                                             entangle_gate=args.entangle_gate).to(device)
                    else:
                        G = ClassicalGANomalyG(latent_dim, feature_dim).to(device)
                    D = ClassicalDiscriminator(feature_dim, bce_eps=args.bce_eps).to(device)
                    g_params, d_params = ganomaly_count_parameters(G), ganomaly_count_parameters(D)
                    print(f"  G params: {g_params}, D params: {d_params}")
                    t0 = time.time()
                    history, best_val, best_epoch = train_ganomaly(
                        G, D, train_loader, val_loader, args, device, model_name)
                    train_time = time.time() - t0
                    torch.save({"G_state_dict": G.state_dict(), "D_state_dict": D.state_dict(),
                                "args": vars(args), "history": history, "best_epoch": best_epoch},
                               os.path.join(run_dir, f"{variant}_model.pt"))
                    ganomaly_plot_losses(history, os.path.join(run_dir, f"{variant}_loss.png"), model_name)
                    plot_recon(G, test_id_np, test_ood_np, device,
                               os.path.join(run_dir, f"{variant}_recon.png"))
                    id_scores = ganomaly_scores(G, test_id_np, device)
                    ood_scores = ganomaly_scores(G, test_ood_np, device)
                    scores = np.concatenate([id_scores, ood_scores])
                    metrics = evaluate(scores, test_y)
                    np.savez(os.path.join(run_dir, f"{variant}_scores.npz"), scores=scores, labels=test_y)
                    ganomaly_plot_score_dist(scores, test_y, os.path.join(run_dir, f"{variant}_scores.png"), model_name)
                    row = {
                        "dataset": args.dataset, "setting": args.setting, "model": model_name,
                        "seed": seed, "id_class": cls, "ood_class": ood_str,
                        "n_qubits": args.n_qubits, "n_layers": args.n_layers,
                        "latent_dim": latent_dim, "entangle_gate": args.entangle_gate,
                        "feature_dim": feature_dim,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "lr_g": args.lr_g, "lr_d": args.lr_d,
                        "w_rec": args.w_rec, "w_lat": args.w_lat, "w_adv": args.w_adv,
                        "grad_clip": args.grad_clip, "bce_eps": args.bce_eps,
                        "scale": args.scale, "test_scale": args.test_scale,
                        "n_train": len(train_np), "n_val": len(val_np),
                        "n_test_id": len(test_id_np), "n_test_ood": len(test_ood_np),
                        "g_params": g_params, "d_params": d_params,
                        "best_epoch": best_epoch, "best_val_gap": round(best_val, 6),
                        "final_g_loss": round(history["g_loss"][-1], 6),
                        "final_d_loss": round(history["d_loss"][-1], 6),
                        "auc_roc": round(metrics["auc_roc"], 4), "auc_pr": round(metrics["auc_pr"], 4),
                        "f1_70": round(metrics["f1_70"], 4), "f1_80": round(metrics["f1_80"], 4),
                        "f1_90": round(metrics["f1_90"], 4), "f1_95": round(metrics["f1_95"], 4),
                        "f1_99": round(metrics["f1_99"], 4), "train_time": round(train_time, 2),
                    }
                    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)
                    print(f"  {model_name} (seed={seed}): AUROC={metrics['auc_roc']:.4f} AUPR={metrics['auc_pr']:.4f} "
                          f"| train={train_time:.1f}s | saved {run_dir}/")
        # Scheme 2 summary
        df = pd.read_csv(csv_path)
        cur = df[(df["dataset"] == args.dataset) & (df["setting"] == args.setting)
                 & (df["scale"] == args.scale) & (df["test_scale"] == args.test_scale)
                 & (df["n_qubits"] == args.n_qubits) & (df["n_layers"] == args.n_layers)
                 & (df["entangle_gate"] == args.entangle_gate)
                 & (df["w_rec"] == args.w_rec) & (df["w_lat"] == args.w_lat) & (df["w_adv"] == args.w_adv)]
        per_class = cur.groupby(["model", "id_class"]).agg(
            auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
            auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
            n_seeds=("seed", "nunique")).round(4).reset_index()
        ov = []
        for m, g in per_class.groupby("model"):
            ov.append({"model": m, "id_class": "Overall",
                       "auc_roc_mean": round(g["auc_roc_mean"].mean(), 4),
                       "auc_roc_std": round(g["auc_roc_mean"].std(), 4),
                       "auc_pr_mean": round(g["auc_pr_mean"].mean(), 4),
                       "auc_pr_std": round(g["auc_pr_mean"].std(), 4), "n_seeds": len(g)})
        summary = pd.concat([per_class, pd.DataFrame(ov)], ignore_index=True)
        summary_path = os.path.join(args.save_dir, "summary_per_class.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\n{'='*70}\nDone. CSV: {csv_path}\nSummary: {summary_path}\n{'='*70}")
        print("\n=== AUC-ROC per-class mean/std (across seeds) + Overall, Scheme 2 ===")
        print(summary.to_string(index=False))


def _run_ganomaly_adversarial(args):

        assert args.attack is not None, "--attack is required when using --adversarial_dir"
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
            device = torch.device("cuda")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        os.makedirs(args.output_dir, exist_ok=True)
        setting_tag = args.attack
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"ganomaly_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                           args.attack)
        print(f"test={len(test_imgs)} (ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        # ---- this family's own preprocessing convention: L2-normalize + flatten ----
        test_imgs_norm = normalize_images(test_imgs).reshape(len(test_imgs), -1)
        feature_dim = args.img_size * args.img_size

        all_rows = []
        for variant in args.variants:
            detector = VARIANT_TO_DETECTOR[variant]
            for seed in args.seeds:
                checkpoint_path = find_ganomaly_checkpoint(args.checkpoint_dir, args.norm_cls, variant, seed)
                print(f"Loading FIXED {detector} checkpoint (seed={seed}) from '{checkpoint_path}' -- never retrained")
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                ckpt_args = ckpt["args"]

                if variant == "qganomaly":
                    G = QuantumGANomalyG(ckpt_args["n_qubits"], ckpt_args["n_layers"], feature_dim,
                                          entangle_gate=ckpt_args.get("entangle_gate", "CRY")).to(device)
                else:
                    G = ClassicalGANomalyG(ckpt_args["classical_latent_dim"], feature_dim).to(device)
                G.load_state_dict(ckpt["G_state_dict"])
                G.eval()

                scores = ganomaly_scores(G, test_imgs_norm, device)
                auc_roc = roc_auc_score(test_labels, scores)
                auc_pr = average_precision_score(test_labels, scores)
                print(f"{detector} (seed={seed})  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": detector, "hyperparam": "",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

        df = pd.DataFrame(all_rows, columns=ADV_RESULTS_COLUMNS)
        df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "eval_setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def run_ganomaly(args):
    if getattr(args, "adversarial_dir", None):
        _run_ganomaly_adversarial(args)
    else:
        _run_ganomaly_natural(args)


# ============================================================================
# Distance-based OOD detectors on a fixed quantum feature extractor (QKNN, QMean, QMedoids, QSVDD)
# ============================================================================

DISTANCE_RESULTS_COLUMNS = ["circuit", "input_resolution", "n_qubits", "feature_loss", "extractor_seed", "seed",
                            "detector", "hyperparam", "auc_roc", "auc_pr", "circuit_execution_seconds", "classical_seconds"]

def read_original_hyperparam(original_results_csv, seed, detector):
    """Pulls e.g. 'K=3' -> 3 out of the ORIGINAL 8-qubit run's own results CSV."""
    df = pd.read_csv(original_results_csv)
    row = df[(df["seed"] == seed) & (df["detector"] == detector)]
    assert len(row) == 1, f"expected exactly 1 original row for seed={seed} detector={detector}, got {len(row)}"
    hp = row["hyperparam"].iloc[0]
    if pd.isna(hp) or hp == "":
        return None
    return int(str(hp).split("=")[1])



def _run_distance_natural(args):

        os.makedirs(args.output_dir, exist_ok=True)
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"distance_fixed_extractor_{args.dataset}_normcls{args.norm_cls}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        input_resolution = "16x16"
        num_latent, num_trash = args.num_latent, args.num_trash

        already_done = set()
        if args.force:
            print("--force given: recomputing regardless of existing results")
        elif os.path.exists(results_csv):
            _prev = pd.read_csv(results_csv)
            already_done = set(zip(_prev["circuit"], _prev["input_resolution"], _prev["n_qubits"],
                                   _prev["feature_loss"], _prev["extractor_seed"], _prev["seed"], _prev["detector"]))

        if args.dataset == "mnist":
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)
        print(f"train_data_scale={args.train_data_scale} -> n_train={n_train} "
              f"(of {_class_count} available {args.norm_cls}-class training images)")

        # ---- load the ONE fixed extractor (extractor_seed's checkpoint), never retrained ----
        extractor_run_tag = build_run_tag_base(
            args.dataset, args.norm_cls, n_train, args.train_data_scale, args.circuit, args.readout,
            args.feature_loss, args.lr, args.batch_size, args.hcqc_unitary, args.drnn_ent_train,
            args.drnn_scaling, args.extractor_seed,
            args.vicreg_lambda_inv, args.vicreg_lambda_var, args.vicreg_lambda_cov, args.vicreg_gamma)
        checkpoint_path = find_resume_checkpoint(args.extractor_dir, extractor_run_tag)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No pretrained {args.circuit} checkpoint found for extractor_seed={args.extractor_seed} "
                f"matching run_tag_base='{extractor_run_tag}' under '{args.extractor_dir}'. Train it first "
                f"(e.g. rq1_feature_extractor/run_rq1.sh)."
            )
        print(f"Loading FIXED {args.circuit} extractor (extractor_seed={args.extractor_seed}) from '{checkpoint_path}' "
              f"-- reused as-is for all evaluation seeds {args.seeds}, never retrained")
        extractor = QuantumFeatureExtractor.from_checkpoint(
            checkpoint_path, args.circuit, args.readout, args.hcqc_unitary, args.drnn_ent_train, args.drnn_scaling)
        n_qubits = extractor.n_qubits

        all_rows = []
        for seed in args.seeds:
            missing_detectors = [d for d in args.detectors
                                  if (args.circuit, input_resolution, n_qubits, args.feature_loss,
                                      args.extractor_seed, seed, d) not in already_done]
            if not missing_detectors:
                print(f"seed={seed}: all requested detectors {args.detectors} already in '{results_csv}', skipping")
                continue

            set_global_seed(seed)

            # each evaluation seed resamples its own split (project convention);
            # the extractor embedding it, however, never changes.
            train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, _, _, _ = \
                load_ood_split_scaled(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, num_latent, num_trash, seed=seed)
            _unused_train_pool, _unused_val_imgs, _unused_val_labels, test_imgs, test_labels, _, _, _ = \
                load_ood_split_scaled(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.test_data_scale, num_latent, num_trash, seed=seed)

            # train_embs/val_embs are used only for classical model-selection (select_k/select_m)
            # and are NOT timed as circuit_execution_seconds -- on real hardware this selection is
            # prepared locally from the training/val split, never run on the QPU. Only the TEST-side
            # embedding circuit (test_embs) is a fair analog of hardware "inference" cost, and it is
            # SHARED across QKNN/QMean/QMedoids (one circuit pass, reused three times).
            with torch.no_grad():
                train_embs = extractor.get_embedding(torch.as_tensor(train_pool_imgs, dtype=torch.float64)).numpy()
                val_embs = extractor.get_embedding(torch.as_tensor(val_imgs, dtype=torch.float64)).numpy()
                t0 = time.perf_counter()
                test_embs = extractor.get_embedding(torch.as_tensor(test_imgs, dtype=torch.float64)).numpy()
                embedding_circuit_seconds = time.perf_counter() - t0

            # classical_seconds times ONLY the test-set scoring call (matches what the real-hardware
            # side measures as classical post-processing), never model-selection (select_k/select_m)
            # or QSVDD fine-tuning/checkpoint-loading, which are one-off classical fitting costs.
            scored = {}
            if "QKNN" in missing_detectors:
                best_k, _ = select_k(train_embs, val_embs, val_labels, args.k_candidates)
                t0 = time.perf_counter()
                knn_s = knn_scores(train_embs, test_embs, best_k)
                scored["QKNN"] = (knn_s, f"K={best_k}", embedding_circuit_seconds, time.perf_counter() - t0)
            if "QMean" in missing_detectors:
                t0 = time.perf_counter()
                mean_s = mean_scores(train_embs, test_embs)
                scored["QMean"] = (mean_s, "", embedding_circuit_seconds, time.perf_counter() - t0)
            if "QMedoids" in missing_detectors:
                best_m, _ = select_m(train_embs, val_embs, val_labels, args.m_candidates, seed=seed)
                t0 = time.perf_counter()
                medoid_s = medoid_scores(train_embs, test_embs, best_m, seed=seed)
                scored["QMedoids"] = (medoid_s, f"M={best_m}", embedding_circuit_seconds, time.perf_counter() - t0)
            if "QSVDD" in missing_detectors:
                # SVDD ALWAYS fits its own head fresh per evaluation seed -- its own
                # checkpoint is keyed by the evaluation seed, never extractor_seed,
                # so it is never cached/shared across seeds the way the base
                # extractor is (that sharing is the whole point here; SVDD's isn't).
                # It also uses its OWN fine-tuned extractor (not the shared `extractor` above), so
                # its circuit_execution_seconds is its own predict_score() timing, not embedding_circuit_seconds.
                svdd_checkpoint_path = os.path.join(
                    args.output_dir,
                    f"qcl_family_quanforge_qsvdd_checkpoint_{extractor_run_tag}_evalseed{seed}"
                    f"_svdd{args.svdd_epochs}ep_lr{args.svdd_lr}_lam{args.svdd_lambda}.pt"
                )
                svdd = QSVDDDetector(extractor)
                svdd.initialize_center(train_pool_imgs)
                svdd.train_svdd(train_pool_imgs, args.svdd_epochs, args.svdd_lr, args.batch_size, args.svdd_lambda,
                                 save_path=svdd_checkpoint_path, force_retrain=args.svdd_force_retrain)
                svdd_s, svdd_circuit_s, svdd_classical_s = svdd.predict_score(test_imgs, return_timing=True)
                scored["QSVDD"] = (svdd_s, f"lambda={args.svdd_lambda}", svdd_circuit_s, svdd_classical_s)

            print(f"\n{'Detector':<10} {'Hyperparam':<12} {'AUC-ROC':<10} {'AUC-PR':<10} {'circuit_s':<10} {'classical_s':<10}")
            for name, (scores, hp, circuit_seconds, classical_seconds) in scored.items():
                auc_roc = roc_auc_score(test_labels, scores)
                auc_pr = average_precision_score(test_labels, scores)
                print(f"seed={seed} {name:<10} {hp:<12} {auc_roc:<10.4f} {auc_pr:<10.4f} "
                      f"{circuit_seconds:<10.4f} {classical_seconds:<10.4f}")
                all_rows.append({"circuit": args.circuit, "input_resolution": input_resolution, "n_qubits": n_qubits,
                                 "feature_loss": args.feature_loss, "extractor_seed": args.extractor_seed, "seed": seed,
                                 "detector": name, "hyperparam": hp, "auc_roc": auc_roc, "auc_pr": auc_pr,
                                 "circuit_execution_seconds": circuit_seconds, "classical_seconds": classical_seconds})

        if all_rows:
            new_df = pd.DataFrame(all_rows, columns=DISTANCE_RESULTS_COLUMNS)
            new_df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
            full_df = pd.read_csv(results_csv)
            full_df = full_df.drop_duplicates(
                subset=["circuit", "input_resolution", "n_qubits", "feature_loss", "extractor_seed", "seed", "detector"],
                keep="last")
            full_df.to_csv(results_csv, index=False)
        else:
            full_df = pd.read_csv(results_csv) if os.path.exists(results_csv) else pd.DataFrame(columns=DISTANCE_RESULTS_COLUMNS)

        selected = full_df[(full_df["circuit"] == args.circuit) & (full_df["input_resolution"] == input_resolution) &
                            (full_df["n_qubits"] == n_qubits) & (full_df["feature_loss"] == args.feature_loss) &
                            (full_df["extractor_seed"] == args.extractor_seed) & (full_df["seed"].isin(args.seeds))]
        if not selected.empty:
            summary = (selected.groupby(["circuit", "input_resolution", "n_qubits", "feature_loss", "extractor_seed",
                                          "detector"], as_index=False)
                       .agg(n_seeds=("seed", "nunique"),
                            auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
                            auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std")))
            summary_path = os.path.splitext(results_csv)[0] + "_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"\nSummary (mean/std over {len(args.seeds)} eval seeds, extractor_seed={args.extractor_seed}) "
                  f"saved to '{summary_path}'")
            print(summary.to_string(index=False))


def _run_distance_adversarial(args):

        assert args.attack is not None, "--attack is required when using --adversarial_dir"
        os.makedirs(args.output_dir, exist_ok=True)
        setting_tag = args.attack
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"quantum_distance_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        if args.dataset == "mnist":
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)

        # ---- load the ONE fixed extractor (extractor_seed's checkpoint), never retrained ----
        extractor_run_tag = build_run_tag_base(
            args.dataset, args.norm_cls, n_train, args.train_data_scale, args.circuit, args.readout,
            args.feature_loss, args.lr, args.batch_size, args.hcqc_unitary, args.drnn_ent_train,
            args.drnn_scaling, args.extractor_seed,
            args.vicreg_lambda_inv, args.vicreg_lambda_var, args.vicreg_lambda_cov, args.vicreg_gamma)
        checkpoint_path = find_resume_checkpoint(args.extractor_dir, extractor_run_tag)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No pretrained {args.circuit} checkpoint found for extractor_seed={args.extractor_seed} "
                f"matching run_tag_base='{extractor_run_tag}' under '{args.extractor_dir}'. Train it first "
                f"(rq1_feature_extractor/run_rq1.sh)."
            )
        print(f"Loading FIXED {args.circuit} extractor (extractor_seed={args.extractor_seed}) from '{checkpoint_path}' "
              f"-- reused as-is for all evaluation seeds {args.seeds}, never retrained")
        extractor = QuantumFeatureExtractor.from_checkpoint(
            checkpoint_path, args.circuit, args.readout, args.hcqc_unitary, args.drnn_ent_train, args.drnn_scaling)

        # ---- clean ID-only train_pool, IDENTICAL raw images to the classical family's own run ----
        # (train_pool_idx selection is the same "first n_train images in dataset order" for every
        # script in this project, quantum or classical -- see classical_OOD_detectors.py's density subcommand's docstring)
        train_pool_imgs_raw, _val_imgs, _val_labels, _unused_test_imgs, _unused_test_labels, _ = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=0)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs_raw, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                              args.attack)
        print(f"train_pool={len(train_pool_imgs_raw)}, test={len(test_imgs_raw)} "
              f"(ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        # ---- quantum-specific preprocessing: L2-normalize + flatten, matching every other quantum script ----
        train_pool_norm = normalize_images(train_pool_imgs_raw)
        test_imgs_norm = normalize_images(test_imgs_raw)

        with torch.no_grad():
            train_embs = extractor.get_embedding(torch.as_tensor(train_pool_norm, dtype=torch.float64)).numpy()
            test_embs = extractor.get_embedding(torch.as_tensor(test_imgs_norm, dtype=torch.float64)).numpy()

        all_rows = []
        for seed in args.seeds:
            set_global_seed(seed)
            print(f"\n-- {args.circuit} / seed={seed} --")

            if "QKNN" in args.detectors:
                best_k = read_original_hyperparam(args.original_results_csv, seed, "QKNN")
                scores = knn_scores(train_embs, test_embs, best_k)
                auc_roc, auc_pr = roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores)
                print(f"QKNN         K={best_k:<6} AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": "QKNN", "hyperparam": f"K={best_k}",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

            if "QMean" in args.detectors:
                scores = mean_scores(train_embs, test_embs)
                auc_roc, auc_pr = roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores)
                print(f"QMean                AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": "QMean", "hyperparam": "",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

            if "QMedoids" in args.detectors:
                best_m = read_original_hyperparam(args.original_results_csv, seed, "QMedoids")
                scores = medoid_scores(train_embs, test_embs, best_m, seed=seed)
                auc_roc, auc_pr = roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores)
                print(f"QMedoids     M={best_m:<6} AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": "QMedoids", "hyperparam": f"M={best_m}",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

            if "QSVDD" in args.detectors:
                # refit fresh per eval seed on the clean ID train_pool only -- identical
                # convention to quantum_OOD_detectors.py's distance subcommand's own QSVDD (its checkpoint
                # is always keyed by the evaluation seed there too, never shared/cached
                # across seeds the way the extractor is).
                svdd_checkpoint_path = os.path.join(
                    args.output_dir,
                    f"quantum_distance_adversarial_qsvdd_checkpoint_{extractor_run_tag}_evalseed{seed}"
                    f"_svdd{args.svdd_epochs}ep_lr{args.svdd_lr}_lam{args.svdd_lambda}.pt"
                )
                svdd = QSVDDDetector(extractor)
                svdd.initialize_center(train_pool_norm)
                svdd.train_svdd(train_pool_norm, args.svdd_epochs, args.svdd_lr, args.svdd_batch_size, args.svdd_lambda,
                                 save_path=svdd_checkpoint_path)
                scores = svdd.predict_score(test_imgs_norm)
                auc_roc, auc_pr = roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores)
                print(f"QSVDD                AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": "QSVDD", "hyperparam": f"lambda={args.svdd_lambda}",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

        df = pd.DataFrame(all_rows, columns=ADV_RESULTS_COLUMNS)
        df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "eval_setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def run_distance(args):
    if getattr(args, "adversarial_dir", None):
        _run_distance_adversarial(args)
    else:
        _run_distance_natural(args)


# ============================================================================
# Density-based OOD detectors on a fixed quantum feature extractor (DMKDE-mixed, IndepGaussian, MVGaussian)
# ============================================================================

class FourierFeatureMap(nn.Module):
    """z(x) = sqrt(2/d) * cos((sqrt(2*gamma)*W) @ x + b), |psi> = z(x)/|z(x)|.
    RFF: W, b fixed at their random init (trainable=False).
    AFF: W, b trained (trainable=True) via train_aff() below.
    """

    def __init__(self, D, d, gamma, trainable=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W_init = torch.randn(d, D, generator=g, dtype=torch.float64) * np.sqrt(2.0 * gamma)
        b_init = torch.rand(d, generator=g, dtype=torch.float64) * 2 * np.pi
        if trainable:
            self.W = nn.Parameter(W_init)
            self.b = nn.Parameter(b_init)
        else:
            self.register_buffer("W", W_init)
            self.register_buffer("b", b_init)
        self.d = d

    def forward(self, x):
        z = np.sqrt(2.0 / self.d) * torch.cos(x @ self.W.t() + self.b)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-12)


def train_aff(feature_map, X, gamma_s, epochs, lr, batch_size, seed=0):
    """Adaptive Fourier Features training (Sec 4.1): two random shuffles of
    the ID training set X, synthetic labels = the (standard-sign, see
    module docstring point (a)) Gaussian kernel between the shuffled pairs
    (Eq 5), trained via MSE against the AFF-approximated kernel
    |<psi_1|psi_2>|^2 (Eq 7). X: (N, D) ID-only training data (no OOD
    labels used, matching the paper's unsupervised setup)."""
    optimizer = torch.optim.Adam(feature_map.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    loss_rec = []
    for ep in range(epochs):
        perm1 = rng.permutation(N)
        perm2 = rng.permutation(N)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            idx1 = perm1[start:start + batch_size]
            idx2 = perm2[start:start + batch_size]
            x1 = torch.as_tensor(X[idx1], dtype=torch.float64)
            x2 = torch.as_tensor(X[idx2], dtype=torch.float64)
            y = torch.exp(-gamma_s * torch.sum((x1 - x2) ** 2, dim=-1))  # see docstring (a)

            optimizer.zero_grad()
            psi1 = feature_map(x1)
            psi2 = feature_map(x2)
            overlap_sq = torch.sum(psi1 * psi2, dim=-1) ** 2
            loss = torch.mean((y - overlap_sq) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        loss_rec.append(avg_loss)
        print(f"AFF epoch {ep + 1}/{epochs}: mse_loss={avg_loss:.6f}")
    return loss_rec


# ====================== 2. DMKDE quantum circuits (Sec 3.2, 4.2) ======================
def build_dmkde_pure_circuit(n_qubits):
    """Fig 1: prepare |psi>_n, apply U_n^dagger (U_n|0>=|phi_train>), and
    read P(|0>_n) = |<phi_train|psi>|^2. p_hat_1(psi) = sqrt(P(|0>_n))
    (dropping the normalizing constant C_gamma,1)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(psi, phi_train):
        qml.AmplitudeEmbedding(psi, wires=range(n_qubits), normalize=True, pad_with=0.0)
        qml.adjoint(qml.AmplitudeEmbedding)(phi_train, wires=range(n_qubits), normalize=True, pad_with=0.0)
        return qml.probs(wires=range(n_qubits))

    return circuit


def build_dmkde_mixed_circuit(n_qubits):
    dev = qml.device("default.qubit", wires=2 * n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(psi, V_dagger_padded, sqrt_eigs_padded):
        qml.AmplitudeEmbedding(psi, wires=range(n_qubits), normalize=True, pad_with=0.0)
        qml.QubitUnitary(V_dagger_padded, wires=range(n_qubits))
        qml.AmplitudeEmbedding(sqrt_eigs_padded, wires=range(n_qubits, 2 * n_qubits), normalize=True, pad_with=0.0)
        for i in range(n_qubits):
            qml.CNOT(wires=[i + n_qubits, i])
        return qml.probs(wires=range(n_qubits))

    return circuit


class DMKDEDetector:

    def __init__(self, mode="mixed"):
        assert mode in ("pure", "mixed")
        self.mode = mode
        self.n_qubits = None
        self.d = None
        self.circuit = None
        self.phi_train = None
        self.V_dagger_padded = None
        self.sqrt_eigs_padded = None

    def fit(self, train_embs):
        train_embs = np.asarray(train_embs, dtype=np.float64)
        train_embs = train_embs / (np.linalg.norm(train_embs, axis=-1, keepdims=True) + 1e-12)
        d = train_embs.shape[1]
        n_qubits = max(1, int(np.ceil(np.log2(d))))
        self.d, self.n_qubits = d, n_qubits

        if self.mode == "pure":
            phi = train_embs.sum(axis=0)
            self.phi_train = phi / (np.linalg.norm(phi) + 1e-12)
            self.circuit = build_dmkde_pure_circuit(n_qubits)
        else:
            rho = (train_embs.T @ train_embs) / train_embs.shape[0]  # Eq 2, mixed training density matrix
            eigvals, eigvecs = np.linalg.eigh(rho)  # ascending
            eigvals = np.clip(eigvals[::-1], 0.0, None)  # descending, clip numerical negatives
            eigvecs = eigvecs[:, ::-1]
            dim2n = 2 ** n_qubits
            V_dagger_padded = np.eye(dim2n, dtype=complex)
            V_dagger_padded[:d, :d] = eigvecs.conj().T  # Eq 10
            sqrt_eigs_padded = np.zeros(dim2n)
            sqrt_eigs_padded[:d] = np.sqrt(eigvals)
            self.V_dagger_padded = V_dagger_padded
            self.sqrt_eigs_padded = sqrt_eigs_padded
            self.circuit = build_dmkde_mixed_circuit(n_qubits)
        self._train_embs_cache = train_embs  # only used by score(verify=True)'s classical cross-check
        return self

    def score(self, embs, verify=False, return_timing=False):
        """Returns DENSITY estimates (higher = more ID-like)."""
        t_classical = 0.0
        t0 = time.perf_counter()
        embs = np.asarray(embs, dtype=np.float64)
        embs = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12)
        t_classical += time.perf_counter() - t0

        densities = []
        t_circuit = 0.0
        for psi in embs:
            if self.mode == "pure":
                t0 = time.perf_counter()
                probs = self.circuit(torch.as_tensor(psi, dtype=torch.float64),
                                      torch.as_tensor(self.phi_train, dtype=torch.float64))
                t_circuit += time.perf_counter() - t0
                t0 = time.perf_counter()
                d_val = float(torch.sqrt(torch.clamp(probs[0], min=0.0)))
                t_classical += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                probs = self.circuit(torch.as_tensor(psi, dtype=torch.float64),
                                      torch.as_tensor(self.V_dagger_padded, dtype=torch.complex128),
                                      torch.as_tensor(self.sqrt_eigs_padded, dtype=torch.float64))
                t_circuit += time.perf_counter() - t0
                t0 = time.perf_counter()
                d_val = float(probs[0])
                t_classical += time.perf_counter() - t0
            densities.append(d_val)
        t0 = time.perf_counter()
        densities = np.array(densities)
        t_classical += time.perf_counter() - t0

        if verify:
            classical = dmkde_score_classical(embs, self.mode, phi_train=self.phi_train,
                                               train_embs=None if self.mode == "pure" else self._train_embs_cache)
            max_err = np.max(np.abs(densities - classical))
            print(f"DMKDE circuit-vs-classical max abs error: {max_err:.2e}")
            assert max_err < 1e-6, "DMKDE quantum circuit disagrees with the classical closed-form -- see verify=True"
        if return_timing:
            return densities, t_circuit, t_classical
        return densities


def dmkde_score_classical(embs, mode, phi_train=None, train_embs=None):
    """Closed-form classical computation of the SAME quantity the quantum
    circuits above compute (Eq 3 without sqrt, Eq 4) -- used as a
    correctness cross-check (DMKDEDetector.score(..., verify=True)), and
    usable on its own as a plain classical baseline."""
    embs = np.asarray(embs, dtype=np.float64)
    embs = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12)
    if mode == "pure":
        return np.abs(embs @ phi_train)
    train_embs = np.asarray(train_embs, dtype=np.float64)
    train_embs = train_embs / (np.linalg.norm(train_embs, axis=-1, keepdims=True) + 1e-12)
    return np.mean(np.abs(embs @ train_embs.T) ** 2, axis=-1)


def build_independent_gaussian_circuit(d):
    """One RY rotation per feature dimension, target state cos(theta/2)|0>+
    sin(theta/2)|1> = ratio|0>+sqrt(1-ratio^2)|1> (theta=2*arccos(ratio)) --
    exactly the paper's stated R1 target state."""
    dev = qml.device("default.qubit", wires=d)

    @qml.qnode(dev, interface="torch")
    def circuit(ratios):
        for j in range(d):
            theta = 2 * torch.arccos(torch.clamp(ratios[j], -1.0, 1.0))
            qml.RY(theta, wires=j)
        return [qml.probs(wires=j) for j in range(d)]

    return circuit


class IndependentGaussianDetector:
    """Per-feature (diagonal-covariance) Gaussian NLL, computed via a
    genuine per-feature quantum rotation circuit for the quadratic term
    (module docstring (a)-(b)); the training-only log(sigma) constant is
    computed classically (module docstring (c))."""

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mu = None
        self.sigma = None
        self.d = None
        self.circuit = None

    def fit(self, train_embs):
        train_embs = np.asarray(train_embs, dtype=np.float64)
        self.mu = train_embs.mean(axis=0)
        self.sigma = train_embs.std(axis=0) + self.eps  # matches paper's sigma_j^2 = (1/M)sum(x-mu)^2
        self.d = train_embs.shape[1]
        self.circuit = build_independent_gaussian_circuit(self.d)
        return self

    def score(self, embs, return_timing=False):
        """Returns the NEGATIVE log-likelihood -ln p(x) (higher = more
        anomalous)."""
        t_classical = 0.0
        t0 = time.perf_counter()
        embs = np.asarray(embs, dtype=np.float64)
        const_term = 0.5 * self.d * np.log(2 * np.pi) + np.sum(np.log(self.sigma))
        t_classical += time.perf_counter() - t0

        scores = []
        t_circuit = 0.0
        for x in embs:
            t0 = time.perf_counter()
            ratios = torch.as_tensor((x - self.mu) / self.sigma, dtype=torch.float64)
            t_classical += time.perf_counter() - t0
            t0 = time.perf_counter()
            probs_per_qubit = self.circuit(ratios)
            t_circuit += time.perf_counter() - t0
            t0 = time.perf_counter()
            m1 = sum(float(p[0]) for p in probs_per_qubit)  # sum_j ratio_j^2
            scores.append(const_term + 0.5 * m1)
            t_classical += time.perf_counter() - t0
        if return_timing:
            return np.array(scores), t_circuit, t_classical
        return np.array(scores)


def independent_gaussian_score_classical(embs, mu, sigma):
    """Closed-form classical cross-check for IndependentGaussianDetector."""
    embs = np.asarray(embs, dtype=np.float64)
    d = embs.shape[1]
    const_term = 0.5 * d * np.log(2 * np.pi) + np.sum(np.log(sigma))
    quad = np.sum(((embs - mu) / sigma) ** 2, axis=-1)
    return const_term + 0.5 * quad


# ====================== 2. Multivariate Gaussian via QPE (Algorithm 4-5) ======================
def build_qpe_circuit(n_data_qubits, n_est_qubits, C, t):
    """n_data_qubits encode |z0> (amplitude-embedded, unit-normalized
    separately -- the caller rescales by |z0|^2 afterward); n_est_qubits
    are the QPE estimation register. U = exp(i*C*t); QPE writes each
    eigenvalue's phase (proportional to lambda_j) into the estimation
    register, entangled with |z0>'s overlap onto that eigenvector."""
    total_wires = n_est_qubits + n_data_qubits
    dev = qml.device("default.qubit", wires=total_wires)
    U = scipy.linalg.expm(1j * C * t)

    @qml.qnode(dev, interface="torch")
    def circuit(z0_unit):
        qml.AmplitudeEmbedding(z0_unit, wires=range(n_est_qubits, total_wires), normalize=True, pad_with=0.0)
        qml.QuantumPhaseEstimation(
            qml.QubitUnitary(U, wires=range(n_est_qubits, total_wires)),
            estimation_wires=range(n_est_qubits),
        )
        return qml.probs(wires=range(n_est_qubits))

    return circuit


class MultivariateGaussianDetector:
    """Full-covariance Gaussian NLL / Mahalanobis-distance detector, via
    quantum phase estimation on the covariance matrix."""

    def __init__(self, n_est_qubits=6, eps=1e-6):
        self.n_est_qubits = n_est_qubits
        self.eps = eps
        self.mu = None
        self.C = None
        self.eigvals = None
        self.eigvecs = None
        self.log_det_C = None
        self.n_data_qubits = None
        self.t = None
        self.circuit = None
        self.d = None

    def fit(self, train_embs):
        train_embs = np.asarray(train_embs, dtype=np.float64)
        self.d = train_embs.shape[1]
        self.mu = train_embs.mean(axis=0)
        centered = train_embs - self.mu
        self.C = (centered.T @ centered) / max(1, train_embs.shape[0] - 1) + self.eps * np.eye(self.d)  # Eq 3
        eigvals, eigvecs = np.linalg.eigh(self.C)
        self.eigvals = np.clip(eigvals, self.eps, None)
        self.eigvecs = eigvecs
        self.log_det_C = float(np.sum(np.log(self.eigvals)))  # Algorithm 5, module docstring (e)

        self.n_data_qubits = max(1, int(np.ceil(np.log2(self.d))))
        dim2n = 2 ** self.n_data_qubits
        if dim2n > self.d:
            C_for_circuit = np.eye(dim2n) * (self.eigvals.max() * 2)
            C_for_circuit[:self.d, :self.d] = self.C
        else:
            C_for_circuit = self.C
        self.t = np.pi / self.eigvals.max()  # keep phases in [0, 0.5], avoiding QPE wraparound
        self.circuit = build_qpe_circuit(self.n_data_qubits, self.n_est_qubits, C_for_circuit, self.t)
        return self

    def _qpe_mahalanobis(self, z0):
        """One test point's Mahalanobis distance via the QPE probability
        distribution: bin each measured phase to its nearest classically-
        known eigenvalue, then combine 1/lambda_j classically, weighted by the measured bin probability
        and rescaled by |z0|^2 (AmplitudeEmbedding only sees the unit
        direction, not the norm).

        Returns (value, circuit_seconds, classical_seconds)."""
        t0 = time.perf_counter()
        norm_sq = float(np.dot(z0, z0))
        if norm_sq < 1e-24:
            return 0.0, 0.0, time.perf_counter() - t0
        t_classical = time.perf_counter() - t0

        t0 = time.perf_counter()
        probs = self.circuit(torch.as_tensor(z0, dtype=torch.float64)).detach().numpy()
        t_circuit = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_bins = len(probs)
        bin_phases = np.arange(n_bins) / n_bins
        bin_lambdas = bin_phases * (2 * np.pi) / self.t
        # match each bin to the nearest of the d TRUE eigenvalues (padding
        # dimensions beyond d, if n_data_qubits implies 2**n_data_qubits > d,
        # correspond to lambda=0 -- excluded, since AmplitudeEmbedding pads
        # z0 with zeros there and those components carry no weight anyway)
        nearest_idx = np.argmin(np.abs(bin_lambdas[:, None] - self.eigvals[None, :]), axis=1)
        contrib = probs / self.eigvals[nearest_idx]
        value = float(np.sum(contrib)) * norm_sq
        t_classical += time.perf_counter() - t0
        return value, t_circuit, t_classical

    def score(self, embs, verify=True, return_timing=False):
        embs = np.asarray(embs, dtype=np.float64)
        d = embs.shape[1]
        const_term = 0.5 * d * np.log(2 * np.pi) + 0.5 * self.log_det_C
        z = embs - self.mu

        maha_results = [self._qpe_mahalanobis(zi) for zi in z]
        quantum_maha = np.array([r[0] for r in maha_results])
        t_circuit = sum(r[1] for r in maha_results)
        t_classical = sum(r[2] for r in maha_results)
        scores = const_term + 0.5 * quantum_maha

        if verify:
            classical_maha = multivariate_gaussian_score_classical(embs, self.mu, self.C, return_maha_only=True)
            max_err = np.max(np.abs(quantum_maha - classical_maha))
            rel_err = max_err / (np.max(np.abs(classical_maha)) + 1e-12)
            print(f"QPE Mahalanobis vs classical: max_abs_err={max_err:.4f}, max_rel_err={rel_err:.4f} "
                  f"(n_est_qubits={self.n_est_qubits} -- increase for finer QPE precision if this is large)")
        if return_timing:
            return scores, t_circuit, t_classical
        return scores


def multivariate_gaussian_score_classical(embs, mu, C, return_maha_only=False):
    """Closed-form classical Mahalanobis-distance/NLL, used as
    MultivariateGaussianDetector's default correctness cross-check, and
    usable standalone as a classical baseline."""
    embs = np.asarray(embs, dtype=np.float64)
    d = embs.shape[1]
    z = embs - mu
    C_inv = np.linalg.pinv(C)
    maha = np.einsum('ni,ij,nj->n', z, C_inv, z)
    if return_maha_only:
        return maha
    sign, logdet = np.linalg.slogdet(C)
    const_term = 0.5 * d * np.log(2 * np.pi) + 0.5 * logdet
    return const_term + 0.5 * maha


def score_density_detectors(train_embs, test_embs, test_labels, detectors, n_est_qubits, verify_mvg,
                             embedding_circuit_seconds=0.0):
    """embedding_circuit_seconds."""
    out = {}
    if "DMKDE-mixed" in detectors:
        det = DMKDEDetector(mode="mixed").fit(train_embs)
        densities, circuit_s, classical_s = det.score(test_embs, return_timing=True)
        scores = -densities  # higher score = more anomalous, matching this project's convention
        out["DMKDE-mixed"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                               "mode=mixed", embedding_circuit_seconds + circuit_s, classical_s)
    if "IndepGaussian" in detectors:
        det = IndependentGaussianDetector().fit(train_embs)
        scores, circuit_s, classical_s = det.score(test_embs, return_timing=True)  # already NLL, higher = more anomalous
        out["IndepGaussian"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                                 "", embedding_circuit_seconds + circuit_s, classical_s)
    if "MVGaussian" in detectors:
        det = MultivariateGaussianDetector(n_est_qubits=n_est_qubits).fit(train_embs)
        scores, circuit_s, classical_s = det.score(test_embs, verify=verify_mvg, return_timing=True)
        out["MVGaussian"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                              f"n_est_qubits={n_est_qubits}", embedding_circuit_seconds + circuit_s, classical_s)
    return out


DENSITY_RESULTS_COLUMNS = ["circuit", "input_resolution", "n_qubits", "feature_loss", "extractor_seed", "seed", "detector",
                           "hyperparam", "auc_roc", "auc_pr", "circuit_execution_seconds", "classical_seconds"]


def _run_density_natural(args):

        if args.seeds is not None:
            run_seeds = list(dict.fromkeys(args.seeds))
        else:
            if args.num_seeds < 1:
                parser.error("--num_seeds must be >= 1")
            run_seeds = list(range(args.seed, args.seed + args.num_seeds))
        print(f"Seeds to run: {run_seeds}")

        os.makedirs(args.output_dir, exist_ok=True)
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"density_{args.dataset}_normcls{args.norm_cls}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        input_resolution = "16x16"
        num_latent, num_trash = 6, 2
        n_qubits = 8

        already_done = set()
        if args.force:
            print("--force given: recomputing regardless of existing results")
        elif os.path.exists(results_csv):
            _prev = pd.read_csv(results_csv)
            if "n_qubits" not in _prev.columns:
                _prev["n_qubits"] = 8
            if "extractor_seed" not in _prev.columns:
                _prev["extractor_seed"] = _prev["seed"]
            already_done = set(zip(_prev["circuit"], _prev["input_resolution"], _prev["n_qubits"],
                                   _prev["feature_loss"], _prev["extractor_seed"], _prev["seed"], _prev["detector"]))

        if args.dataset == "mnist":
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)
        print(f"train_data_scale={args.train_data_scale} -> n_train={n_train} "
              f"(of {_class_count} available {args.norm_cls}-class training images)")

        def load_extractor(extractor_seed):
            run_tag_base = build_run_tag_base(
                args.dataset, args.norm_cls, n_train, args.train_data_scale, args.circuit, args.readout,
                args.feature_loss, args.lr, args.batch_size, args.hcqc_unitary, args.drnn_ent_train,
                args.drnn_scaling, extractor_seed,
                args.vicreg_lambda_inv, args.vicreg_lambda_var, args.vicreg_lambda_cov, args.vicreg_gamma)
            checkpoint_path = find_resume_checkpoint(args.extractor_dir, run_tag_base)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No pretrained {args.circuit} checkpoint found for extractor_seed={extractor_seed} matching "
                    f"run_tag_base='{run_tag_base}' under '{args.extractor_dir}'. Density detectors reuse the SAME "
                    f"pretrained extractor as the distance-based sweep -- run "
                    f"rq1_feature_extractor/run_rq1.sh (or run_quanforge_pipeline_mnist.py directly) for "
                    f"--dataset {args.dataset} --norm_cls {args.norm_cls} first."
                )
            print(f"Loading pretrained {args.circuit} extractor (extractor_seed={extractor_seed}) from '{checkpoint_path}'")
            return QuantumFeatureExtractor.from_checkpoint(
                checkpoint_path, args.circuit, args.readout, args.hcqc_unitary, args.drnn_ent_train, args.drnn_scaling)

        # if --extractor_seed is fixed, load it ONCE and reuse for every evaluation
        # seed below; otherwise each seed loads its own checkpoint (original behavior).
        fixed_extractor = load_extractor(args.extractor_seed) if args.extractor_seed is not None else None

        all_rows = []
        for seed in run_seeds:
            extractor_seed = args.extractor_seed if args.extractor_seed is not None else seed
            missing_detectors = [d for d in args.detectors
                                  if (args.circuit, input_resolution, n_qubits, args.feature_loss,
                                      extractor_seed, seed, d) not in already_done]
            if not missing_detectors:
                print(f"seed={seed}: all requested detectors {args.detectors} already in '{results_csv}', skipping")
                continue

            set_global_seed(seed)

            # Each evaluation seed resamples its own split (this project's standard
            # per-seed convention); the extractor embedding it may or may not
            # change depending on whether --extractor_seed was fixed above.
            train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, _, _, _ = \
                load_ood_split_scaled(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, num_latent, num_trash, seed=seed)
            _unused_train_pool, _unused_val_imgs, _unused_val_labels, test_imgs, test_labels, _, _, _ = \
                load_ood_split_scaled(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.test_data_scale, num_latent, num_trash, seed=seed)

            # train_embs (used only for fitting mu/covariance/density-matrix, all classical) is NOT
            # timed as circuit_execution_seconds -- on real hardware this fitting is prepared locally
            # from the training split, never run on the QPU; only
            # the TEST-side embedding circuit is a fair analog of hardware "inference" cost.
            extractor = fixed_extractor if fixed_extractor is not None else load_extractor(seed)
            with torch.no_grad():
                train_embs = extractor.get_embedding(torch.as_tensor(train_pool_imgs, dtype=torch.float64)).numpy()
                t0 = time.perf_counter()
                test_embs = extractor.get_embedding(torch.as_tensor(test_imgs, dtype=torch.float64)).numpy()
                embedding_circuit_seconds = time.perf_counter() - t0

            out = score_density_detectors(train_embs, test_embs, test_labels, missing_detectors,
                                           args.n_est_qubits, verify_mvg=not args.no_verify_mvg,
                                           embedding_circuit_seconds=embedding_circuit_seconds)
            print(f"\n{'Detector':<14} {'Hyperparam':<18} {'AUC-ROC':<10} {'AUC-PR':<10} {'circuit_s':<10} {'classical_s':<10}")
            for det, (auc_roc, auc_pr, hp, circuit_s, classical_s) in out.items():
                print(f"{det:<14} {hp:<18} {auc_roc:<10.4f} {auc_pr:<10.4f} {circuit_s:<10.4f} {classical_s:<10.4f}")
                all_rows.append({"circuit": args.circuit, "input_resolution": input_resolution, "n_qubits": n_qubits,
                                 "feature_loss": args.feature_loss, "extractor_seed": extractor_seed, "seed": seed,
                                 "detector": det, "hyperparam": hp, "auc_roc": auc_roc, "auc_pr": auc_pr,
                                 "circuit_execution_seconds": circuit_s, "classical_seconds": classical_s})

        if all_rows:
            new_df = pd.DataFrame(all_rows, columns=DENSITY_RESULTS_COLUMNS)
            new_df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
            full_df = pd.read_csv(results_csv)
            if "n_qubits" not in full_df.columns:
                full_df["n_qubits"] = np.where(full_df["circuit"] == "DRNN", 6, 8)
            if "extractor_seed" not in full_df.columns:
                full_df["extractor_seed"] = full_df["seed"]
            full_df = full_df.drop_duplicates(
                subset=["circuit", "input_resolution", "n_qubits", "feature_loss", "extractor_seed", "seed", "detector"],
                keep="last")
            full_df.to_csv(results_csv, index=False)
        else:
            full_df = pd.read_csv(results_csv) if os.path.exists(results_csv) else pd.DataFrame(columns=DENSITY_RESULTS_COLUMNS)
            if "n_qubits" not in full_df.columns and not full_df.empty:
                full_df["n_qubits"] = np.where(full_df["circuit"] == "DRNN", 6, 8)
            if "extractor_seed" not in full_df.columns and not full_df.empty:
                full_df["extractor_seed"] = full_df["seed"]

        _extractor_seed_filter = args.extractor_seed if args.extractor_seed is not None else full_df["seed"]
        selected = full_df[(full_df["circuit"] == args.circuit) & (full_df["input_resolution"] == input_resolution) &
                            (full_df["n_qubits"] == n_qubits) &
                            (full_df["feature_loss"] == args.feature_loss) & (full_df["seed"].isin(run_seeds)) &
                            (full_df["extractor_seed"] == _extractor_seed_filter)]
        if not selected.empty:
            summary = (selected.groupby(["circuit", "input_resolution", "n_qubits", "feature_loss", "extractor_seed",
                                          "detector"], as_index=False)
                       .agg(n_seeds=("seed", "nunique"),
                            auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
                            auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std")))
            summary_path = os.path.splitext(results_csv)[0] + "_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"\nSummary (mean/std over {len(run_seeds)} seeds) saved to '{summary_path}'")
            print(summary.to_string(index=False))


def _run_density_adversarial(args):

        assert args.attack is not None, "--attack is required when using --adversarial_dir"
        os.makedirs(args.output_dir, exist_ok=True)
        setting_tag = args.attack
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"quantum_density_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        if args.dataset == "mnist":
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)

        # ---- load the ONE fixed extractor (extractor_seed's checkpoint), never retrained ----
        extractor_run_tag = build_run_tag_base(
            args.dataset, args.norm_cls, n_train, args.train_data_scale, args.circuit, args.readout,
            args.feature_loss, args.lr, args.batch_size, args.hcqc_unitary, args.drnn_ent_train,
            args.drnn_scaling, args.extractor_seed,
            args.vicreg_lambda_inv, args.vicreg_lambda_var, args.vicreg_lambda_cov, args.vicreg_gamma)
        checkpoint_path = find_resume_checkpoint(args.extractor_dir, extractor_run_tag)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No pretrained {args.circuit} checkpoint found for extractor_seed={args.extractor_seed} "
                f"matching run_tag_base='{extractor_run_tag}' under '{args.extractor_dir}'. Train it first "
                f"(rq1_feature_extractor/run_rq1.sh)."
            )
        print(f"Loading FIXED {args.circuit} extractor (extractor_seed={args.extractor_seed}) from '{checkpoint_path}' "
              f"-- reused as-is for all evaluation seeds {args.seeds}, never retrained")
        extractor = QuantumFeatureExtractor.from_checkpoint(
            checkpoint_path, args.circuit, args.readout, args.hcqc_unitary, args.drnn_ent_train, args.drnn_scaling)

        # ---- clean ID-only train_pool, IDENTICAL raw images to the other families' own runs ----
        train_pool_imgs_raw, _val_imgs, _val_labels, _unused_test_imgs, _unused_test_labels, _ = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=0)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs_raw, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                              args.attack)
        print(f"train_pool={len(train_pool_imgs_raw)}, test={len(test_imgs_raw)} "
              f"(ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        # ---- quantum-specific preprocessing: L2-normalize + flatten, matching every other quantum script ----
        train_pool_norm = normalize_images(train_pool_imgs_raw)
        test_imgs_norm = normalize_images(test_imgs_raw)

        with torch.no_grad():
            train_embs = extractor.get_embedding(torch.as_tensor(train_pool_norm, dtype=torch.float64)).numpy()
            test_embs = extractor.get_embedding(torch.as_tensor(test_imgs_norm, dtype=torch.float64)).numpy()

        all_rows = []
        for seed in args.seeds:
            print(f"\n-- {args.circuit} / seed={seed} --")
            out = score_density_detectors(train_embs, test_embs, test_labels, args.detectors,
                                           args.n_est_qubits, verify_mvg=not args.no_verify_mvg)
            for det, (auc_roc, auc_pr, hp, _infer_time) in out.items():
                print(f"{det:<14} {hp:<18} AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": det, "hyperparam": hp,
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

        df = pd.DataFrame(all_rows, columns=ADV_RESULTS_COLUMNS)
        df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "eval_setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def run_density(args):
    if getattr(args, "adversarial_dir", None):
        _run_density_adversarial(args)
    else:
        _run_density_natural(args)


DISTANCE_ALL_DETECTORS = ["QKNN", "QMean", "QMedoids", "QSVDD"]
DENSITY_ALL_DETECTORS = ["DMKDE-mixed", "IndepGaussian", "MVGaussian"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="detector_family", required=True)

    sub = subparsers.add_parser("qae", help="Quantum autoencoder (reconstruction) detector. Natural-shift training+eval by default; pass --adversarial_dir to evaluate an already-trained --checkpoint_dir on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--target_class", type=int, default=0)
    sub.add_argument("--run_all_id", action="store_true")
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2],
                   help="1: one-vs-all OOD, 2: one-vs-one (--ood_class)")
    sub.add_argument("--ood_class", type=int, default=1)
    sub.add_argument("--scale", type=float, default=0.2, help="Training fraction")
    sub.add_argument("--test_scale", type=float, default=1.0, help="ID test fraction")
    sub.add_argument("--img_size", type=int, default=16)
    sub.add_argument("--n_qubits", type=int, default=8)
    sub.add_argument("--n_layers", type=int, default=3)
    sub.add_argument("--rot_gate", type=str, default="RX+RZ", choices=["RX", "RY", "RX+RZ"])
    sub.add_argument("--entangle_gate", type=str, default="CRY", choices=["CNOT", "CRY"])
    sub.add_argument("--latent_dim", type=int, default=8)
    sub.add_argument("--classical_latent_dim", type=int, default=32)
    sub.add_argument("--model", type=str, default="both", choices=["qae", "classical", "both"])
    sub.add_argument("--epochs", type=int, default=100)
    sub.add_argument("--batch_size", type=int, default=64)
    sub.add_argument("--lr", type=float, default=1e-3)
    sub.add_argument("--grad_clip", type=float, default=5.0)
    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    sub.add_argument("--save_dir", type=str, default="results/qae_nq8_cry")
    sub.add_argument("--fast_test", action="store_true")
    sub.add_argument("--norm_cls", type=int, default=0)

    sub.add_argument("--checkpoint_dir", type=str, default=None,
                         help="directory holding the qae subcommand's per-run dirs for this dataset, e.g. "
                              "outputs/results_nq8/QAE/results/mnist")

    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    sub.add_argument("--output_dir", type=str, default="outputs/adversarial_eval")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/qae_adversarial_<dataset>_"
                              "normcls<norm_cls>_<attack>_results.csv")

    sub = subparsers.add_parser("qvae", help="Quantum VAE (reconstruction) detector. Natural-shift training+eval by default; pass --adversarial_dir to evaluate an already-trained --checkpoint_dir on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--target_class", type=int, default=0)
    sub.add_argument("--run_all_id", action="store_true")
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--ood_class", type=int, default=1)
    sub.add_argument("--scale", type=float, default=0.2)
    sub.add_argument("--test_scale", type=float, default=1.0)
    sub.add_argument("--img_size", type=int, default=16)
    sub.add_argument("--n_qubits", type=int, default=8)
    sub.add_argument("--n_layers", type=int, default=3)
    sub.add_argument("--rot_gate", type=str, default="RX+RZ", choices=["RX", "RY", "RX+RZ"])
    sub.add_argument("--entangle_gate", type=str, default="CRY", choices=["CNOT", "CRY"])
    sub.add_argument("--latent_dim", type=int, default=8)
    sub.add_argument("--classical_latent_dim", type=int, default=32)
    sub.add_argument("--beta", type=float, default=1.0, help="KL weight (beta-VAE)")
    sub.add_argument("--model", type=str, default="both", choices=["qvae", "classical", "both"])
    sub.add_argument("--epochs", type=int, default=100)
    sub.add_argument("--batch_size", type=int, default=64)
    sub.add_argument("--lr", type=float, default=1e-3)
    sub.add_argument("--grad_clip", type=float, default=5.0)
    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    sub.add_argument("--save_dir", type=str, default="results/qvae_nq8_cry")
    sub.add_argument("--fast_test", action="store_true")
    sub.add_argument("--norm_cls", type=int, default=0)

    sub.add_argument("--checkpoint_dir", type=str, default=None,
                         help="directory holding the qvae subcommand's per-run dirs for this dataset, e.g. "
                              "outputs/results_nq8/QVAE/results/mnist")

    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    sub.add_argument("--output_dir", type=str, default="outputs/adversarial_eval")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/qvae_adversarial_<dataset>_"
                              "normcls<norm_cls>_<attack>_results.csv")

    sub = subparsers.add_parser("gan", help="Quantum GAN-based detectors (Q-AnoGAN / QWGAN-GP). Natural-shift training+eval by default; pass --adversarial_dir to evaluate an already-trained --checkpoint_dir on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--target_class", type=int, default=0)
    sub.add_argument("--ood_class", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--scale", type=float, default=0.2)
    sub.add_argument("--test_scale", type=float, default=1.0)
    sub.add_argument("--img_size", type=int, default=16)
    sub.add_argument("--n_qubits", type=int, default=8)
    sub.add_argument("--n_layers", type=int, default=3)
    sub.add_argument("--rot_gate", type=str, default="RX+RZ", choices=["RX", "RY", "RX+RZ"])
    sub.add_argument("--entangle_gate", type=str, default="CRY", choices=["CNOT", "CRY"])
    sub.add_argument("--gan_type", type=str, default="both", choices=["q_anogan", "qwgan_gp", "both"])
    sub.add_argument("--epochs", type=int, default=100)
    sub.add_argument("--batch_size", type=int, default=64)
    sub.add_argument("--lr_g", type=float, default=1e-3)
    sub.add_argument("--lr_d", type=float, default=1e-3)
    sub.add_argument("--n_critic", type=int, default=5)
    sub.add_argument("--lambda_gp", type=float, default=10.0)
    sub.add_argument("--latent_dim", type=int, default=8)
    sub.add_argument("--z_iter", type=int, default=500)
    sub.add_argument("--z_lr", type=float, default=1e-2)
    sub.add_argument("--alpha", type=float, default=0.5)
    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    sub.add_argument("--run_all_id", action="store_true")
    sub.add_argument("--save_dir", type=str, default="results/quantum_gan_ood_nq8_cry")
    sub.add_argument("--fast_test", action="store_true")
    sub.add_argument("--grad_clip", type=float, default=0.0)
    sub.add_argument("--bce_eps", type=float, default=0.0)
    sub.add_argument("--norm_cls", type=int, default=0)

    sub.add_argument("--checkpoint_dir", type=str, default=None,
                         help="directory holding the gan subcommand's per-run dirs for this dataset, e.g. "
                              "outputs/results_nq8/Gan-based/results/qanogan_nq8/mnist")

    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")

    sub.add_argument("--gan_types", nargs="+", type=str, default=["q_anogan", "qwgan_gp"],
                         choices=["q_anogan", "qwgan_gp"])
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    sub.add_argument("--output_dir", type=str, default="outputs/adversarial_eval")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/quantum_gan_adversarial_<dataset>_"
                              "normcls<norm_cls>_<attack>_results.csv")

    sub = subparsers.add_parser("ganomaly", help="Quantum/classical GANomaly detector. Natural-shift training+eval by default; pass --adversarial_dir to evaluate an already-trained --checkpoint_dir on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--target_class", type=int, default=0)
    sub.add_argument("--ood_class", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--run_all_id", action="store_true")
    sub.add_argument("--scale", type=float, default=0.2)
    sub.add_argument("--test_scale", type=float, default=1.0)
    sub.add_argument("--img_size", type=int, default=16)
    sub.add_argument("--n_qubits", type=int, default=8)
    sub.add_argument("--n_layers", type=int, default=3)
    sub.add_argument("--latent_dim", type=int, default=8)
    sub.add_argument("--entangle_gate", type=str, default="CRY", choices=["CNOT", "CRY"])
    sub.add_argument("--classical_latent_dim", type=int, default=32)
    sub.add_argument("--model", type=str, default="both", choices=["qganomaly", "classical", "both"])
    sub.add_argument("--epochs", type=int, default=100)
    sub.add_argument("--batch_size", type=int, default=64)
    sub.add_argument("--lr_g", type=float, default=1e-3)
    sub.add_argument("--lr_d", type=float, default=1e-3)
    sub.add_argument("--w_rec", type=float, default=10.0)
    sub.add_argument("--w_lat", type=float, default=20.0)
    sub.add_argument("--w_adv", type=float, default=0.5)
    sub.add_argument("--grad_clip", type=float, default=5.0)
    sub.add_argument("--bce_eps", type=float, default=1e-7)
    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    sub.add_argument("--save_dir", type=str, default="results/qganomaly_nq8_cry")
    sub.add_argument("--fast_test", action="store_true")
    sub.add_argument("--norm_cls", type=int, default=0)

    sub.add_argument("--checkpoint_dir", type=str, default=None,
                         help="directory holding the ganomaly subcommand's per-run dirs for this dataset, e.g. "
                              "outputs/results_nq8/Gan-based/results/qganomaly_nq8/mnist")

    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")

    sub.add_argument("--variants", nargs="+", type=str, default=["qganomaly", "classical"],
                         choices=["qganomaly", "classical"])
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    sub.add_argument("--output_dir", type=str, default="outputs/adversarial_eval")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/ganomaly_adversarial_<dataset>_"
                              "normcls<norm_cls>_<attack>_results.csv")

    sub = subparsers.add_parser("distance", help="Distance-based detectors (QKNN, QMean, QMedoids, QSVDD) on a fixed quantum feature extractor. Natural-shift sweep by default; pass --adversarial_dir (and --original_results_csv) to evaluate on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--norm_cls", type=int, default=0)
    sub.add_argument("--ood_cls", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--train_data_scale", type=float, default=0.2,
                         help="must match the value used to train the checkpoint being reused")
    sub.add_argument("--test_data_scale", type=float, default=1.0)
    sub.add_argument("--n_val_per_cls", type=int, default=20)
    sub.add_argument("--num_latent", type=int, default=6)
    sub.add_argument("--num_trash", type=int, default=2)
    sub.add_argument("--circuit", type=str, default="DRNN", choices=["QCL", "QCNN", "HCQC", "DRNN"])
    sub.add_argument("--readout", type=str, default="expval", choices=["probs", "expval"])
    sub.add_argument("--feature_loss", type=str, default="compact",
                         choices=["random", "cosine", "vicreg", "compact"])
    sub.add_argument("--lr", type=float, default=1e-2, help="checkpoint-tag matching only, not used for training here")
    sub.add_argument("--batch_size", type=int, default=32, help="checkpoint-tag matching only")
    sub.add_argument("--hcqc_unitary", type=str, default="U_SU4",
                         choices=["U_TTN", "U_5", "U_6", "U_9", "U_13", "U_14", "U_15", "U_SO4", "U_SU4"])
    sub.add_argument("--drnn_ent_train", action="store_true")
    sub.add_argument("--drnn_scaling", type=float, default=1.5)
    sub.add_argument("--vicreg_lambda_inv", type=float, default=25.0)
    sub.add_argument("--vicreg_lambda_var", type=float, default=25.0)
    sub.add_argument("--vicreg_lambda_cov", type=float, default=1.0)
    sub.add_argument("--vicreg_gamma", type=float, default=1.0)
    sub.add_argument("--detectors", nargs="+", type=str, default=DISTANCE_ALL_DETECTORS, choices=DISTANCE_ALL_DETECTORS)
    sub.add_argument("--k_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    sub.add_argument("--m_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    sub.add_argument("--svdd_epochs", type=int, default=10)
    sub.add_argument("--svdd_lr", type=float, default=1e-3)
    sub.add_argument("--svdd_lambda", type=float, default=1e-3)
    sub.add_argument("--svdd_force_retrain", action="store_true")

    sub.add_argument("--extractor_seed", type=int, default=0,
                         help="which seed's already-trained extractor checkpoint to reuse for ALL "
                              "evaluation seeds below -- the extractor is loaded ONCE and never retrained")
    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                         help="evaluation seeds -- each resamples its own train/val/test split and "
                              "refits K/M/QSVDD fresh, but all share the SAME frozen extractor above")
    sub.add_argument("--extractor_dir", type=str, default="outputs/qcl_family_quanforge",
                         help="where the already-trained checkpoint lives "
                              "(run_quanforge_pipeline_mnist.py's --output_dir)")
    sub.add_argument("--output_dir", type=str, default="outputs/qcl_family_quanforge")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/distance_fixed_extractor_<dataset>_normcls<norm_cls>_results.csv")
    sub.add_argument("--force", action="store_true", help="recompute even if already in --results_csv")
    sub.add_argument("--img_size", type=int, default=16)

    sub.add_argument("--original_results_csv", type=str, default=None,
                         help="the ORIGINAL clean-OOD 8-qubit run's results CSV to read already-selected K/M "
                              "from, e.g. outputs/QNN-8-qubit-results/distance_fixed_extractor_"
                              "<dataset>_normcls<norm_cls>_nq8_results.csv")
    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")
    sub.add_argument("--svdd_batch_size", type=int, default=32)

    sub = subparsers.add_parser("density", help="Density-based detectors (DMKDE-mixed, IndepGaussian, MVGaussian) on a fixed quantum feature extractor. Natural-shift sweep by default; pass --adversarial_dir to evaluate on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--norm_cls", type=int, default=0)
    sub.add_argument("--ood_cls", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--train_data_scale", type=float, default=0.2,
                         help="must match the value used to train the DRNN checkpoints being reused")
    sub.add_argument("--test_data_scale", type=float, default=1.0)
    sub.add_argument("--n_val_per_cls", type=int, default=20)
    sub.add_argument("--circuit", type=str, default="DRNN", choices=["QCL", "QCNN", "HCQC", "DRNN"])
    sub.add_argument("--readout", type=str, default="expval", choices=["probs", "expval"])
    sub.add_argument("--feature_loss", type=str, default="compact",
                         choices=["random", "cosine", "vicreg", "compact"])
    sub.add_argument("--lr", type=float, default=1e-2, help="checkpoint-tag matching only, not used for training here")
    sub.add_argument("--batch_size", type=int, default=32, help="checkpoint-tag matching only")
    sub.add_argument("--hcqc_unitary", type=str, default="U_SU4",
                         choices=["U_TTN", "U_5", "U_6", "U_9", "U_13", "U_14", "U_15", "U_SO4", "U_SU4"])
    sub.add_argument("--drnn_ent_train", action="store_true")
    sub.add_argument("--drnn_scaling", type=float, default=1.5)
    sub.add_argument("--vicreg_lambda_inv", type=float, default=25.0)
    sub.add_argument("--vicreg_lambda_var", type=float, default=25.0)
    sub.add_argument("--vicreg_lambda_cov", type=float, default=1.0)
    sub.add_argument("--vicreg_gamma", type=float, default=1.0)
    sub.add_argument("--detectors", nargs="+", type=str, default=DENSITY_ALL_DETECTORS, choices=DENSITY_ALL_DETECTORS)
    sub.add_argument("--n_est_qubits", type=int, default=6, help="MVGaussian's QPE precision register size")
    sub.add_argument("--no_verify_mvg", action="store_true",
                         help="skip MVGaussian's classical Mahalanobis cross-check (default: verify)")

    sub.add_argument("--seed", type=int, default=0, help="single seed, or starting seed when --num_seeds > 1")
    sub.add_argument("--num_seeds", type=int, default=1)
    sub.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="explicit seed list, e.g. --seeds 0 1 2 3 4; overrides --seed/--num_seeds")
    sub.add_argument("--extractor_seed", type=int, default=None,
                         help="if given, load ONLY this seed's checkpoint once and reuse it for every "
                              "evaluation seed above, instead of each seed loading its own checkpoint")
    sub.add_argument("--extractor_dir", type=str, default="outputs/qcl_family_quanforge",
                         help="where the already-trained DRNN checkpoints live "
                              "(run_quanforge_pipeline_mnist.py's --output_dir)")
    sub.add_argument("--output_dir", type=str, default="outputs/density_quanforge")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/density_<dataset>_normcls<norm_cls>_results.csv")
    sub.add_argument("--force", action="store_true", help="recompute even if already in --results_csv")
    sub.add_argument("--img_size", type=int, default=16)

    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")

    return parser


def main():
    args = build_parser().parse_args()
    dispatch = {
        "distance": run_distance,
        "density": run_density,
        "qae": run_qae,
        "qvae": run_qvae,
        "gan": run_gan,
        "ganomaly": run_ganomaly,
    }
    dispatch[args.detector_family](args)


if __name__ == "__main__":
    main()
