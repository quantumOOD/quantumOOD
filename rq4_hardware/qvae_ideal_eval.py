"""Quantum VAE reconstruction detector scored under ideal simulation at a reduced test scale."""
import argparse
import glob
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main_experiments"))
from quantum_OOD_detectors import QuantumVAE, load_ood_test, load_real_dataset  # noqa: E402

RESULTS_COLUMNS = ["dataset", "norm_cls", "test_scale", "seed", "detector", "hyperparam", "auc_roc", "auc_pr",
                    "n_id", "n_ood", "circuit_execution_seconds", "classical_seconds",
                    "checkpoint_path", "checkpoint_sha256"]


def timed_reconstruction_scores(model, x_np, device, batch_size=128):
    """Reproduces QuantumVAE.forward() at eval time EXACTLY: model.eval()
    makes forward() use z=mu (not the stochastic reparameterize draw, see
    the qvae subcommand's forward()/encode()/decode()), so only the mu branch
    (enc_vqc_mu) is exercised.

    Returns (scores, circuit_seconds, classical_seconds)."""
    model.eval()
    out, circuit_s, classical_s = [], 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.tensor(x_np[i:i + batch_size], dtype=torch.float32).to(device)

            t0 = time.perf_counter()
            compressed = model.enc_compress(xb)
            angles = torch.arccos(torch.clamp(compressed, -1.0, 1.0))
            classical_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            enc_out = model.enc_vqc_mu(angles)
            circuit_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            mu = enc_out[:, :model.latent_dim]
            expanded = model.dec_expand(mu)
            angles2 = torch.arccos(torch.clamp(expanded, -1.0, 1.0))
            classical_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            q_out = model.dec_vqc(angles2)
            circuit_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            recon = F.normalize(model.dec_upscale(q_out), p=2, dim=1)
            out.append(((xb - recon) ** 2).mean(dim=1).cpu().numpy())
            classical_s += time.perf_counter() - t0
    return np.concatenate(out), circuit_s, classical_s


def find_qvae_checkpoint(checkpoint_dir, norm_cls, seed):
    pattern = os.path.join(checkpoint_dir, f"*_id{norm_cls}_nq*_seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 run dir matching '{pattern}', found {len(candidates)}: {candidates}. "
            f"Train it first via the qvae subcommand of quantum_OOD_detectors.py."
        )
    model_path = os.path.join(candidates[0], "qvae_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Run dir '{candidates[0]}' found but '{model_path}' is missing.")
    return model_path


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--norm_cls", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=16)
    parser.add_argument("--data_dir", type=str, default="./data")

    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help="directory holding the qvae subcommand's per-run dirs for this dataset, e.g. "
                              "outputs/results_nq8/QVAE/results/mnist")

    parser.add_argument("--test_scale", type=float, default=0.1)
    parser.add_argument("--setting", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ood_class", type=int, default=1, help="only used if --setting 2")
    parser.add_argument("--ood_seed", type=int, default=42)

    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                         help="each seed loads ITS OWN already-trained QVAE checkpoint (never retrained)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="outputs/ideal_sim_eval")
    parser.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/qvae_ideal_<dataset>_normcls<norm_cls>_"
                              "scale<test_scale>_results.csv")
    args = parser.parse_args()

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
    results_csv = args.results_csv or os.path.join(
        args.output_dir, f"qvae_ideal_{args.dataset}_normcls{args.norm_cls}_scale{args.test_scale}_results.csv")
    os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

    img_shape = (args.img_size, args.img_size)
    _train_np, _val_np, test_id_np = load_real_dataset(
        args.dataset, args.data_dir, args.norm_cls, 0.2, args.test_scale, img_shape)
    test_ood_np = load_ood_test(args.dataset, args.data_dir, args.norm_cls, args.test_scale, len(test_id_np),
                                 args.setting, args.ood_class, img_shape, seed=args.ood_seed)
    test_x = torch.cat([torch.tensor(test_id_np, dtype=torch.float32),
                         torch.tensor(test_ood_np, dtype=torch.float32)], dim=0).numpy()
    test_labels = ([0] * len(test_id_np)) + ([1] * len(test_ood_np))
    print(f"test_scale={args.test_scale}: n_id={len(test_id_np)} n_ood={len(test_ood_np)}")
    feature_dim = args.img_size * args.img_size

    all_rows = []
    for seed in args.seeds:
        checkpoint_path = find_qvae_checkpoint(args.checkpoint_dir, args.norm_cls, seed)
        print(f"Loading FIXED QVAE checkpoint (seed={seed}) from '{checkpoint_path}' -- never retrained")
        ckpt_sha256 = sha256_of(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt["args"]

        model = QuantumVAE(ckpt_args["n_qubits"], ckpt_args["n_layers"], ckpt_args["latent_dim"],
                            feature_dim, ckpt_args.get("rot_gate", "RX+RZ"),
                            ckpt_args.get("entangle_gate", "CRY")).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        scores, circuit_seconds, classical_seconds = timed_reconstruction_scores(model, test_x, device)

        auc_roc = roc_auc_score(test_labels, scores)
        auc_pr = average_precision_score(test_labels, scores)
        print(f"QVAE-Recon (seed={seed})  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f} "
              f"circuit={circuit_seconds:.4f}s classical={classical_seconds:.4f}s")
        all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "test_scale": args.test_scale,
                         "seed": seed, "detector": "QVAE-Recon", "hyperparam": "",
                         "auc_roc": auc_roc, "auc_pr": auc_pr, "n_id": len(test_id_np), "n_ood": len(test_ood_np),
                         "circuit_execution_seconds": circuit_seconds, "classical_seconds": classical_seconds,
                         "checkpoint_path": checkpoint_path, "checkpoint_sha256": ckpt_sha256})

    df = pd.DataFrame(all_rows, columns=RESULTS_COLUMNS)
    df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
    full_df = pd.read_csv(results_csv)
    full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "test_scale", "seed", "detector"], keep="last")
    full_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to '{results_csv}'")
