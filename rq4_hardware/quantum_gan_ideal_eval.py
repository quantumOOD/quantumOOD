"""Quantum GAN-based detectors scored under ideal simulation at a reduced test scale."""
import argparse
import glob
import hashlib
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main_experiments"))
from quantum_OOD_detectors import (ClassicalCritic, QuantumDiscriminator, QuantumGenerator,  # noqa: E402
                                    compute_anomaly_scores, load_ood_test, load_real_dataset)

RESULTS_COLUMNS = ["dataset", "norm_cls", "test_scale", "seed", "detector", "hyperparam", "auc_roc", "auc_pr",
                    "n_id", "n_ood", "circuit_execution_seconds", "classical_seconds",
                    "checkpoint_path", "checkpoint_sha256"]
GAN_NAME_TO_DETECTOR = {"q_anogan": "Q-AnoGAN", "qwgan_gp": "QWGAN-GP"}


def find_gan_checkpoint(checkpoint_dir, norm_cls, gan_name, seed):
    pattern = os.path.join(checkpoint_dir, f"*_id{norm_cls}_ood*_{gan_name}_seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 run dir matching '{pattern}', found {len(candidates)}: {candidates}. "
            f"Train it first via the gan subcommand of quantum_OOD_detectors.py."
        )
    model_path = os.path.join(candidates[0], f"{gan_name}_model.pt")
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
                         help="directory holding the gan subcommand's per-run dirs for this dataset, e.g. "
                              "outputs/results_nq8/Gan-based/results/qanogan_nq8/mnist")

    parser.add_argument("--test_scale", type=float, default=0.1)
    parser.add_argument("--setting", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ood_class", type=int, default=1, help="only used if --setting 2")
    parser.add_argument("--ood_seed", type=int, default=42)

    parser.add_argument("--gan_types", nargs="+", type=str, default=["q_anogan", "qwgan_gp"],
                         choices=["q_anogan", "qwgan_gp"])
    parser.add_argument("--batch_size", type=int, default=64, help="only affects scoring speed, not results -- "
                                                                    "each test sample's z is optimized independently")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                         help="each seed loads ITS OWN already-trained G/D checkpoint (never retrained)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="outputs/ideal_sim_eval")
    parser.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/quantum_gan_ideal_<dataset>_normcls<norm_cls>_"
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
        args.output_dir,
        f"quantum_gan_ideal_{args.dataset}_normcls{args.norm_cls}_scale{args.test_scale}_results.csv")
    os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

    img_shape = (args.img_size, args.img_size)
    _train_np, _val_np, test_id_np = load_real_dataset(
        args.dataset, args.data_dir, args.norm_cls, 0.2, args.test_scale, img_shape)
    test_ood_np = load_ood_test(args.dataset, args.data_dir, args.norm_cls, args.test_scale, len(test_id_np),
                                 args.setting, args.ood_class, img_shape, seed=args.ood_seed)
    test_x = torch.cat([torch.tensor(test_id_np, dtype=torch.float32),
                         torch.tensor(test_ood_np, dtype=torch.float32)], dim=0)
    test_labels = torch.cat([torch.zeros(len(test_id_np), dtype=torch.long),
                              torch.ones(len(test_ood_np), dtype=torch.long)])
    test_loader = DataLoader(TensorDataset(test_x, test_labels), batch_size=args.batch_size, shuffle=False)
    print(f"test_scale={args.test_scale}: n_id={len(test_id_np)} n_ood={len(test_ood_np)}")
    feature_dim = args.img_size * args.img_size

    all_rows = []
    for gan_name in args.gan_types:
        detector = GAN_NAME_TO_DETECTOR[gan_name]
        for seed in args.seeds:
            checkpoint_path = find_gan_checkpoint(args.checkpoint_dir, args.norm_cls, gan_name, seed)
            print(f"Loading FIXED {detector} checkpoint (seed={seed}) from '{checkpoint_path}' -- never retrained")
            ckpt_sha256 = sha256_of(checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_args = ckpt["args"]

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
            # Timing is not split into circuit/classical -- the full
            # find_optimal_z time is attributed to circuit_execution_seconds (G/D calls dominate).
            scores, circuit_seconds = compute_anomaly_scores(G, D, test_loader, score_args, device)
            classical_seconds = 0.0
            auc_roc = roc_auc_score(test_labels.numpy(), scores)
            auc_pr = average_precision_score(test_labels.numpy(), scores)
            print(f"{detector} (seed={seed})  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f} "
                  f"circuit(approx)={circuit_seconds:.1f}s")
            all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "test_scale": args.test_scale,
                             "seed": seed, "detector": detector, "hyperparam": f"z_iter={ckpt_args['z_iter']}",
                             "auc_roc": auc_roc, "auc_pr": auc_pr, "n_id": len(test_id_np),
                             "n_ood": len(test_ood_np), "circuit_execution_seconds": circuit_seconds,
                             "classical_seconds": classical_seconds,
                             "checkpoint_path": checkpoint_path, "checkpoint_sha256": ckpt_sha256})

    df = pd.DataFrame(all_rows, columns=RESULTS_COLUMNS)
    df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
    full_df = pd.read_csv(results_csv)
    full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "test_scale", "seed", "detector"], keep="last")
    full_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to '{results_csv}'")
