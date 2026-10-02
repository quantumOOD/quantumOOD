"""End-to-end training, tuning, and evaluation pipeline for the quantum feature-extractor family."""
import argparse
import glob
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main_experiments"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import datasets, transforms

from scoring_pipelines import load_ood_split_scaled  # noqa: E402
from qcl_family_quanforge_km_tuning import select_k, select_m
from qcl_family_quanforge_ood import (QSVDDDetector, QuantumFeatureExtractor,
                                       count_circuit_params, knn_scores, mean_scores,
                                       medoid_scores, train_cosine_pair, train_random,
                                       train_shared_ansatz, train_vicreg)

RESULTS_COLUMNS = ["circuit", "input_resolution", "n_qubits", "feature_loss", "seed", "converged_epochs",
                    "converged", "best_k", "best_m", "svdd_final_loss",
                    "detector", "hyperparam", "auc_roc", "auc_pr"]


def ensure_results_csv_schema(results_csv):
    """Upgrade old result CSVs for input-resolution and multi-seed runs.

    Before 16x16 DRNN support, every DRNN row necessarily came from the 8x8
    path, while QCL/QCNN/HCQC rows used 16x16. Before multi-seed support,
    --seed defaulted to 0, so legacy rows are tagged as seed 0.
    """
    if not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0:
        return
    df = pd.read_csv(results_csv)
    changed = False
    if "input_resolution" not in df.columns:
        insert_at = 1 if "circuit" in df.columns else 0
        df.insert(insert_at, "input_resolution", "16x16")
        changed = True
    if "n_qubits" not in df.columns:
        insert_at = df.columns.get_loc("input_resolution") + 1 if "input_resolution" in df.columns else len(df.columns)
        df.insert(insert_at, "n_qubits", 8)
        changed = True
    if "seed" not in df.columns:
        insert_at = df.columns.get_loc("feature_loss") + 1 if "feature_loss" in df.columns else len(df.columns)
        df.insert(insert_at, "seed", 0)
        changed = True
    if changed:
        df.to_csv(results_csv, index=False)
        print(f"Upgraded existing results CSV with input_resolution/seed metadata: '{results_csv}'")


def set_global_seed(seed):
    """Seed Python, NumPy, and Torch so each repeated run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_tag_base(dataset, norm_cls, n_train, train_data_scale, circuit, readout, feature_loss,
                        lr, batch_size, hcqc_unitary, drnn_ent_train, drnn_scaling, seed,
                        vicreg_lambda_inv, vicreg_lambda_var, vicreg_lambda_cov, vicreg_gamma):
    """Build a qcl_family_quanforge-style run tag with seed isolation.

    The base naming follows qcl_family_quanforge_ood.py, with an additional
    _seedN suffix so repeated runs cannot accidentally share checkpoints.
    """
    tag = (f"{dataset}_{norm_cls}_{n_train}_scale{train_data_scale}_{circuit}"
           f"_{readout}_{feature_loss}_lr{lr}_bs{batch_size}")
    if circuit == "HCQC":
        tag += f"_{hcqc_unitary}"
    if circuit == "DRNN":
        tag += f"_ent{drnn_ent_train}_scl{drnn_scaling}"
    if feature_loss == "vicreg":
        tag += f"_li{vicreg_lambda_inv}_lv{vicreg_lambda_var}_lc{vicreg_lambda_cov}_g{vicreg_gamma}"
    # Multi-seed runs must never reuse another seed's extractor/QSVDD cache.
    tag += f"_seed{seed}"
    return tag


def checkpoint_path_for(output_dir, run_tag_base, step):
    return os.path.join(output_dir, f"qcl_family_quanforge_checkpoint_{run_tag_base}_epoch{step}.pt")


def find_resume_checkpoint(output_dir, run_tag_base):
    candidates = glob.glob(os.path.join(output_dir, f"qcl_family_quanforge_checkpoint_{run_tag_base}_epoch*.pt"))
    if not candidates:
        return None

    def extract_epoch(path):
        m = re.search(r'_epoch(\d+)\.pt$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(candidates, key=extract_epoch)


def save_checkpoint(extractor, loss_history, path, converged, circuit, readout, feature_loss,
                     hcqc_unitary, drnn_ent_train, drnn_scaling, seed):
    torch.save({
        "model_state_dict": extractor.state_dict(),
        "loss_history": np.array(loss_history),
        "circuit": circuit,
        "readout": readout,
        "hcqc_unitary": hcqc_unitary,
        "drnn_ent_train": drnn_ent_train,
        "drnn_scaling": drnn_scaling,
        "feature_loss": feature_loss,
        "seed": seed,
        "converged": converged,
    }, path)


def train_until_converged(extractor, train_pool_imgs, img_shape, feature_loss, lr, batch_size,
                           epochs_per_chunk, max_epochs, convergence_tol, convergence_patience,
                           aug_kwargs, vicreg_kwargs, loss_history, output_dir, run_tag_base,
                           circuit, readout, hcqc_unitary, drnn_ent_train, drnn_scaling, seed):
    """Returns (loss_history, current_epochs, converged: bool)."""
    if feature_loss == "random":
        train_random(extractor)
        current_epochs = len(loss_history)  # stays 0 unless resumed from a prior (pointless) checkpoint
        path = checkpoint_path_for(output_dir, run_tag_base, current_epochs)
        save_checkpoint(extractor, loss_history, path, True, circuit, readout, feature_loss,
                         hcqc_unitary, drnn_ent_train, drnn_scaling, seed)
        print(f"[{circuit}/{feature_loss}] no training needed -- checkpoint saved to '{path}'")
        return loss_history, current_epochs, True

    stable_chunks = 0
    prev_chunk_mean = None
    # if resuming mid-convergence, re-derive prev_chunk_mean from the
    # tail of loss_history so the patience counter doesn't reset to 0
    # on a resumed run
    if len(loss_history) >= epochs_per_chunk:
        prev_chunk_mean = float(np.mean(loss_history[-epochs_per_chunk:]))

    while True:
        current_epochs = len(loss_history)
        if current_epochs >= max_epochs:
            print(f"[{circuit}/{feature_loss}] reached max_epochs={max_epochs} without confirmed "
                  f"convergence -- stopping anyway")
            path = checkpoint_path_for(output_dir, run_tag_base, current_epochs)
            save_checkpoint(extractor, loss_history, path, False, circuit, readout, feature_loss,
                             hcqc_unitary, drnn_ent_train, drnn_scaling, seed)
            return loss_history, current_epochs, False

        chunk_epochs = min(epochs_per_chunk, max_epochs - current_epochs)
        # Use a deterministic but different augmentation RNG for each chunk.
        # current_epochs makes resumed runs continue from the same chunk seed.
        chunk_seed = seed + current_epochs
        if feature_loss == "compact":
            new_losses = train_shared_ansatz(extractor, train_pool_imgs, chunk_epochs, lr, batch_size)
        elif feature_loss == "cosine":
            new_losses = train_cosine_pair(extractor, train_pool_imgs, img_shape, chunk_epochs, lr,
                                            batch_size, aug_kwargs, seed=chunk_seed)
        elif feature_loss == "vicreg":
            new_losses = train_vicreg(extractor, train_pool_imgs, img_shape, chunk_epochs, lr, batch_size,
                                       aug_kwargs, seed=chunk_seed, **vicreg_kwargs)
        else:
            raise ValueError(f"unknown feature_loss {feature_loss}")

        loss_history.extend(new_losses)
        current_epochs = len(loss_history)

        converged = False
        chunk_mean = float(np.mean(new_losses)) if new_losses else None
        if prev_chunk_mean is not None and chunk_mean is not None:
            rel_change = abs(prev_chunk_mean - chunk_mean) / (abs(prev_chunk_mean) + 1e-12)
            stable_chunks = stable_chunks + 1 if rel_change < convergence_tol else 0
            print(f"[{circuit}/{feature_loss}] epoch {current_epochs}: chunk_mean_loss={chunk_mean:.6f} "
                  f"(prev={prev_chunk_mean:.6f}, rel_change={rel_change:.6f}), "
                  f"stable_chunks={stable_chunks}/{convergence_patience}")
            if stable_chunks >= convergence_patience:
                print(f"[{circuit}/{feature_loss}] CONVERGED after {current_epochs} epochs")
                converged = True
        prev_chunk_mean = chunk_mean

        # checkpoint every chunk (== every epochs_per_chunk epochs)
        path = checkpoint_path_for(output_dir, run_tag_base, current_epochs)
        save_checkpoint(extractor, loss_history, path, converged, circuit, readout, feature_loss,
                         hcqc_unitary, drnn_ent_train, drnn_scaling, seed)
        print(f"[{circuit}/{feature_loss}] checkpoint saved to '{path}' (converged={converged})")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(loss_history) + 1), loss_history)
        ax.set_title(f"{circuit} (QuanForge) {feature_loss} loss against epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"qcl_family_quanforge_loss_{run_tag_base}_epoch{current_epochs}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        if converged:
            return loss_history, current_epochs, True


def run_one_combo(circuit, feature_loss, args, n_train, train_pool_imgs, val_imgs, val_labels,
                   test_imgs, test_labels, img_shape, results_csv, detectors):
    run_tag_base = build_run_tag_base(
        args.dataset, args.norm_cls, n_train, args.train_data_scale, circuit, args.readout, feature_loss,
        args.lr, args.batch_size, args.hcqc_unitary, args.drnn_ent_train, args.drnn_scaling, args.seed,
        args.vicreg_lambda_inv, args.vicreg_lambda_var, args.vicreg_lambda_cov, args.vicreg_gamma
    )

    extractor = QuantumFeatureExtractor(circuit, args.readout, args.hcqc_unitary,
                                         args.drnn_ent_train, args.drnn_scaling)
    print(f"\n{'=' * 80}\n{circuit}/{feature_loss}: "
          f"{count_circuit_params(circuit, args.hcqc_unitary, args.drnn_ent_train)} "
          f"trainable params\n{'=' * 80}")

    if args.force:
        stale = find_resume_checkpoint(args.output_dir, run_tag_base)
        if stale:
            print(f"[{circuit}/{feature_loss}] --force given -- ignoring existing checkpoint '{stale}', "
                  f"training from scratch")
    resume_path = None if args.force else find_resume_checkpoint(args.output_dir, run_tag_base)
    loss_history = []
    already_converged = False
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        extractor.load_state_dict(ckpt["model_state_dict"])
        loss_history = list(ckpt["loss_history"])
        already_converged = bool(ckpt.get("converged", False))
        print(f"[{circuit}/{feature_loss}] resumed from '{resume_path}' at epoch {len(loss_history)} "
              f"(converged={already_converged})")

    aug_kwargs = dict(max_rotate=args.aug_max_rotate, max_translate=args.aug_max_translate,
                       noise_std=args.aug_noise_std)
    vicreg_kwargs = dict(lambda_inv=args.vicreg_lambda_inv, lambda_var=args.vicreg_lambda_var,
                          lambda_cov=args.vicreg_lambda_cov, gamma=args.vicreg_gamma)

    if already_converged:
        current_epochs = len(loss_history)
    else:
        start = time.time()
        loss_history, current_epochs, converged = train_until_converged(
            extractor, train_pool_imgs, img_shape, feature_loss, args.lr, args.batch_size,
            args.epochs_per_chunk, args.max_epochs, args.convergence_tol, args.convergence_patience,
            aug_kwargs, vicreg_kwargs, loss_history, args.output_dir, run_tag_base,
            circuit, args.readout, args.hcqc_unitary, args.drnn_ent_train, args.drnn_scaling, args.seed
        )
        print(f"[{circuit}/{feature_loss}] training finished in {time.time() - start:0.1f}s "
              f"({current_epochs} total epochs, converged={converged})")
        already_converged = converged

    # -------------------- K/M tuning on val (only for the detectors requested) --------------------
    extractor.eval()
    with torch.no_grad():
        train_embs = extractor.get_embedding(torch.as_tensor(train_pool_imgs, dtype=torch.float64)).numpy()
        test_embs = extractor.get_embedding(torch.as_tensor(test_imgs, dtype=torch.float64)).numpy()

    best_k = best_m = None
    if "QKNN" in detectors:
        with torch.no_grad():
            val_embs = extractor.get_embedding(torch.as_tensor(val_imgs, dtype=torch.float64)).numpy()
        best_k, k_records = select_k(train_embs, val_embs, val_labels, args.k_candidates)
    if "QMedoids" in detectors:
        with torch.no_grad():
            val_embs = extractor.get_embedding(torch.as_tensor(val_imgs, dtype=torch.float64)).numpy()
        best_m, m_records = select_m(train_embs, val_embs, val_labels, args.m_candidates, seed=args.seed)

    scored = {}
    svdd_final_loss = None
    if "QKNN" in detectors:
        scored["QKNN"] = (knn_scores(train_embs, test_embs, best_k), f"K={best_k}")
    if "QMean" in detectors:
        scored["QMean"] = (mean_scores(train_embs, test_embs), "")
    if "QMedoids" in detectors:
        scored["QMedoids"] = (medoid_scores(train_embs, test_embs, best_m, seed=args.seed), f"M={best_m}")
    if "QSVDD" in detectors:
        # -------------------- QSVDD fine-tune (cached) --------------------
        svdd_checkpoint_path = os.path.join(
            args.output_dir,
            f"qcl_family_quanforge_qsvdd_checkpoint_{run_tag_base}_epoch{current_epochs}"
            f"_svdd{args.svdd_epochs}ep_lr{args.svdd_lr}_lam{args.svdd_lambda}.pt"
        )
        svdd = QSVDDDetector(extractor)
        svdd.initialize_center(train_pool_imgs)
        svdd.train_svdd(train_pool_imgs, args.svdd_epochs, args.svdd_lr, args.batch_size, args.svdd_lambda,
                         save_path=svdd_checkpoint_path, force_retrain=args.force)
        svdd_final_loss = svdd.final_loss
        scored["QSVDD"] = (svdd.predict_score(test_imgs), f"lambda={args.svdd_lambda}")

    rows = []
    input_resolution = f"{img_shape[0]}x{img_shape[1]}"
    for name, (scores, hp) in scored.items():
        auc_roc = roc_auc_score(test_labels, scores)
        auc_pr = average_precision_score(test_labels, scores)
        print(f"[{circuit}/{feature_loss}] {name:<10} {hp:<12} AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
        rows.append({
            "circuit": circuit, "input_resolution": input_resolution, "n_qubits": extractor.n_qubits,
            "feature_loss": feature_loss, "seed": args.seed, "converged_epochs": current_epochs,
            "converged": already_converged, "best_k": best_k, "best_m": best_m,
            "svdd_final_loss": svdd_final_loss, "detector": name, "hyperparam": hp,
            "auc_roc": auc_roc, "auc_pr": auc_pr,
        })

    df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    ensure_results_csv_schema(results_csv)
    df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
    # de-duplicate on (circuit, input_resolution, n_qubits, feature_loss, seed, detector), keeping
    # the LAST (most recent) row -- so a --force re-run replaces stale results instead of leaving
    # old and new rows both in the CSV.
    full_df = pd.read_csv(results_csv)
    if "input_resolution" not in full_df.columns:
        full_df["input_resolution"] = "16x16"
    if "n_qubits" not in full_df.columns:
        full_df["n_qubits"] = 8
    if "seed" not in full_df.columns:
        full_df["seed"] = 0
    full_df = full_df.drop_duplicates(
        subset=["circuit", "input_resolution", "n_qubits", "feature_loss", "seed", "detector"], keep="last"
    )
    full_df.to_csv(results_csv, index=False)
    print(f"[{circuit}/{feature_loss}] results appended to '{results_csv}'")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--norm_cls", type=int, default=0)
    parser.add_argument("--ood_cls", type=int, default=1)
    parser.add_argument("--setting", type=int, default=1, choices=[1, 2])
    parser.add_argument("--train_data_scale", type=float, default=0.2,
                         help="data_scale for the training pool AND the K/M-tuning val split")
    parser.add_argument("--test_data_scale", type=float, default=1.0,
                         help="data_scale for the FINAL reported test set only (separate call to "
                              "load_ood_split_scaled, same --seed so val stays identical/disjoint "
                              "between the two calls)")
    parser.add_argument("--n_val_per_cls", type=int, default=20)
    parser.add_argument("--num_latent", type=int, default=6)
    parser.add_argument("--num_trash", type=int, default=2)
    parser.add_argument("--circuits", nargs="+", type=str, default=["QCNN", "DRNN", "HCQC"],
                         choices=["QCL", "QCNN", "HCQC", "DRNN"])
    parser.add_argument("--feature_losses", nargs="+", type=str, default=["random", "cosine", "vicreg", "compact"],
                         choices=["random", "cosine", "vicreg", "compact"])
    parser.add_argument("--readout", type=str, default="expval", choices=["probs", "expval"],
                         help="fixed to expval per this pipeline's spec; probs is still accepted if you "
                              "want to reuse this script for a probs sweep instead")
    parser.add_argument("--hcqc_unitary", type=str, default="U_SU4",
                         choices=["U_TTN", "U_5", "U_6", "U_9", "U_13", "U_14", "U_15", "U_SO4", "U_SU4"])
    parser.add_argument("--drnn_ent_train", action="store_true")
    parser.add_argument("--drnn_scaling", type=float, default=1.5)
    parser.add_argument("--epochs_per_chunk", type=int, default=5,
                         help="checkpoint-save cadence AND the convergence-check granularity")
    parser.add_argument("--max_epochs", type=int, default=30, help="hard cap regardless of convergence")
    parser.add_argument("--convergence_tol", type=float, default=1e-3,
                         help="relative chunk-mean-loss change below which a chunk counts as 'stable'")
    parser.add_argument("--convergence_patience", type=int, default=2,
                         help="consecutive stable chunks required to declare convergence")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--aug_max_rotate", type=float, default=15.0)
    parser.add_argument("--aug_max_translate", type=int, default=2)
    parser.add_argument("--aug_noise_std", type=float, default=0.03)
    parser.add_argument("--vicreg_lambda_inv", type=float, default=25.0)
    parser.add_argument("--vicreg_lambda_var", type=float, default=25.0)
    parser.add_argument("--vicreg_lambda_cov", type=float, default=1.0)
    parser.add_argument("--vicreg_gamma", type=float, default=1.0,
                         help="expval-appropriate default (dims in [-1,1]) -- since --readout is fixed to "
                              "expval in this pipeline, the probs-specific lower gamma is not needed")
    parser.add_argument("--k_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    parser.add_argument("--m_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    parser.add_argument("--svdd_epochs", type=int, default=10)
    parser.add_argument("--svdd_lr", type=float, default=1e-3)
    parser.add_argument("--svdd_lambda", type=float, default=1e-3)
    parser.add_argument("--detectors", nargs="+", type=str, default=["QKNN", "QMean", "QMedoids", "QSVDD"],
                         choices=["QKNN", "QMean", "QMedoids", "QSVDD"],
                         help="which detector(s) to compute/evaluate per combo -- e.g. --detectors QSVDD "
                              "skips QKNN/QMedoids' K/M grid search on val entirely (a real time cost), "
                              "not just the final scoring")
    parser.add_argument("--force", action="store_true",
                         help="ignore existing results in --results_csv AND any existing checkpoint/QSVDD "
                              "cache for the selected --circuits/--feature_losses -- retrains and "
                              "re-evaluates from scratch instead of skipping/resuming. Use this whenever a "
                              "circuit's implementation changed and old cached results/checkpoints are "
                              "stale (narrow the blast radius with --circuits/--feature_losses)")
    parser.add_argument("--seed", type=int, default=0,
                         help="single seed, or starting seed when --num_seeds > 1")
    parser.add_argument("--num_seeds", type=int, default=1,
                         help="number of consecutive seeds to run starting from --seed; e.g. --num_seeds 5 runs 0..4 by default")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="explicit seed list, e.g. --seeds 0 1 2 3 4; overrides --seed/--num_seeds")
    parser.add_argument("--output_dir", type=str, default="outputs/qcl_family_quanforge")
    parser.add_argument("--results_csv", type=str, default="outputs/qcl_family_quanforge/full_pipeline_mnist_results.csv")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)

    if args.seeds is not None:
        run_seeds = list(dict.fromkeys(args.seeds))
    else:
        if args.num_seeds < 1:
            parser.error("--num_seeds must be >= 1")
        run_seeds = list(range(args.seed, args.seed + args.num_seeds))
    if not run_seeds:
        parser.error("at least one seed is required")
    print(f"Seeds to run: {run_seeds}")

    for c in args.circuits:
        assert args.num_latent + args.num_trash == 8, (
            f"16x16 input requires 2**8=256 features; for {c}, "
            f"--num_latent+--num_trash must sum to 8"
        )

    if args.dataset == "mnist":
        _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    else:
        _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
    n_train = int(args.train_data_scale * _class_count)
    print(f"train_data_scale={args.train_data_scale} -> n_train={n_train} "
          f"(of {_class_count} available {args.norm_cls}-class training images)")

    def _load_data(num_latent, num_trash, seed):
        """train_pool/val at train_data_scale; test at test_data_scale.

        Within a seed, both calls use the same seed so the validation split is
        identical/disjoint while only the final test size changes.
        """
        train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, img_shape, feat_dim, ood_str = \
            load_ood_split_scaled(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                   n_train, args.n_val_per_cls, args.train_data_scale, num_latent,
                                   num_trash, seed=seed)
        _unused_train_pool, _unused_val_imgs, _unused_val_labels, test_imgs, test_labels, _, _, _ = \
            load_ood_split_scaled(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                   n_train, args.n_val_per_cls, args.test_data_scale, num_latent,
                                   num_trash, seed=seed)
        return train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, img_shape

    combos = [(c, f) for c in args.circuits for f in args.feature_losses]
    total_runs = len(run_seeds) * len(combos)
    print(f"\nPipeline: {total_runs} seed/combo runs = {len(run_seeds)} seeds x {len(combos)} combos "
          f"({args.circuits} x {args.feature_losses}), "
          f"train_data_scale={args.train_data_scale}, test_data_scale={args.test_data_scale}\n")

    # keyed by (circuit, input_resolution, n_qubits, feature_loss, seed, detector).
    # This lets different seeds and multiple DRNN resolution/qubit variants coexist in one CSV.
    already_done = set()
    if args.force:
        print("--force given: ignoring existing results/checkpoints for the selected "
              f"--circuits={args.circuits} --feature_losses={args.feature_losses} "
              f"seeds={run_seeds}")
    elif os.path.exists(args.results_csv):
        ensure_results_csv_schema(args.results_csv)
        _prev = pd.read_csv(args.results_csv)
        already_done = set(zip(_prev["circuit"], _prev["input_resolution"], _prev["n_qubits"],
                               _prev["feature_loss"], _prev["seed"], _prev["detector"]))

    pipeline_start = time.time()
    run_index = 0
    for seed in run_seeds:
        args.seed = seed
        set_global_seed(seed)
        print(f"\n{'#' * 100}\nSEED {seed}\n{'#' * 100}")

        # Data must be rebuilt for every seed so the train/val/test sampling is
        # actually repeated rather than only reinitializing the circuit. DRNN
        # uses the SAME 16x16=256-pixel input as every other circuit.
        data_by_family = {"standard": _load_data(args.num_latent, args.num_trash, seed)}
        data_by_family["drnn"] = data_by_family["standard"]

        for circuit, feature_loss in combos:
            run_index += 1
            expected_resolution = "16x16"
            expected_n_qubits = 8
            missing_detectors = [
                d for d in args.detectors
                if (circuit, expected_resolution, expected_n_qubits, feature_loss, seed, d) not in already_done
            ]
            if not missing_detectors:
                print(f"[{run_index}/{total_runs}] seed={seed} {circuit}/{feature_loss}: all requested detectors "
                      f"{args.detectors} already in '{args.results_csv}', skipping")
                continue
            if len(missing_detectors) < len(args.detectors):
                print(f"[{run_index}/{total_runs}] seed={seed} {circuit}/{feature_loss}: "
                      f"{set(args.detectors) - set(missing_detectors)} already done, "
                      f"computing only {missing_detectors}")

            # Reset model/training RNG before every combo. For a fixed seed this
            # gives a controlled comparison across training objectives.
            set_global_seed(seed)
            print(f"\n[{run_index}/{total_runs}] STARTING seed={seed} {circuit}/{feature_loss} "
                  f"(elapsed so far: {time.time() - pipeline_start:0.1f}s)")
            train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, img_shape = \
                data_by_family["drnn" if circuit == "DRNN" else "standard"]
            run_one_combo(circuit, feature_loss, args, n_train, train_pool_imgs, val_imgs, val_labels,
                          test_imgs, test_labels, img_shape, args.results_csv, missing_detectors)

    print(f"\nPipeline finished in {time.time() - pipeline_start:0.1f}s total")
    ensure_results_csv_schema(args.results_csv)
    final_df = pd.read_csv(args.results_csv)
    print(f"\n{'=' * 100}\nFINAL PER-SEED RESULTS ({args.results_csv})\n{'=' * 100}")
    print(final_df.to_string(index=False))

    # Also write a compact mean/std table across the seeds requested in this run.
    selected = final_df[
        final_df["seed"].isin(run_seeds)
        & final_df["circuit"].isin(args.circuits)
        & final_df["feature_loss"].isin(args.feature_losses)
    ].copy()
    # If DRNN is included, keep only the 8-qubit/16x16 variant.
    if "DRNN" in args.circuits:
        selected = selected[(selected["circuit"] != "DRNN")
                            | ((selected["input_resolution"] == "16x16") & (selected["n_qubits"] == 8))]
    if not selected.empty:
        group_cols = ["circuit", "input_resolution", "n_qubits", "feature_loss", "detector"]
        summary = (selected.groupby(group_cols, as_index=False)
                   .agg(n_seeds=("seed", "nunique"),
                        auc_roc_mean=("auc_roc", "mean"),
                        auc_roc_std=("auc_roc", "std"),
                        auc_pr_mean=("auc_pr", "mean"),
                        auc_pr_std=("auc_pr", "std")))
        summary_path = os.path.splitext(args.results_csv)[0] + "_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n{'=' * 100}\nMEAN +/- STD ACROSS SEEDS {run_seeds}\n{'=' * 100}")
        print(summary.to_string(index=False))
        print(f"\nSeed summary saved to '{summary_path}'")
