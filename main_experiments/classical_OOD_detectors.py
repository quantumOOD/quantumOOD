"""All classical OOD detectors: distance, density, reconstruction, and GAN-based families."""
import argparse
import copy
import glob
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as tv_datasets
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from scoring_pipelines import (Autoencoder, VAE, kmedoids_euclidean, knn_scores_euclidean,  # noqa: E402
                                load_ood_split_scaled_raw, mean_scores_euclidean, medoid_scores_euclidean,
                                select_k_euclidean, select_m_euclidean, vae_loss_per_sample,
                                BACKBONE_FEATURE_DIM, LeNet5FeatureExtractor, PretrainedDeepSVDDDetector,
                                PretrainedFeatureExtractor, ComposedEmbedding, LinearProjector, train_projector,
                                normalize_images, resize_and_flatten)
from build_adversarial_ood_split import ATTACK_ORDER, load_adversarial_test_set  # noqa: E402

ADV_RESULTS_COLUMNS = ["dataset", "norm_cls", "eval_setting", "seed", "detector", "hyperparam", "auc_roc", "auc_pr",
                        "n_id", "n_fgsm", "n_pgd", "n_spsa", "n_salt_pepper"]


# ============================================================================
# SAE: classical autoencoder-based distance detectors (DeepKNN, DeepMean, DeepMedoids, DeepSVDD)
# ============================================================================

class DeepSVDDDetector:
    """Starts from a deep-copy of the trained SAE encoder, then fine-tunes
    its OWN copy toward a one-class hypersphere objective -- mirrors
    QSVDDDetector, but copy.deepcopy is safe here (plain PyTorch CNN, no
    PennyLane device/QNode closure to worry about)."""

    def __init__(self, base_encoder):
        self.base_encoder = base_encoder
        self.center = None
        self.R = None
        self.final_loss = None
        self.encoder = None

    def initialize_center(self, train_pool_imgs, eps=0.1):
        self.device = next(self.base_encoder.parameters()).device
        self.base_encoder.eval()
        with torch.no_grad():
            embs = self.base_encoder(torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=self.device))
        c = embs.mean(dim=0)
        # standard DeepSVDD trick: nudge near-zero center dims away from 0,
        # so the hypersphere can't trivially collapse to a single point
        near_zero = c.abs() < eps
        c = torch.where(near_zero & (c >= 0), torch.full_like(c, eps), c)
        c = torch.where(near_zero & (c < 0), torch.full_like(c, -eps), c)
        self.center = c
        return c

    def train_svdd(self, train_pool_imgs, epochs, lr, batch_size, lambda_reg, save_path=None, force_retrain=False):
        assert self.center is not None, "call initialize_center() before train_svdd()"

        if save_path and os.path.exists(save_path) and not force_retrain:
            print(f"Loading cached fine-tuned DeepSVDD encoder from '{save_path}'")
            ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
            self.encoder = copy.deepcopy(self.base_encoder)
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            self.encoder.eval()
            self.R = ckpt["R"].to(self.device)
            self.final_loss = ckpt.get("final_loss")
            print(f"DeepSVDD training outcome (cached): final_loss={self.final_loss}, R={self.R.item():.4f}")
            return []

        self.encoder = copy.deepcopy(self.base_encoder)  # already on self.device, deepcopy preserves that
        R = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + [R], lr=lr)

        images_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(images_t), batch_size=batch_size, shuffle=True)

        self.encoder.train()
        loss_rec = []
        for ep in range(epochs):
            total_loss = 0.0
            for (x,) in tqdm(loader, desc=f"DeepSVDD fine-tune epoch {ep + 1}/{epochs}"):
                optimizer.zero_grad()
                emb = self.encoder(x)
                dist_sq = torch.sum((emb - self.center) ** 2, dim=-1)
                loss = torch.mean(dist_sq) + lambda_reg * R ** 2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            loss_rec.append(avg_loss)
            print(f"DeepSVDD fine-tune epoch {ep + 1}/{epochs}: loss={avg_loss:.6f}, R={R.item():.4f}")
        self.R = R.detach()
        self.final_loss = loss_rec[-1] if loss_rec else None

        if save_path:
            torch.save({"encoder_state_dict": self.encoder.state_dict(), "R": self.R,
                        "final_loss": self.final_loss}, save_path)
            print(f"DeepSVDD fine-tuned encoder saved to '{save_path}'")
        return loss_rec

    def predict_score(self, imgs):
        self.encoder.eval()
        with torch.no_grad():
            emb = self.encoder(torch.as_tensor(imgs, dtype=torch.float32, device=self.device))
            scores = torch.sum((emb - self.center) ** 2, dim=-1).cpu().numpy()
        return scores


def find_sae_checkpoint(checkpoint_dir, run_tag_base):
    candidates = glob.glob(os.path.join(checkpoint_dir, f"classical_sae_checkpoint_{run_tag_base}_epoch*.pt"))
    if not candidates:
        return None

    def extract_epoch(path):
        m = re.search(r'_epoch(\d+)\.pt$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(candidates, key=extract_epoch)



def _run_sae_natural(args):
        os.makedirs(args.output_dir, exist_ok=True)
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"classical_sae_{args.dataset}_normcls{args.norm_cls}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if args.dataset == "mnist":
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)
        print(f"train_data_scale={args.train_data_scale} -> n_train={n_train} "
              f"(of {_class_count} available {args.norm_cls}-class training images)")

        # train_pool/val at train_data_scale; test at test_data_scale -- same
        # seed so val is identical/disjoint between the two calls.
        train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, _unused_ood_str = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=args.seed)
        _unused_train_pool, _unused_val_imgs, _unused_val_labels, test_imgs, test_labels, ood_str = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.test_data_scale, args.img_size, seed=args.seed)

        run_tag_base = (f"{args.dataset}_{args.norm_cls}_{len(train_pool_imgs)}_scale{args.train_data_scale}"
                         f"_ld{args.latent_dim}_lr{args.lr}_bs{args.batch_size}")


        def checkpoint_path_for(step):
            if args.checkpoint:
                return args.checkpoint
            return os.path.join(args.output_dir, f"classical_sae_checkpoint_{run_tag_base}_epoch{step}.pt")


        def find_resume_checkpoint():
            if args.checkpoint:
                return args.checkpoint if os.path.exists(args.checkpoint) else None
            candidates = glob.glob(os.path.join(args.output_dir, f"classical_sae_checkpoint_{run_tag_base}_epoch*.pt"))
            if not candidates:
                return None

            def extract_epoch(path):
                m = re.search(r'_epoch(\d+)\.pt$', os.path.basename(path))
                return int(m.group(1)) if m else -1

            return max(candidates, key=extract_epoch)


        model = Autoencoder(args.latent_dim).to(device)

        resume_path = find_resume_checkpoint() if args.resume else None
        if resume_path:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            train_loss_history = list(ckpt["train_loss_history"])
            val_loss_history = list(ckpt["val_loss_history"])
            val_loss_epochs = list(ckpt["val_loss_epochs"])
            print(f"Resumed from '{resume_path}' at epoch {len(train_loss_history)} "
                  f"(prior val loss: {val_loss_history[-1]:0.6f})")
        else:
            if args.resume:
                print(f"--resume given but no checkpoint found matching "
                      f"'classical_sae_checkpoint_{run_tag_base}_epoch*.pt', starting from scratch")
            train_loss_history, val_loss_history, val_loss_epochs = [], [], []

        train_imgs_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
        val_imgs_t = torch.as_tensor(val_imgs, dtype=torch.float32, device=device)
        test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)


        def save_progress(step):
            checkpoint_path = checkpoint_path_for(step)
            torch.save({
                "model_state_dict": model.state_dict(),
                "train_loss_history": np.array(train_loss_history),
                "val_loss_history": np.array(val_loss_history),
                "val_loss_epochs": np.array(val_loss_epochs),
                "latent_dim": args.latent_dim,
            }, checkpoint_path)
            print(f"[epoch {step}] checkpoint saved to '{checkpoint_path}'")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="train")
            ax.plot(val_loss_epochs, val_loss_history, marker="o", label="validation")
            ax.set_title("SAE reconstruction MSE against epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE loss")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f"classical_sae_loss_{run_tag_base}_epoch{step}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)

            was_training = model.training
            model.eval()
            num_show = min(5, test_imgs_t.shape[0])
            fig, axes = plt.subplots(num_show, 2, figsize=(6, 3 * num_show))
            if num_show == 1:
                axes = axes.reshape(1, 2)
            with torch.no_grad():
                for i in range(num_show):
                    image = test_imgs_t[i:i + 1]
                    recon = model(image)
                    mse = torch.nn.functional.mse_loss(recon, image).item()
                    axes[i, 0].imshow(image[0, 0].cpu().numpy(), cmap="gray")
                    axes[i, 0].set_title("Input" if i > 0 else "Input Data")
                    axes[i, 1].imshow(recon[0, 0].cpu().numpy(), cmap="gray")
                    axes[i, 1].set_title(f"Output (MSE={mse:.4f})")
            fig.suptitle(f"{args.dataset} class {args.norm_cls} SAE reconstructions @ epoch {step}")
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f"classical_sae_reconstruct_{run_tag_base}_epoch{step}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            if was_training:
                model.train()


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loader = DataLoader(TensorDataset(train_imgs_t), batch_size=args.batch_size, shuffle=True)

        start = time.time()
        for ep in range(args.epochs):
            model.train()
            total_loss = 0.0
            for (x,) in tqdm(loader, desc=f"SAE epoch {ep + 1}/{args.epochs}"):
                optimizer.zero_grad()
                recon = model(x)
                loss = torch.nn.functional.mse_loss(recon, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(loader)
            train_loss_history.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                val_loss = torch.nn.functional.mse_loss(model(val_imgs_t), val_imgs_t).item()
            val_loss_history.append(val_loss)
            val_loss_epochs.append(len(train_loss_history))
            print(f"SAE epoch {ep + 1}/{args.epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

            current_epochs = len(train_loss_history)
            if current_epochs % args.save_interval == 0:
                save_progress(current_epochs)
        elapsed = time.time() - start
        print(f"Fit in {elapsed:0.2f} seconds ({args.epochs} epochs this run)")

        current_epochs = len(train_loss_history)
        save_progress(current_epochs)

        # -------------------- K/M tuning on val, then quick check on test --------------------
        model.eval()
        with torch.no_grad():
            train_embs = model.get_embedding(train_imgs_t).cpu().numpy()
            test_embs = model.get_embedding(test_imgs_t).cpu().numpy()

        best_k = best_m = None
        if "DeepKNN" in args.detectors:
            with torch.no_grad():
                val_embs = model.get_embedding(val_imgs_t).cpu().numpy()
            best_k, _ = select_k_euclidean(train_embs, val_embs, val_labels, args.k_candidates)
        if "Deep-Medoids" in args.detectors:
            with torch.no_grad():
                val_embs = model.get_embedding(val_imgs_t).cpu().numpy()
            best_m, _ = select_m_euclidean(train_embs, val_embs, val_labels, args.m_candidates, seed=args.seed)

        scored = {}
        if "SAE-Recon" in args.detectors:
            model.eval()
            with torch.no_grad():
                recon = model(test_imgs_t)
                recon_errors = torch.nn.functional.mse_loss(recon, test_imgs_t, reduction='none')
                recon_errors = recon_errors.reshape(recon_errors.size(0), -1).mean(dim=1).cpu().numpy()
            scored["SAE-Recon"] = (recon_errors, "")
        if "DeepKNN" in args.detectors:
            scored["DeepKNN"] = (knn_scores_euclidean(train_embs, test_embs, best_k), f"K={best_k}")
        if "Deep-Mean" in args.detectors:
            scored["Deep-Mean"] = (mean_scores_euclidean(train_embs, test_embs), "")
        if "Deep-Medoids" in args.detectors:
            scored["Deep-Medoids"] = (medoid_scores_euclidean(train_embs, test_embs, best_m, seed=args.seed), f"M={best_m}")
        if "DeepSVDD" in args.detectors:
            svdd_checkpoint_path = args.svdd_checkpoint or os.path.join(
                args.output_dir,
                f"classical_svdd_checkpoint_{run_tag_base}_epoch{current_epochs}"
                f"_svdd{args.svdd_epochs}ep_lr{args.svdd_lr}_lam{args.svdd_lambda}_seed{args.seed}.pt"
            )
            svdd = DeepSVDDDetector(model.encoder)
            svdd.initialize_center(train_pool_imgs)
            svdd_loss = svdd.train_svdd(train_pool_imgs, args.svdd_epochs, args.svdd_lr, args.batch_size,
                                         args.svdd_lambda, save_path=svdd_checkpoint_path,
                                         force_retrain=args.svdd_force_retrain)
            svdd_final_loss_str = f"{svdd.final_loss:.6f}" if svdd.final_loss is not None else "n/a"
            print(f"DeepSVDD fine-tune done: final_loss={svdd_final_loss_str}, R={svdd.R.item():.4f} "
                  f"({args.svdd_epochs} epochs, lr={args.svdd_lr}, lambda={args.svdd_lambda})")
            if svdd_loss:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(range(1, len(svdd_loss) + 1), svdd_loss)
                ax.set_title("DeepSVDD fine-tuning loss against epoch")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss (dist_sq + lambda * R^2)")
                ax.grid(True)
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, f"classical_svdd_loss_{run_tag_base}_svddepoch{len(svdd_loss)}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
            scored["DeepSVDD"] = (svdd.predict_score(test_imgs), f"lambda={args.svdd_lambda}")

        print(f"\n{'Detector':<14} {'Hyperparam':<14} {'AUC-ROC':<10} {'AUC-PR':<10}")
        rows = []
        for name, (scores, hp) in scored.items():
            auc_roc = roc_auc_score(test_labels, scores)
            auc_pr = average_precision_score(test_labels, scores)
            print(f"{name:<14} {hp:<14} {auc_roc:<10.4f} {auc_pr:<10.4f}")
            rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "detector": name, "hyperparam": hp,
                         "auc_roc": auc_roc, "auc_pr": auc_pr, "epochs": current_epochs, "seed": args.seed})

        RESULTS_COLUMNS = ["dataset", "norm_cls", "detector", "hyperparam", "auc_roc", "auc_pr", "epochs", "seed"]
        new_df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
        new_df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "detector", "seed"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def _run_sae_adversarial(args):

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
            args.output_dir, f"classical_sae_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        if args.dataset == "mnist":
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)

        # ---- locate the ALREADY-TRAINED SAE checkpoint (never retrained here) ----
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            run_tag_base = (f"{args.dataset}_{args.norm_cls}_{n_train}_scale{args.train_data_scale}"
                             f"_ld{args.latent_dim}_lr{args.lr}_bs{args.batch_size}")
            checkpoint_path = find_sae_checkpoint(args.checkpoint_dir, run_tag_base)
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No trained SAE checkpoint found matching run_tag_base='{run_tag_base}' under "
                f"'{args.checkpoint_dir}'. Train it first via the sae subcommand, or pass --checkpoint directly."
            )
        print(f"Loading FIXED SAE checkpoint from '{checkpoint_path}' -- never retrained")
        model = Autoencoder.from_checkpoint(checkpoint_path, args.latent_dim).to(device)
        model.eval()

        # ---- clean ID-only train_pool, IDENTICAL to the original (clean-OOD) run (unused beyond sizing above) ----
        _train_pool_imgs, _val_imgs, _val_labels, _unused_test_imgs, _unused_test_labels, _ = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=0)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                           args.attack)
        print(f"test={len(test_imgs)} (ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)
        with torch.no_grad():
            recon = model(test_imgs_t)
            recon_errors = torch.nn.functional.mse_loss(recon, test_imgs_t, reduction='none')
            recon_errors = recon_errors.reshape(recon_errors.size(0), -1).mean(dim=1).cpu().numpy()

        auc_roc = roc_auc_score(test_labels, recon_errors)
        auc_pr = average_precision_score(test_labels, recon_errors)
        print(f"SAE-Recon  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f} (deterministic -- identical across all seeds)")

        all_rows = [{"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                     "seed": seed, "detector": "SAE-Recon", "hyperparam": "",
                     "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols} for seed in args.seeds]

        df = pd.DataFrame(all_rows, columns=ADV_RESULTS_COLUMNS)
        df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "eval_setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def run_sae(args):
    if getattr(args, "adversarial_dir", None):
        _run_sae_adversarial(args)
    else:
        _run_sae_natural(args)


# ============================================================================
# VAE: classical VAE-based reconstruction OOD detector
# ============================================================================

def find_vae_checkpoint(checkpoint_dir, run_tag_base):
    candidates = glob.glob(os.path.join(checkpoint_dir, f"classical_vae_checkpoint_{run_tag_base}_epoch*.pt"))
    if not candidates:
        return None

    def extract_epoch(path):
        m = re.search(r'_epoch(\d+)\.pt$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(candidates, key=extract_epoch)


def read_original_beta(original_results_csv, dataset, norm_cls):
    df = pd.read_csv(original_results_csv)
    row = df[(df["dataset"] == dataset) & (df["norm_cls"] == norm_cls) & (df["detector"] == "VAE-Recon")]
    assert len(row) >= 1, f"no original VAE-Recon row found for dataset={dataset} norm_cls={norm_cls}"
    hp = row["hyperparam"].iloc[0]  # "beta=1.0" -- same beta used for every seed in the original run
    return float(str(hp).split("=")[1])



def _run_vae_natural(args):
        os.makedirs(args.output_dir, exist_ok=True)
        results_csv = args.results_csv or os.path.join(
            args.output_dir, f"classical_vae_{args.dataset}_normcls{args.norm_cls}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if args.dataset == "mnist":
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)
        print(f"train_data_scale={args.train_data_scale} -> n_train={n_train} "
              f"(of {_class_count} available {args.norm_cls}-class training images)")

        # train_pool/val at train_data_scale; test at test_data_scale -- same
        # seed so val is identical/disjoint between the two calls.
        train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, _unused_ood_str = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=args.seed)
        _unused_train_pool, _unused_val_imgs, _unused_val_labels, test_imgs, test_labels, ood_str = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.test_data_scale, args.img_size, seed=args.seed)

        run_tag_base = (f"{args.dataset}_{args.norm_cls}_{len(train_pool_imgs)}_scale{args.train_data_scale}"
                         f"_ld{args.latent_dim}_beta{args.beta}_lr{args.lr}_bs{args.batch_size}")


        def checkpoint_path_for(step):
            if args.checkpoint:
                return args.checkpoint
            return os.path.join(args.output_dir, f"classical_vae_checkpoint_{run_tag_base}_epoch{step}.pt")


        def find_resume_checkpoint():
            if args.checkpoint:
                return args.checkpoint if os.path.exists(args.checkpoint) else None
            candidates = glob.glob(os.path.join(args.output_dir, f"classical_vae_checkpoint_{run_tag_base}_epoch*.pt"))
            if not candidates:
                return None

            def extract_epoch(path):
                m = re.search(r'_epoch(\d+)\.pt$', os.path.basename(path))
                return int(m.group(1)) if m else -1

            return max(candidates, key=extract_epoch)


        model = VAE(args.latent_dim).to(device)

        resume_path = find_resume_checkpoint() if args.resume else None
        if resume_path:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            train_loss_history = list(ckpt["train_loss_history"])
            val_loss_history = list(ckpt["val_loss_history"])
            val_loss_epochs = list(ckpt["val_loss_epochs"])
            print(f"Resumed from '{resume_path}' at epoch {len(train_loss_history)} "
                  f"(prior val loss: {val_loss_history[-1]:0.6f})")
        else:
            if args.resume and args.epochs == 0:
                raise FileNotFoundError(
                    f"--resume --epochs 0 (inference-only) given but no checkpoint found matching "
                    f"'classical_vae_checkpoint_{run_tag_base}_epoch*.pt' under '{args.output_dir}' -- "
                    f"refusing to silently score an untrained model. Check --latent_dim/--beta/--lr/"
                    f"--batch_size/--train_data_scale match the checkpoint you meant to load."
                )
            if args.resume:
                print(f"--resume given but no checkpoint found matching "
                      f"'classical_vae_checkpoint_{run_tag_base}_epoch*.pt', starting from scratch")
            train_loss_history, val_loss_history, val_loss_epochs = [], [], []

        train_imgs_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
        val_imgs_t = torch.as_tensor(val_imgs, dtype=torch.float32, device=device)
        test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)


        def save_progress(step):
            checkpoint_path = checkpoint_path_for(step)
            torch.save({
                "model_state_dict": model.state_dict(),
                "train_loss_history": np.array(train_loss_history),
                "val_loss_history": np.array(val_loss_history),
                "val_loss_epochs": np.array(val_loss_epochs),
                "latent_dim": args.latent_dim,
                "beta": args.beta,
            }, checkpoint_path)
            print(f"[epoch {step}] checkpoint saved to '{checkpoint_path}'")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="train")
            ax.plot(val_loss_epochs, val_loss_history, marker="o", label="validation")
            ax.set_title("VAE loss (recon MSE + beta*KL) against epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f"classical_vae_loss_{run_tag_base}_epoch{step}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)

            was_training = model.training
            model.eval()
            num_show = min(5, test_imgs_t.shape[0])
            fig, axes = plt.subplots(num_show, 2, figsize=(6, 3 * num_show))
            if num_show == 1:
                axes = axes.reshape(1, 2)
            with torch.no_grad():
                for i in range(num_show):
                    image = test_imgs_t[i:i + 1]
                    mu, logvar = model.encode(image)
                    recon = model.decode(mu)  # posterior mean, deterministic preview
                    mse = torch.nn.functional.mse_loss(recon, image).item()
                    axes[i, 0].imshow(image[0, 0].cpu().numpy(), cmap="gray")
                    axes[i, 0].set_title("Input" if i > 0 else "Input Data")
                    axes[i, 1].imshow(recon[0, 0].cpu().numpy(), cmap="gray")
                    axes[i, 1].set_title(f"Output (MSE={mse:.4f})")
            fig.suptitle(f"{args.dataset} class {args.norm_cls} VAE reconstructions @ epoch {step}")
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f"classical_vae_reconstruct_{run_tag_base}_epoch{step}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            if was_training:
                model.train()


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loader = DataLoader(TensorDataset(train_imgs_t), batch_size=args.batch_size, shuffle=True)

        start = time.time()
        for ep in range(args.epochs):
            model.train()
            total_loss = 0.0
            for (x,) in tqdm(loader, desc=f"VAE epoch {ep + 1}/{args.epochs}"):
                optimizer.zero_grad()
                recon, mu, logvar = model(x)
                _, loss = vae_loss_per_sample(recon, x, mu, logvar, beta=args.beta)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(loader)
            train_loss_history.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                recon, mu, logvar = model(val_imgs_t)
                _, val_loss = vae_loss_per_sample(recon, val_imgs_t, mu, logvar, beta=args.beta)
                val_loss = val_loss.item()
            val_loss_history.append(val_loss)
            val_loss_epochs.append(len(train_loss_history))
            print(f"VAE epoch {ep + 1}/{args.epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

            current_epochs = len(train_loss_history)
            if current_epochs % args.save_interval == 0:
                save_progress(current_epochs)
        elapsed = time.time() - start
        print(f"Fit in {elapsed:0.2f} seconds ({args.epochs} epochs this run)")

        current_epochs = len(train_loss_history)
        save_progress(current_epochs)

        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(test_imgs_t)
            per_sample_scores, _ = vae_loss_per_sample(recon, test_imgs_t, mu, logvar, beta=args.beta)
            per_sample_scores = per_sample_scores.cpu().numpy()

        auc_roc = roc_auc_score(test_labels, per_sample_scores)
        auc_pr = average_precision_score(test_labels, per_sample_scores)
        hp = f"beta={args.beta}"
        print(f"\n{'Detector':<14} {'Hyperparam':<14} {'AUC-ROC':<10} {'AUC-PR':<10}")
        print(f"{'VAE-Recon':<14} {hp:<14} {auc_roc:<10.4f} {auc_pr:<10.4f}")

        RESULTS_COLUMNS = ["dataset", "norm_cls", "detector", "hyperparam", "auc_roc", "auc_pr", "epochs", "seed"]
        new_df = pd.DataFrame([{"dataset": args.dataset, "norm_cls": args.norm_cls, "detector": "VAE-Recon",
                                 "hyperparam": hp, "auc_roc": auc_roc, "auc_pr": auc_pr,
                                 "epochs": current_epochs, "seed": args.seed}], columns=RESULTS_COLUMNS)
        new_df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "detector", "seed"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def _run_vae_adversarial(args):

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
            args.output_dir, f"classical_vae_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        beta = read_original_beta(args.original_results_csv, args.dataset, args.norm_cls)
        print(f"Reusing original beta={beta} (read from '{args.original_results_csv}')")

        if args.dataset == "mnist":
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)

        # ---- locate the ALREADY-TRAINED VAE checkpoint (never retrained here) ----
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            run_tag_base = (f"{args.dataset}_{args.norm_cls}_{n_train}_scale{args.train_data_scale}"
                             f"_ld{args.latent_dim}_beta{beta}_lr{args.lr}_bs{args.batch_size}")
            checkpoint_path = find_vae_checkpoint(args.checkpoint_dir, run_tag_base)
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No trained VAE checkpoint found matching run_tag_base='{run_tag_base}' under "
                f"'{args.checkpoint_dir}'. Train it first via the vae subcommand, or pass --checkpoint directly."
            )
        print(f"Loading FIXED VAE checkpoint from '{checkpoint_path}' -- never retrained")
        model = VAE.from_checkpoint(checkpoint_path, args.latent_dim).to(device)
        model.eval()

        # ---- clean ID-only train_pool, IDENTICAL to the original (clean-OOD) run (unused beyond sizing above) ----
        _train_pool_imgs, _val_imgs, _val_labels, _unused_test_imgs, _unused_test_labels, _ = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=0)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                           args.attack)
        print(f"test={len(test_imgs)} (ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)

        all_rows = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            with torch.no_grad():
                recon, mu, logvar = model(test_imgs_t)
                per_sample_scores, _ = vae_loss_per_sample(recon, test_imgs_t, mu, logvar, beta=beta)
                per_sample_scores = per_sample_scores.cpu().numpy()

            auc_roc = roc_auc_score(test_labels, per_sample_scores)
            auc_pr = average_precision_score(test_labels, per_sample_scores)
            print(f"seed={seed} VAE-Recon  beta={beta}  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
            all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                             "seed": seed, "detector": "VAE-Recon", "hyperparam": f"beta={beta}",
                             "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

        df = pd.DataFrame(all_rows, columns=ADV_RESULTS_COLUMNS)
        df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["dataset", "norm_cls", "eval_setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to '{results_csv}'")


def run_vae(args):
    if getattr(args, "adversarial_dir", None):
        _run_vae_adversarial(args)
    else:
        _run_vae_natural(args)


# ============================================================================
# Density: classical density-based detectors (DMKDE, IndepGaussian, MVGaussian)
# ============================================================================

DENSITY_RESULTS_COLUMNS = ["backbone", "setting", "seed", "detector", "hyperparam", "auc_roc", "auc_pr"]
DENSITY_ALL_DETECTORS = ["DMKDE-mixed", "IndepGaussian", "MVGaussian"]


def dmkde_score_classical(embs, train_embs):
    """classical_OOD_detectors.py's density subcommand's dmkde_score_classical(mode="mixed")."""
    embs = np.asarray(embs, dtype=np.float64)
    embs = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12)
    train_embs = np.asarray(train_embs, dtype=np.float64)
    train_embs = train_embs / (np.linalg.norm(train_embs, axis=-1, keepdims=True) + 1e-12)
    return np.mean(np.abs(embs @ train_embs.T) ** 2, axis=-1)


def independent_gaussian_score_classical(embs, mu, sigma):
    """classical_OOD_detectors.py's density subcommand's independent_gaussian_score_classical."""
    embs = np.asarray(embs, dtype=np.float64)
    d = embs.shape[1]
    const_term = 0.5 * d * np.log(2 * np.pi) + np.sum(np.log(sigma))
    quad = np.sum(((embs - mu) / sigma) ** 2, axis=-1)
    return const_term + 0.5 * quad


def multivariate_gaussian_score_classical(embs, mu, C):
    """classical_OOD_detectors.py's density subcommand's multivariate_gaussian_score_classical."""
    embs = np.asarray(embs, dtype=np.float64)
    d = embs.shape[1]
    z = embs - mu
    C_inv = np.linalg.pinv(C)
    maha = np.einsum('ni,ij,nj->n', z, C_inv, z)
    sign, logdet = np.linalg.slogdet(C)
    const_term = 0.5 * d * np.log(2 * np.pi) + 0.5 * logdet
    return const_term + 0.5 * maha


def score_density_detectors(train_embs, test_embs, test_labels, detectors):
    out = {}
    if "DMKDE-mixed" in detectors:
        t0 = time.time()
        densities = dmkde_score_classical(test_embs, train_embs)
        scores = -densities  # higher score = more anomalous, matching this project's convention
        out["DMKDE-mixed"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                               "mode=mixed", time.time() - t0)
    if "IndepGaussian" in detectors:
        t0 = time.time()
        mu = train_embs.mean(axis=0)
        sigma = train_embs.std(axis=0) + 1e-6  # matches IndependentGaussianDetector.fit()
        scores = independent_gaussian_score_classical(test_embs, mu, sigma)  # NLL, higher = more anomalous
        out["IndepGaussian"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                                 "", time.time() - t0)
    if "MVGaussian" in detectors:
        t0 = time.time()
        mu = train_embs.mean(axis=0)
        centered = train_embs - mu
        eps = 1e-6
        C = (centered.T @ centered) / max(1, train_embs.shape[0] - 1) + eps * np.eye(train_embs.shape[1])  # matches MultivariateGaussianDetector.fit()
        scores = multivariate_gaussian_score_classical(test_embs, mu, C)  # exact closed-form, no QPE approximation needed
        out["MVGaussian"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                              "", time.time() - t0)
    return out



def _run_density_natural(args):

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
            args.output_dir, f"classical_density_{args.dataset}_normcls{args.norm_cls}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        if args.dataset == "mnist":
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)
        print(f"train_data_scale={args.train_data_scale} -> n_train={n_train} "
              f"(of {_class_count} available {args.norm_cls}-class training images)")

        backbone = PretrainedFeatureExtractor(args.backbone, img_size=args.backbone_img_size, device=device)
        print(f"{args.backbone}: feature_dim={backbone.feature_dim}")

        rows = []
        for seed in args.seeds:
            # train_pool/val at train_data_scale; test at test_data_scale -- same
            # seed so val is identical/disjoint between the two calls. Reloaded
            # fresh per seed -- val/test composition
            # varies by seed even though train_pool and the backbone do not.
            train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, _unused_ood_str = \
                load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                           n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=seed)
            _unused_train_pool, _unused_val_imgs, _unused_val_labels, test_imgs, test_labels, ood_str = \
                load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                           n_train, args.n_val_per_cls, args.test_data_scale, args.img_size, seed=seed)

            train_imgs_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
            test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)
            train_embs = backbone.get_embedding(train_imgs_t).cpu().numpy()
            test_embs = backbone.get_embedding(test_imgs_t).cpu().numpy()

            print(f"\n-- {args.backbone} / native / seed={seed} -- train_pool={len(train_pool_imgs)}, "
                  f"val={len(val_imgs)}, test={len(test_imgs)} ({ood_str})")
            out = score_density_detectors(train_embs, test_embs, test_labels, args.detectors)
            print(f"{'Detector':<14} {'Hyperparam':<12} {'AUC-ROC':<10} {'AUC-PR':<10}")
            for det, (auc_roc, auc_pr, hp, infer_time) in out.items():
                print(f"{det:<14} {hp:<12} {auc_roc:<10.4f} {auc_pr:<10.4f}")
                rows.append({"backbone": args.backbone, "setting": "native", "seed": seed, "detector": det,
                             "hyperparam": hp, "auc_roc": auc_roc, "auc_pr": auc_pr})

        new_df = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
        new_df.to_csv(results_csv, mode="a", header=not os.path.exists(results_csv), index=False)
        full_df = pd.read_csv(results_csv)
        full_df = full_df.drop_duplicates(subset=["backbone", "setting", "seed", "detector"], keep="last")
        full_df.to_csv(results_csv, index=False)

        print(f"\n{'=' * 100}\nFINAL RESULTS ({results_csv})\n{'=' * 100}")
        print(full_df.to_string(index=False))


def _run_density_adversarial(args):

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
            args.output_dir, f"classical_density_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        if args.dataset == "mnist":
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)

        # ---- clean ID-only train_pool, IDENTICAL to the original (clean-OOD) run ----
        train_pool_imgs, _val_imgs, _val_labels, _unused_test_imgs, _unused_test_labels, _ = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=0)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                           args.attack)
        print(f"train_pool={len(train_pool_imgs)}, test={len(test_imgs)} "
              f"(ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        backbone = PretrainedFeatureExtractor(args.backbone, img_size=args.backbone_img_size, device=device)
        train_imgs_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
        test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)
        train_embs = backbone.get_embedding(train_imgs_t).cpu().numpy()
        test_embs = backbone.get_embedding(test_imgs_t).cpu().numpy()

        all_rows = []
        for seed in args.seeds:
            print(f"\n-- {args.backbone} / seed={seed} --")
            out = score_density_detectors(train_embs, test_embs, test_labels, args.detectors)
            for det, (auc_roc, auc_pr, hp, _infer_time) in out.items():
                print(f"{det:<14} {hp:<12} AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
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


# ============================================================================
# Distance: distance-based detectors (DeepKNN, DeepMean, DeepMedoids, DeepSVDD), native pretrained-backbone embeddings or a 6D learned projection
# ============================================================================

DISTANCE_RESULTS_COLUMNS = ["backbone", "setting", "feature_loss", "seed", "detector", "hyperparam", "auc_roc", "auc_pr"]


ALL_LOSSES = ["random", "compact", "cosine", "vicreg"]
ALL_DETECTORS = ["DeepKNN", "Deep-Mean", "Deep-Medoids", "DeepSVDD"]
ABLATION_COLUMNS = ["feature_loss", "detector", "hyperparam", "auc_roc", "auc_pr"]
RESULTS_COLUMNS = ["backbone", "setting", "feature_loss", "seed", "detector", "hyperparam", "auc_roc", "auc_pr"]


def load_split(args, n_train, seed):
    """train_pool/val at train_data_scale; test at test_data_scale -- same
    seed so val is identical/disjoint between the two calls. Mirrors
    run_quanforge_pipeline_mnist.py's _load_data(seed) closure."""
    train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, _unused_ood_str = \
        load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                   n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=seed)
    _unused_train_pool, _unused_val_imgs, _unused_val_labels, test_imgs, test_labels, ood_str = \
        load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                   n_train, args.n_val_per_cls, args.test_data_scale, args.img_size, seed=seed)
    return train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, ood_str


def score_detectors(train_embs, val_embs, val_labels, test_embs, test_labels,
                     emb_source, train_pool_imgs, test_imgs, args, seed, detectors):
    """emb_source exposes get_embedding/feature_dim/device (a
    ComposedEmbedding), matching whatever produced train/val/test_embs.
    Tunes K/M on val, scores each requested detector on test. Returns
    {detector: (auc_roc, auc_pr, hyperparam_str)}."""
    out = {}
    if "DeepKNN" in detectors:
        best_k, _ = select_k_euclidean(train_embs, val_embs, val_labels, args.k_candidates)
        scores = knn_scores_euclidean(train_embs, test_embs, best_k)
        out["DeepKNN"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                           f"K={best_k}")
    if "Deep-Mean" in detectors:
        scores = mean_scores_euclidean(train_embs, test_embs)
        out["Deep-Mean"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores), "")
    if "Deep-Medoids" in detectors:
        best_m, _ = select_m_euclidean(train_embs, val_embs, val_labels, args.m_candidates, seed=seed)
        scores = medoid_scores_euclidean(train_embs, test_embs, best_m, seed=seed)
        out["Deep-Medoids"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                                f"M={best_m}")
    if "DeepSVDD" in detectors:
        svdd = PretrainedDeepSVDDDetector(emb_source, proj_dim=args.svdd_proj_dim)
        svdd.initialize_center(train_pool_imgs)
        svdd.train_svdd(train_pool_imgs, args.svdd_epochs, args.svdd_lr, args.svdd_batch_size, args.svdd_lambda)
        scores = svdd.predict_score(test_imgs)
        out["DeepSVDD"] = (roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores),
                            f"lambda={args.svdd_lambda}")
    return out


def run_ablation(args, n_train, device):
    """resnet18 only. Trains all args.ablation_losses variants, scores each
    on VAL with fixed K/M (args.ablation_k/m, no tuning), averages the
    requested detectors' val AUC-ROC per loss, returns
    (best_loss, ablation_df, {loss: mean_val_auc_roc}).

    Uses a single fixed split (args.ablation_seed) -- the ablation phase
    doesn't sweep seeds itself, it's a one-shot check to pick a loss."""
    train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, _unused_ood_str = \
        load_split(args, n_train, args.ablation_seed)
    backbone = PretrainedFeatureExtractor("resnet18", img_size=args.backbone_img_size, device=device)
    aug_kwargs = dict(max_rotate=args.aug_max_rotate, max_translate=args.aug_max_translate,
                       noise_std=args.aug_noise_std)
    vicreg_kwargs = dict(lambda_inv=args.vicreg_lambda_inv, lambda_var=args.vicreg_lambda_var,
                          lambda_cov=args.vicreg_lambda_cov, gamma=args.vicreg_gamma)
    train_imgs_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
    val_imgs_t = torch.as_tensor(val_imgs, dtype=torch.float32, device=device)

    rows = []
    mean_auc_by_loss = {}
    for feature_loss in args.ablation_losses:
        print(f"\n{'=' * 80}\nAblation: feature_loss={feature_loss} (resnet18)\n{'=' * 80}")
        torch.manual_seed(args.ablation_seed)
        np.random.seed(args.ablation_seed)
        projector = LinearProjector(backbone.feature_dim, args.proj_dim).to(device)
        train_projector(feature_loss, backbone, projector, train_pool_imgs, args.ablation_epochs,
                         args.ablation_lr, args.ablation_batch_size, device, aug_kwargs, vicreg_kwargs,
                         seed=args.ablation_seed)
        emb_source = ComposedEmbedding(backbone, projector)
        train_embs = emb_source.get_embedding(train_imgs_t).cpu().numpy()
        val_embs = emb_source.get_embedding(val_imgs_t).cpu().numpy()

        detector_aucs = []
        if "DeepKNN" in args.ablation_detectors:
            scores = knn_scores_euclidean(train_embs, val_embs, args.ablation_k)
            auc_roc = roc_auc_score(val_labels, scores)
            rows.append({"feature_loss": feature_loss, "detector": "DeepKNN", "hyperparam": f"K={args.ablation_k}",
                         "auc_roc": auc_roc, "auc_pr": average_precision_score(val_labels, scores)})
            detector_aucs.append(auc_roc)
        if "Deep-Mean" in args.ablation_detectors:
            scores = mean_scores_euclidean(train_embs, val_embs)
            auc_roc = roc_auc_score(val_labels, scores)
            rows.append({"feature_loss": feature_loss, "detector": "Deep-Mean", "hyperparam": "",
                         "auc_roc": auc_roc, "auc_pr": average_precision_score(val_labels, scores)})
            detector_aucs.append(auc_roc)
        if "Deep-Medoids" in args.ablation_detectors:
            scores = medoid_scores_euclidean(train_embs, val_embs, args.ablation_m, seed=args.ablation_seed)
            auc_roc = roc_auc_score(val_labels, scores)
            rows.append({"feature_loss": feature_loss, "detector": "Deep-Medoids", "hyperparam": f"M={args.ablation_m}",
                         "auc_roc": auc_roc, "auc_pr": average_precision_score(val_labels, scores)})
            detector_aucs.append(auc_roc)
        if "DeepSVDD" in args.ablation_detectors:
            svdd = PretrainedDeepSVDDDetector(emb_source, proj_dim=args.svdd_proj_dim)
            svdd.initialize_center(train_pool_imgs)
            svdd.train_svdd(train_pool_imgs, args.svdd_epochs, args.svdd_lr, args.svdd_batch_size, args.svdd_lambda)
            scores = svdd.predict_score(val_imgs)
            auc_roc = roc_auc_score(val_labels, scores)
            rows.append({"feature_loss": feature_loss, "detector": "DeepSVDD", "hyperparam": f"lambda={args.svdd_lambda}",
                         "auc_roc": auc_roc, "auc_pr": average_precision_score(val_labels, scores)})
            detector_aucs.append(auc_roc)

        mean_auc = float(np.mean(detector_aucs)) if detector_aucs else float("nan")
        mean_auc_by_loss[feature_loss] = mean_auc
        rows.append({"feature_loss": feature_loss, "detector": "MEAN", "hyperparam": "",
                     "auc_roc": mean_auc, "auc_pr": float("nan")})
        print(f"feature_loss={feature_loss}: mean val AUC-ROC across detectors = {mean_auc:.4f}")

    ablation_df = pd.DataFrame(rows, columns=ABLATION_COLUMNS)
    best_loss = max(mean_auc_by_loss, key=mean_auc_by_loss.get)
    print(f"\nBest feature_loss selected on validation: {best_loss} "
          f"(mean val AUC-ROC={mean_auc_by_loss[best_loss]:.4f})")
    return best_loss, ablation_df, mean_auc_by_loss


def run_final_comparison(best_loss, args, n_train, device):
    aug_kwargs = dict(max_rotate=args.aug_max_rotate, max_translate=args.aug_max_translate,
                       noise_std=args.aug_noise_std)
    vicreg_kwargs = dict(lambda_inv=args.vicreg_lambda_inv, lambda_var=args.vicreg_lambda_var,
                          lambda_cov=args.vicreg_lambda_cov, gamma=args.vicreg_gamma)

    rows = []
    run_native = args.settings in ("native", "both")
    run_6d = args.settings in ("6d", "both")

    for backbone_name in args.final_backbones:
        print(f"\n{'#' * 90}\nFinal comparison: backbone={backbone_name}\n{'#' * 90}")
        backbone = PretrainedFeatureExtractor(backbone_name, img_size=args.backbone_img_size, device=device)

        if run_native:
            # native setting: backbone itself is frozen, but the split is
            # reloaded per seed (see load_split), so val/test embeddings
            # still vary seed to seed even though the backbone never changes.
            native_source = ComposedEmbedding(backbone, projector=None)
            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, _ = load_split(args, n_train, seed)
                train_imgs_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
                val_imgs_t = torch.as_tensor(val_imgs, dtype=torch.float32, device=device)
                test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)
                train_embs_native = native_source.get_embedding(train_imgs_t).cpu().numpy()
                val_embs_native = native_source.get_embedding(val_imgs_t).cpu().numpy()
                test_embs_native = native_source.get_embedding(test_imgs_t).cpu().numpy()

                print(f"\n-- {backbone_name} / native (dim={backbone.feature_dim}) / seed={seed} --")
                out = score_detectors(train_embs_native, val_embs_native, val_labels, test_embs_native, test_labels,
                                       native_source, train_pool_imgs, test_imgs, args, seed, args.detectors)
                for det, (auc_roc, auc_pr, hp) in out.items():
                    print(f"{backbone_name:<14} {'native':<8} {det:<14} {hp:<14} {auc_roc:<10.4f} {auc_pr:<10.4f}")
                    rows.append({"backbone": backbone_name, "setting": "native", "feature_loss": "",
                                 "seed": seed, "detector": det, "hyperparam": hp, "auc_roc": auc_roc, "auc_pr": auc_pr})

        if run_6d:
            # 6D setting: a fresh projector (init + training) per seed, on
            # that seed's own reloaded split, so the sweep captures split-
            # sampling, projector-training, AND detector-fitting variance.
            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, _ = load_split(args, n_train, seed)
                train_imgs_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
                val_imgs_t = torch.as_tensor(val_imgs, dtype=torch.float32, device=device)
                test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)

                print(f"\n-- {backbone_name} / 6D (loss={best_loss}, dim={args.proj_dim}) / seed={seed} --")
                projector = LinearProjector(backbone.feature_dim, args.proj_dim).to(device)
                train_projector(best_loss, backbone, projector, train_pool_imgs, args.final_epochs, args.final_lr,
                                 args.final_batch_size, device, aug_kwargs, vicreg_kwargs, seed=seed)
                proj_source = ComposedEmbedding(backbone, projector)
                train_embs_6d = proj_source.get_embedding(train_imgs_t).cpu().numpy()
                val_embs_6d = proj_source.get_embedding(val_imgs_t).cpu().numpy()
                test_embs_6d = proj_source.get_embedding(test_imgs_t).cpu().numpy()
                out = score_detectors(train_embs_6d, val_embs_6d, val_labels, test_embs_6d, test_labels,
                                       proj_source, train_pool_imgs, test_imgs, args, seed, args.detectors)
                for det, (auc_roc, auc_pr, hp) in out.items():
                    print(f"{backbone_name:<14} {'6D':<8} {det:<14} {hp:<14} {auc_roc:<10.4f} {auc_pr:<10.4f}")
                    rows.append({"backbone": backbone_name, "setting": "6D", "feature_loss": best_loss,
                                 "seed": seed, "detector": det, "hyperparam": hp, "auc_roc": auc_roc, "auc_pr": auc_pr})

    return pd.DataFrame(rows, columns=RESULTS_COLUMNS)


def read_original_hyperparam(original_results_csv, dataset, norm_cls, seed, detector):
    """Pulls e.g. 'K=3' -> 3 out of the ORIGINAL run's own results CSV --
    the whole point being to reuse exactly what that run already decided,
    not re-select anything here."""
    df = pd.read_csv(original_results_csv)
    row = df[(df["seed"] == seed) & (df["detector"] == detector)]
    assert len(row) == 1, f"expected exactly 1 original row for seed={seed} detector={detector}, got {len(row)}"
    hp = row["hyperparam"].iloc[0]
    if pd.isna(hp) or hp == "":
        return None
    return int(str(hp).split("=")[1])



def _run_distance_natural(args):

        needs_6d = args.settings in ("6d", "both")
        if needs_6d and args.skip_ablation and args.best_loss is None:
            raise ValueError("--skip_ablation requires --best_loss when --settings includes the 6D setting")

        os.makedirs(args.output_dir, exist_ok=True)
        args.ablation_csv = args.ablation_csv or os.path.join(args.output_dir, "ablation_results.csv")
        args.results_csv = args.results_csv or os.path.join(args.output_dir, "final_results.csv")
        os.makedirs(os.path.dirname(args.ablation_csv) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)

        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False "
                                    "(check `nvidia-smi` and that this PyTorch build has a matching CUDA runtime)")
            device = torch.device("cuda")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if args.dataset == "mnist":
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)
        print(f"train_data_scale={args.train_data_scale} -> n_train={n_train} "
              f"(of {_class_count} available {args.norm_cls}-class training images)")
        print("Note: the train/val/test split is reloaded per seed (load_split).")

        best_loss = args.best_loss
        if not needs_6d:
            print(f"\n--settings={args.settings} -> no projector involved, skipping the ablation phase entirely")
        elif args.skip_ablation:
            print(f"\n--skip_ablation set -> using --best_loss={best_loss} directly, no ablation run")
        else:
            best_loss, ablation_df, mean_auc_by_loss = run_ablation(args, n_train, device)
            ablation_df.to_csv(args.ablation_csv, mode="a", header=not os.path.exists(args.ablation_csv), index=False)
            full_ablation_df = pd.read_csv(args.ablation_csv)
            full_ablation_df = full_ablation_df.drop_duplicates(subset=["feature_loss", "detector"], keep="last")
            full_ablation_df.to_csv(args.ablation_csv, index=False)
            print(f"\nAblation results saved to '{args.ablation_csv}'")

        results_df = run_final_comparison(best_loss, args, n_train, device)
        results_df.to_csv(args.results_csv, mode="a", header=not os.path.exists(args.results_csv), index=False)
        full_results_df = pd.read_csv(args.results_csv)
        full_results_df = full_results_df.drop_duplicates(subset=["backbone", "setting", "feature_loss", "seed", "detector"],
                                                            keep="last")
        full_results_df.to_csv(args.results_csv, index=False)

        print(f"\n{'=' * 100}\nFINAL RESULTS -- mean +/- std across seeds ({args.results_csv})\n{'=' * 100}")
        summary = full_results_df.groupby(["backbone", "setting", "detector"]).agg(
            auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
            auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
            n_seeds=("seed", "nunique"),
        ).reset_index()
        print(summary.to_string(index=False))


def _run_distance_adversarial(args):

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
            args.output_dir, f"classical_distance_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
        os.makedirs(os.path.dirname(results_csv) or ".", exist_ok=True)

        if args.dataset == "mnist":
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        else:
            from torchvision import datasets, transforms
            _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
        n_train = int(args.train_data_scale * _class_count)

        # ---- clean ID-only train_pool, IDENTICAL to the original (clean-OOD) run ----
        train_pool_imgs, _val_imgs, _val_labels, _unused_test_imgs, _unused_test_labels, _ = \
            load_ood_split_scaled_raw(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                                       n_train, args.n_val_per_cls, args.train_data_scale, args.img_size, seed=0)

        # ---- test set: clean ID (never adversarial) + adversarial OOD (never other-class) ----
        test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                           args.attack)
        print(f"train_pool={len(train_pool_imgs)}, test={len(test_imgs)} "
              f"(ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
        print(f"composition: {composition}")
        # merged into every results row below, so each row records exactly how
        # many clean/adversarial-per-attack images (including any padding/
        # duplicates) made up the test set it was scored against.
        composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                             "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

        backbone = PretrainedFeatureExtractor(args.backbone, img_size=args.backbone_img_size, device=device)
        train_imgs_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
        test_imgs_t = torch.as_tensor(test_imgs, dtype=torch.float32, device=device)
        train_embs = backbone.get_embedding(train_imgs_t).cpu().numpy()
        test_embs = backbone.get_embedding(test_imgs_t).cpu().numpy()

        all_rows = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"\n-- {args.backbone} / seed={seed} --")

            if "DeepKNN" in args.detectors:
                best_k = read_original_hyperparam(args.original_results_csv, args.dataset, args.norm_cls, seed, "DeepKNN")
                scores = knn_scores_euclidean(train_embs, test_embs, best_k)
                auc_roc, auc_pr = roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores)
                print(f"DeepKNN      K={best_k:<6} AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": "DeepKNN", "hyperparam": f"K={best_k}",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

            if "Deep-Mean" in args.detectors:
                scores = mean_scores_euclidean(train_embs, test_embs)
                auc_roc, auc_pr = roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores)
                print(f"Deep-Mean            AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": "Deep-Mean", "hyperparam": "",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

            if "Deep-Medoids" in args.detectors:
                best_m = read_original_hyperparam(args.original_results_csv, args.dataset, args.norm_cls, seed, "Deep-Medoids")
                scores = medoid_scores_euclidean(train_embs, test_embs, best_m, seed=seed)
                auc_roc, auc_pr = roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores)
                print(f"Deep-Medoids M={best_m:<6} AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": "Deep-Medoids", "hyperparam": f"M={best_m}",
                                 "auc_roc": auc_roc, "auc_pr": auc_pr, **composition_cols})

            if "DeepSVDD" in args.detectors:
                # re-fit on the SAME clean ID train_pool as always (see module
                # docstring) -- no cached checkpoint exists to reuse instead.
                svdd = PretrainedDeepSVDDDetector(backbone, proj_dim=args.svdd_proj_dim)
                svdd.initialize_center(train_pool_imgs)
                svdd.train_svdd(train_pool_imgs, args.svdd_epochs, args.svdd_lr, args.svdd_batch_size, args.svdd_lambda)
                scores = svdd.predict_score(test_imgs)
                auc_roc, auc_pr = roc_auc_score(test_labels, scores), average_precision_score(test_labels, scores)
                print(f"DeepSVDD             AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
                all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                                 "seed": seed, "detector": "DeepSVDD", "hyperparam": f"lambda={args.svdd_lambda}",
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
# GAN: Classical-AnoGAN / Classical-WGAN-GP GAN-based OOD detectors
# ============================================================================

class ClassicalGenerator(nn.Module):
    """noise z -> MLP -> fake image (L2-normalized). Counterpart of QuantumGenerator."""
    def __init__(self, latent_dim, hidden, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid(),
        )
    def forward(self, z):
        x = self.net(z)
        return F.normalize(x, p=2, dim=1)

class ClassicalDiscriminator(nn.Module):
    """image -> real/fake probability (BCE). Counterpart of QuantumDiscriminator."""
    def __init__(self, feature_dim, hidden, bce_eps=0.0):
        super().__init__()
        self.bce_eps = bce_eps
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        prob = self.net(x)
        if self.bce_eps > 0:
            prob = torch.clamp(prob, self.bce_eps, 1.0 - self.bce_eps)
        return prob

class ClassicalCritic(nn.Module):
    """Unbounded critic for WGAN-GP (same as the critic used in qgan_ood.py)."""
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)

# ===============================================================
# Data loading (identical to qgan_ood.py)
# ===============================================================
def load_real_dataset(name, data_dir, target_class, scale=0.2, test_scale=1.0, img_shape=(16, 16)):
    if name == "mnist":
        train_set = tv_datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = tv_datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        train_set = tv_datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = tv_datasets.FashionMNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

    train_class_idx = torch.where(train_set.targets == target_class)[0]
    test_class_idx = torch.where(test_set.targets == target_class)[0]
    train_pool_idx = train_class_idx[:int(scale * len(train_class_idx))]
    train_size = int(0.8 * len(train_pool_idx))
    train_idx = train_pool_idx[:train_size]
    val_idx = train_pool_idx[train_size:]
    test_idx = test_class_idx[:int(test_scale * len(test_class_idx))]

    print(f"  [{name}] class {target_class}, train_scale={scale*100:.0f}%, test_scale={test_scale*100:.0f}%: "
          f"train={len(train_idx)}, val={len(val_idx)}, test_id={len(test_idx)}")

    train_images = normalize_images(resize_and_flatten(train_set, train_idx, img_shape))
    val_images = normalize_images(resize_and_flatten(train_set, val_idx, img_shape))
    test_images = normalize_images(resize_and_flatten(test_set, test_idx, img_shape))
    return train_images, val_images, test_images

def load_ood_test(name, data_dir, target_class, test_scale, n_id_test, setting,
                  ood_class, img_shape=(16, 16), seed=42):
    """OOD count exactly equals ID count (50/50), random & evenly across non-ID classes."""
    rng = np.random.RandomState(seed)
    if name == "mnist":
        test_set = tv_datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        test_set = tv_datasets.FashionMNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

    test_targets = test_set.targets
    ood_classes = [ood_class] if setting == 2 else [c for c in range(10) if c != target_class]
    n_ood_classes = len(ood_classes)
    base_per_class = n_id_test // n_ood_classes
    remainder = n_id_test % n_ood_classes
    extra_classes = set(rng.choice(ood_classes, size=remainder, replace=False).tolist())

    ood_list = []
    for c in ood_classes:
        c_idx = torch.where(test_targets == c)[0]
        c_idx = c_idx[:int(test_scale * len(c_idx))]
        n_c = base_per_class + (1 if c in extra_classes else 0)
        n_c = min(n_c, len(c_idx))
        chosen = rng.choice(len(c_idx), size=n_c, replace=False)
        c_idx_sampled = c_idx[chosen]
        if len(c_idx_sampled) > 0:
            imgs = normalize_images(resize_and_flatten(test_set, c_idx_sampled, img_shape))
            ood_list.append(imgs)

    ood_images = np.concatenate(ood_list, axis=0)
    if len(ood_images) > n_id_test:
        ood_images = ood_images[:n_id_test]
    elif len(ood_images) < n_id_test:
        pad = rng.choice(len(ood_images), size=n_id_test - len(ood_images), replace=True)
        ood_images = np.concatenate([ood_images, ood_images[pad]], axis=0)
    print(f"  test_ood={len(ood_images)} (~{base_per_class}/class, ID/OOD=50/50)")
    return ood_images

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

# ===============================================================
# Training: Classical AnoGAN (BCE)
# ===============================================================
def train_c_anogan(G, D, train_loader, val_loader, args, device):
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    history = {"g_loss": [], "d_loss": [], "val_residual": []}
    best_val, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(args.epochs):
        G.train(); D.train()
        ep_g, ep_d, n_batches = 0.0, 0.0, 0
        for (real_data,) in train_loader:
            bs = real_data.shape[0]
            real_data = real_data.to(device)
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            opt_d.zero_grad()
            z = torch.empty(bs, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
            d_loss = (criterion(D(real_data), valid) + criterion(D(G(z).detach()), fake)) / 2.0
            if torch.isnan(d_loss) or torch.isinf(d_loss):
                continue
            d_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
            opt_d.step()

            for p in D.parameters(): p.requires_grad = False
            opt_g.zero_grad()
            z = torch.empty(bs, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
            g_loss = criterion(D(G(z)), valid)
            if torch.isnan(g_loss) or torch.isinf(g_loss):
                for p in D.parameters(): p.requires_grad = True
                continue
            g_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            opt_g.step()
            for p in D.parameters(): p.requires_grad = True

            ep_g += g_loss.item(); ep_d += d_loss.item(); n_batches += 1

        if n_batches == 0:
            print(f"  Epoch {epoch+1}: all batches skipped, stopping."); break
        avg_g, avg_d = ep_g / n_batches, ep_d / n_batches
        history["g_loss"].append(avg_g); history["d_loss"].append(avg_d)
        val_res = compute_val_residual(G, val_loader, args, device)
        history["val_residual"].append(val_res)
        if val_res < best_val:
            best_val, best_epoch = val_res, epoch + 1
            best_state = {"G": {k: v.clone() for k, v in G.state_dict().items()},
                          "D": {k: v.clone() for k, v in D.state_dict().items()}}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | D_loss={avg_d:.4f} G_loss={avg_g:.4f} | Val_L1={val_res:.6f}")

    if best_state is not None:
        G.load_state_dict(best_state["G"]); D.load_state_dict(best_state["D"])
    print(f"  Best epoch {best_epoch}, val_residual={best_val:.6f}")
    return history, best_val, best_epoch

# ===============================================================
# Training: Classical WGAN-GP
# ===============================================================
def compute_gradient_penalty(D, real_data, fake_data, device):
    bs = real_data.shape[0]
    alpha = torch.rand(bs, 1, device=device)
    interp = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interp = D(interp)
    grad = torch.autograd.grad(outputs=d_interp, inputs=interp,
                               grad_outputs=torch.ones_like(d_interp),
                               create_graph=True, retain_graph=True)[0]
    grad = grad.view(bs, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

def train_cwgan_gp(G, D, train_loader, val_loader, args, device):
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    history = {"g_loss": [], "d_loss": [], "val_residual": []}
    best_val, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(args.epochs):
        G.train(); D.train()
        ep_g, ep_d, n_g = 0.0, 0.0, 0
        data_iter = iter(train_loader)
        n_batches = len(train_loader)
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
                        torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
                    opt_d.step()
                ep_d += d_loss.item()

            for p in D.parameters(): p.requires_grad = False
            opt_g.zero_grad()
            z = torch.empty(bs, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
            g_loss = -D(G(z)).mean()
            if not (torch.isnan(g_loss) or torch.isinf(g_loss)):
                g_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
                opt_g.step()
            for p in D.parameters(): p.requires_grad = True
            ep_g += g_loss.item(); n_g += 1

        avg_g = ep_g / max(n_g, 1); avg_d = ep_d / max(n_batches * args.n_critic, 1)
        history["g_loss"].append(avg_g); history["d_loss"].append(avg_d)
        val_res = compute_val_residual(G, val_loader, args, device)
        history["val_residual"].append(val_res)
        if val_res < best_val:
            best_val, best_epoch = val_res, epoch + 1
            best_state = {"G": {k: v.clone() for k, v in G.state_dict().items()},
                          "D": {k: v.clone() for k, v in D.state_dict().items()}}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | C_loss={avg_d:.4f} G_loss={avg_g:.4f} | Val_L1={val_res:.6f}")

    G.load_state_dict(best_state["G"]); D.load_state_dict(best_state["D"])
    print(f"  Best epoch {best_epoch}, val_residual={best_val:.6f}")
    return history, best_val, best_epoch

def compute_val_residual(G, val_loader, args, device):
    G.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)
            z = torch.zeros(x.shape[0], args.latent_dim, device=device)
            total += torch.mean(torch.abs(x - G(z))).item() * x.shape[0]
            n += x.shape[0]
    return total / max(n, 1)

# ===============================================================
# AnoGAN inference (identical formula to qgan_ood.py)
# ===============================================================
def find_optimal_z(G, D, x, args, device):
    G.eval(); D.eval()
    bs = x.shape[0]
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
        discrim = torch.abs(d_real - d_fake)
        score = (1.0 / args.alpha) * residual + args.alpha * discrim
        score.mean().backward()
        optimizer.step()
    with torch.no_grad():
        fake = G(z); d_fake = D(fake)
        residual = torch.sum(torch.abs(x - fake), dim=1)
        discrim = torch.abs(d_real.squeeze() - d_fake.squeeze())
        scores = (1.0 / args.alpha) * residual + args.alpha * discrim
    for p in G.parameters(): p.requires_grad = True
    for p in D.parameters(): p.requires_grad = True
    return scores.detach().cpu().numpy()

def compute_anomaly_scores(G, D, test_loader, args, device):
    all_scores, all_times = [], []
    for x_batch, _ in tqdm(test_loader, desc="  Anomaly detection", leave=False):
        x_batch = x_batch.to(device)
        t0 = time.time()
        all_scores.append(find_optimal_z(G, D, x_batch, args, device))
        all_times.append(time.time() - t0)
    return np.concatenate(all_scores), sum(all_times)

def evaluate(scores, labels):
    results = {"auc_roc": roc_auc_score(labels, scores),
               "auc_pr": average_precision_score(labels, scores)}
    id_scores = scores[labels == 0]
    for q in [0.70, 0.80, 0.90, 0.95, 0.99]:
        thresh = np.quantile(id_scores, q)
        results[f"f1_{int(q*100)}"] = f1_score(labels, (scores > thresh).astype(int))
    return results

def plot_losses(history, save_path, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    e = range(1, len(history["g_loss"]) + 1)
    a1.plot(e, history["g_loss"], label="Generator"); a1.plot(e, history["d_loss"], label="Discriminator/Critic")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.set_title(f"{title} - Loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(e, history["val_residual"], color="green"); a2.set_xlabel("Epoch")
    a2.set_ylabel("Mean L1 Residual"); a2.set_title(f"{title} - Val Residual"); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_samples(G, args, device, save_path, n=16):
    G.eval()
    with torch.no_grad():
        z = torch.empty(n, args.latent_dim, device=device).uniform_(-np.pi, np.pi)
        fake = G(z).cpu().numpy()
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    for i, ax in enumerate(axes):
        ax.imshow(fake[i].reshape(args.img_size, args.img_size), cmap="gray"); ax.axis("off")
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()

def plot_scores(scores, labels, save_path, title):
    plt.figure(figsize=(8, 4))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="ID", density=True)
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="OOD", density=True)
    plt.xlabel("Anomaly score"); plt.ylabel("Density"); plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def find_gan_checkpoint(checkpoint_dir, norm_cls, gan_name, seed):
    pattern = os.path.join(checkpoint_dir, f"*_id{norm_cls}_ood*_{gan_name}_seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 run dir matching '{pattern}', found {len(candidates)}: {candidates}. "
            f"Train it first via canogan_ood.py."
        )
    model_path = os.path.join(candidates[0], f"{gan_name}_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Run dir '{candidates[0]}' found but '{model_path}' is missing.")
    return model_path



def _run_gan_natural(args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.fast_test:
            args.epochs = 3; args.z_iter = 10; args.seeds = [0]; args.scale = 0.05; args.test_scale = 0.1
            args.batch_size = 32

        os.makedirs(args.save_dir, exist_ok=True)
        feature_dim = args.img_size * args.img_size
        print(f"PyTorch: {torch.__version__}, Device: {device}")
        print(f"Classical GAN, latent_dim={args.latent_dim}, hidden_g={args.hidden_g}, hidden_d={args.hidden_d}")
        print(f"gan_type={args.gan_type}, epochs={args.epochs}, z_iter={args.z_iter}, seeds={args.seeds}")

        csv_columns = [
            "dataset", "setting", "gan_type", "model_family", "detector",
            "seed", "id_class", "ood_class", "scale", "test_scale",
            "latent_dim", "hidden_g", "hidden_d", "feature_dim",
            "epochs", "batch_size", "lr_g", "lr_d", "n_critic", "lambda_gp",
            "z_iter", "z_lr", "alpha", "grad_clip", "bce_eps",
            "n_train", "n_val", "n_test_id", "n_test_ood",
            "g_params", "d_params", "train_time", "best_epoch", "best_val_residual",
            "final_g_loss", "final_d_loss", "infer_time",
            "auc_roc", "auc_pr", "f1_70", "f1_80", "f1_90", "f1_95", "f1_99",
        ]
        csv_path = os.path.join(args.save_dir, "cgan_results.csv")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)

        id_classes = list(range(10)) if args.run_all_id else [args.target_class]
        for id_cls in id_classes:
            print(f"\n{'#'*70}\n# ID class: {id_cls}\n{'#'*70}")
            img_shape = (args.img_size, args.img_size)
            train_np, val_np, test_id_np = load_real_dataset(
                args.dataset, args.data_dir, id_cls, args.scale, args.test_scale, img_shape)
            test_ood_np = load_ood_test(
                args.dataset, args.data_dir, id_cls, args.test_scale, len(test_id_np),
                args.setting, args.ood_class, img_shape)
            train_loader, val_loader, test_loader, test_labels = build_dataloaders(
                train_np, val_np, test_id_np, test_ood_np, args.batch_size)
            ood_str = f"all_except_{id_cls}" if args.setting == 1 else str(args.ood_class)

            for seed in args.seeds:
                set_seed(seed)
                print(f"\n{'='*60}\nSeed: {seed}\n{'='*60}")
                gan_types = ["c_anogan", "cwgan_gp"] if args.gan_type == "both" else [args.gan_type]

                for gan_name in gan_types:
                    print(f"\n--- {gan_name} ---")
                    set_seed(seed)
                    run_tag = (f"{args.dataset}_set{args.setting}_id{id_cls}_ood{ood_str}"
                               f"_lat{args.latent_dim}_{gan_name}_seed{seed}")
                    run_dir = os.path.join(args.save_dir, run_tag)
                    os.makedirs(run_dir, exist_ok=True)

                    G = ClassicalGenerator(args.latent_dim, args.hidden_g, feature_dim).to(device)
                    t0 = time.time()
                    if gan_name == "c_anogan":
                        D = ClassicalDiscriminator(feature_dim, args.hidden_d, args.bce_eps).to(device)
                        detector = "Classical-AnoGAN"
                        history, best_val, best_epoch = train_c_anogan(G, D, train_loader, val_loader, args, device)
                    else:
                        D = ClassicalCritic(feature_dim).to(device)
                        detector = "Classical-WGAN-GP"
                        history, best_val, best_epoch = train_cwgan_gp(G, D, train_loader, val_loader, args, device)
                    train_time = time.time() - t0

                    g_params, d_params = count_params(G), count_params(D)
                    print(f"  G params: {g_params}, D params: {d_params}")

                    torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(args),
                                "history": history, "best_epoch": best_epoch},
                               os.path.join(run_dir, f"{gan_name}_model.pt"))
                    np.savez(os.path.join(run_dir, f"{gan_name}_history.npz"), **history)
                    plot_losses(history, os.path.join(run_dir, f"{gan_name}_loss.png"), detector)
                    plot_samples(G, args, device, os.path.join(run_dir, f"{gan_name}_samples.png"))

                    t1 = time.time()
                    scores, _ = compute_anomaly_scores(G, D, test_loader, args, device)
                    infer_time = time.time() - t1
                    np.savez(os.path.join(run_dir, f"{gan_name}_scores.npz"), scores=scores, labels=test_labels)
                    metrics = evaluate(scores, test_labels)
                    plot_scores(scores, test_labels, os.path.join(run_dir, f"{gan_name}_scores.png"), detector)

                    row = {
                        "dataset": args.dataset, "setting": args.setting, "gan_type": gan_name,
                        "model_family": "classical", "detector": detector,
                        "seed": seed, "id_class": id_cls, "ood_class": ood_str,
                        "scale": args.scale, "test_scale": args.test_scale,
                        "latent_dim": args.latent_dim, "hidden_g": args.hidden_g, "hidden_d": args.hidden_d,
                        "feature_dim": feature_dim, "epochs": args.epochs, "batch_size": args.batch_size,
                        "lr_g": args.lr_g, "lr_d": args.lr_d, "n_critic": args.n_critic,
                        "lambda_gp": args.lambda_gp, "z_iter": args.z_iter, "z_lr": args.z_lr,
                        "alpha": args.alpha, "grad_clip": args.grad_clip, "bce_eps": args.bce_eps,
                        "n_train": len(train_np), "n_val": len(val_np),
                        "n_test_id": len(test_id_np), "n_test_ood": len(test_ood_np),
                        "g_params": g_params, "d_params": d_params,
                        "train_time": round(train_time, 2), "best_epoch": best_epoch,
                        "best_val_residual": round(best_val, 6),
                        "final_g_loss": round(history["g_loss"][-1], 6),
                        "final_d_loss": round(history["d_loss"][-1], 6),
                        "infer_time": round(infer_time, 2),
                        "auc_roc": round(metrics["auc_roc"], 4), "auc_pr": round(metrics["auc_pr"], 4),
                        "f1_70": round(metrics["f1_70"], 4), "f1_80": round(metrics["f1_80"], 4),
                        "f1_90": round(metrics["f1_90"], 4), "f1_95": round(metrics["f1_95"], 4),
                        "f1_99": round(metrics["f1_99"], 4),
                    }
                    pd.DataFrame([row], columns=csv_columns).to_csv(csv_path, mode="a", header=False, index=False)
                    print(f"  {detector} (seed={seed}): AUROC={metrics['auc_roc']:.4f} AUPR={metrics['auc_pr']:.4f} "
                          f"| infer={infer_time:.1f}s | saved {run_dir}/")

        print(f"\n{'='*60}\nDone. CSV: {csv_path}\n{'='*60}")
        df = pd.read_csv(csv_path)
        # Only summarize the current run configuration (CSV is append-only)
        cur = df[(df["dataset"] == args.dataset) & (df["setting"] == args.setting)
                 & (df["scale"] == args.scale) & (df["test_scale"] == args.test_scale)
                 & (df["latent_dim"] == args.latent_dim)
                 & (df["hidden_g"] == args.hidden_g) & (df["hidden_d"] == args.hidden_d)]
        # Scheme 2: per ID class, mean/std ACROSS seeds
        per_class = cur.groupby(["gan_type", "id_class"]).agg(
            auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
            auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
            n_seeds=("seed", "nunique")).round(4).reset_index()
        # Overall: average each class over seeds first, then mean/std across classes
        ov = []
        for gan, g in per_class.groupby("gan_type"):
            ov.append({"gan_type": gan, "id_class": "Overall",
                       "auc_roc_mean": round(g["auc_roc_mean"].mean(), 4),
                       "auc_roc_std": round(g["auc_roc_mean"].std(), 4),
                       "auc_pr_mean": round(g["auc_pr_mean"].mean(), 4),
                       "auc_pr_std": round(g["auc_pr_mean"].std(), 4),
                       "n_seeds": len(g)})
        summary = pd.concat([per_class, pd.DataFrame(ov)], ignore_index=True)
        summary_path = os.path.join(args.save_dir, "summary_per_class.csv")
        summary.to_csv(summary_path, index=False)
        print("\n=== Per-class mean/std (across seeds) + Overall (across classes), Scheme 2 ===")
        print(summary.to_string(index=False))
        print(f"Summary saved: {summary_path}")


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
            args.output_dir, f"classical_gan_adversarial_{args.dataset}_normcls{args.norm_cls}_{setting_tag}_results.csv")
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
                G = ClassicalGenerator(ckpt_args["latent_dim"], ckpt_args["hidden_g"], feature_dim).to(device)
                G.load_state_dict(ckpt["G"])
                G.eval()
                if gan_name == "c_anogan":
                    D = ClassicalDiscriminator(feature_dim, ckpt_args["hidden_d"], ckpt_args.get("bce_eps", 0.0)).to(device)
                else:
                    D = ClassicalCritic(feature_dim).to(device)
                D.load_state_dict(ckpt["D"])
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
# GANomaly: classical GANomaly detector (C-GANomaly)
# ============================================================================

class GanomalyClassicalDiscriminator(nn.Module):
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


def val_latent_gap(G, val_loader, device):
    G.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)
            _, z1, z2 = G(x)
            tot += torch.abs(z1 - z2).sum(dim=1).mean().item() * x.shape[0]
            n += x.shape[0]
    return tot / max(n, 1)


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


def find_ganomaly_checkpoint(checkpoint_dir, norm_cls, seed):
    pattern = os.path.join(checkpoint_dir, f"*_id{norm_cls}_ood*_classical_*_seed{seed}")
    candidates = glob.glob(pattern)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 run dir matching '{pattern}', found {len(candidates)}: {candidates}. "
            f"Train it first via the ganomaly subcommand."
        )
    model_path = os.path.join(candidates[0], "classical_model.pt")
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
    print(f"Classical latent={args.classical_latent_dim}; loss weights: w_rec={args.w_rec}, w_lat={args.w_lat}, w_adv={args.w_adv}")
    csv_columns = [
        "dataset", "setting", "model", "seed", "id_class", "ood_class",
        "latent_dim", "feature_dim",
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
            model_name = "C-GANomaly"
            print(f"\n--- {model_name} ---")
            run_tag = (f"{args.dataset}_set{args.setting}_id{cls}_ood{ood_str}"
                       f"_classical_lat{args.classical_latent_dim}_seed{seed}")
            run_dir = os.path.join(args.save_dir, run_tag)
            os.makedirs(run_dir, exist_ok=True)
            G = ClassicalGANomalyG(args.classical_latent_dim, feature_dim).to(device)
            D = GanomalyClassicalDiscriminator(feature_dim, bce_eps=args.bce_eps).to(device)
            g_params, d_params = count_parameters(G), count_parameters(D)
            print(f"  G params: {g_params}, D params: {d_params}")
            t0 = time.time()
            history, best_val, best_epoch = train_ganomaly(
                G, D, train_loader, val_loader, args, device, model_name)
            train_time = time.time() - t0
            torch.save({"G_state_dict": G.state_dict(), "D_state_dict": D.state_dict(),
                        "args": vars(args), "history": history, "best_epoch": best_epoch},
                       os.path.join(run_dir, "classical_model.pt"))
            ganomaly_plot_losses(history, os.path.join(run_dir, "classical_loss.png"), model_name)
            plot_recon(G, test_id_np, test_ood_np, device,
                       os.path.join(run_dir, "classical_recon.png"))
            id_scores = ganomaly_scores(G, test_id_np, device)
            ood_scores = ganomaly_scores(G, test_ood_np, device)
            scores = np.concatenate([id_scores, ood_scores])
            metrics = evaluate(scores, test_y)
            np.savez(os.path.join(run_dir, "classical_scores.npz"), scores=scores, labels=test_y)
            ganomaly_plot_score_dist(scores, test_y, os.path.join(run_dir, "classical_scores.png"), model_name)
            row = {
                "dataset": args.dataset, "setting": args.setting, "model": model_name,
                "seed": seed, "id_class": cls, "ood_class": ood_str,
                "latent_dim": args.classical_latent_dim, "feature_dim": feature_dim,
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
    df = pd.read_csv(csv_path)
    cur = df[(df["dataset"] == args.dataset) & (df["setting"] == args.setting)
             & (df["scale"] == args.scale) & (df["test_scale"] == args.test_scale)
             & (df["w_rec"] == args.w_rec) & (df["w_lat"] == args.w_lat) & (df["w_adv"] == args.w_adv)]
    per_class = cur.groupby(["model", "id_class"]).agg(
        auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
        n_seeds=("seed", "nunique")).round(4).reset_index()
    summary_path = os.path.join(args.save_dir, "summary_per_class.csv")
    per_class.to_csv(summary_path, index=False)
    print(f"\n{'='*70}\nDone. CSV: {csv_path}\nSummary: {summary_path}\n{'='*70}")
    print(per_class.to_string(index=False))


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

    test_imgs, test_labels, composition = load_adversarial_test_set(args.dataset, args.norm_cls, args.adversarial_dir,
                                                                       args.attack)
    print(f"test={len(test_imgs)} (ID={int((test_labels == 0).sum())}, adversarial-OOD={int((test_labels == 1).sum())})")
    print(f"composition: {composition}")
    composition_cols = {"n_id": composition["n_id"], "n_fgsm": composition["fgsm"], "n_pgd": composition["pgd"],
                         "n_spsa": composition["spsa"], "n_salt_pepper": composition["salt_pepper"]}

    test_imgs_norm = normalize_images(test_imgs).reshape(len(test_imgs), -1)
    feature_dim = args.img_size * args.img_size

    all_rows = []
    for seed in args.seeds:
        checkpoint_path = find_ganomaly_checkpoint(args.checkpoint_dir, args.norm_cls, seed)
        print(f"Loading FIXED C-GANomaly checkpoint (seed={seed}) from '{checkpoint_path}' -- never retrained")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt["args"]
        G = ClassicalGANomalyG(ckpt_args["classical_latent_dim"], feature_dim).to(device)
        G.load_state_dict(ckpt["G_state_dict"])
        G.eval()
        scores = ganomaly_scores(G, test_imgs_norm, device)
        auc_roc = roc_auc_score(test_labels, scores)
        auc_pr = average_precision_score(test_labels, scores)
        print(f"C-GANomaly (seed={seed})  AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}")
        all_rows.append({"dataset": args.dataset, "norm_cls": args.norm_cls, "eval_setting": setting_tag,
                         "seed": seed, "detector": "C-GANomaly", "hyperparam": "",
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



def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="detector_family", required=True)

    sub = subparsers.add_parser("sae", help="Classical SAE-based distance detectors (SAE-Recon, DeepKNN, Deep-Mean, Deep-Medoids, DeepSVDD). Natural-shift training+eval by default; pass --adversarial_dir to evaluate an already-trained --checkpoint_dir/--checkpoint on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--norm_cls", type=int, default=0)
    sub.add_argument("--ood_cls", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--train_data_scale", type=float, default=0.2,
                         help="data_scale for the training pool AND the fitting/val split (same convention "
                              "as run_quanforge_pipeline_mnist.py): int(scale*len(train class images))")
    sub.add_argument("--test_data_scale", type=float, default=1.0,
                         help="data_scale for the FINAL reported test set only (separate call to "
                              "load_ood_split_scaled_raw, same --seed so val stays identical/disjoint "
                              "between the two calls)")
    sub.add_argument("--n_val_per_cls", type=int, default=20)
    sub.add_argument("--img_size", type=int, default=16, help="16x16 matches this project's quantum circuits")
    sub.add_argument("--latent_dim", type=int, default=32)
    sub.add_argument("--epochs", type=int, default=30)
    sub.add_argument("--batch_size", type=int, default=32)
    sub.add_argument("--lr", type=float, default=1e-3)
    sub.add_argument("--k_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10],
                         help="DeepKNN K grid, tuned on the validation split via AUC-ROC")
    sub.add_argument("--m_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10],
                         help="Deep-Medoids M grid, tuned on the validation split via AUC-ROC")
    sub.add_argument("--svdd_epochs", type=int, default=10)
    sub.add_argument("--svdd_lr", type=float, default=1e-3)
    sub.add_argument("--svdd_lambda", type=float, default=1e-3, help="weight on R^2 in the DeepSVDD loss")
    sub.add_argument("--svdd_checkpoint", type=str, default=None)
    sub.add_argument("--svdd_force_retrain", action="store_true")
    sub.add_argument("--detectors", nargs="+", type=str,
                         default=["SAE-Recon", "DeepKNN", "Deep-Mean", "Deep-Medoids", "DeepSVDD"],
                         choices=["SAE-Recon", "DeepKNN", "Deep-Mean", "Deep-Medoids", "DeepSVDD"])
    sub.add_argument("--save_interval", type=int, default=25,
                         help="save a checkpoint + loss curve + reconstruction snapshot every this many epochs")
    sub.add_argument("--resume", action="store_true")
    sub.add_argument("--checkpoint", type=str, default=None)
    sub.add_argument("--output_dir", type=str, default="outputs/classical_sae")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/classical_sae_<dataset>_normcls<norm_cls>_results.csv")
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--checkpoint_dir", type=str, default="outputs/classical_sae",
                         help="where the sae subcommand's checkpoints live")

    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")

    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                         help="SAE-Recon is fully deterministic at inference (plain feed-forward "
                              "reconstruction, no stochastic sampling) -- kept only so the results CSV has "
                              "one row per seed for a fair mean-over-5-seeds comparison")
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    sub = subparsers.add_parser("vae", help="Classical VAE reconstruction detector. Natural-shift training+eval by default; pass --adversarial_dir and --original_results_csv to evaluate an already-trained --checkpoint_dir/--checkpoint on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--norm_cls", type=int, default=0)
    sub.add_argument("--ood_cls", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--train_data_scale", type=float, default=0.2,
                         help="data_scale for the training pool AND the fitting/val split")
    sub.add_argument("--test_data_scale", type=float, default=1.0,
                         help="data_scale for the FINAL reported test set only (separate call to "
                              "load_ood_split_scaled_raw, same --seed so val stays identical/disjoint "
                              "between the two calls)")
    sub.add_argument("--n_val_per_cls", type=int, default=20)
    sub.add_argument("--img_size", type=int, default=16, help="16x16 matches this project's quantum circuits")
    sub.add_argument("--latent_dim", type=int, default=32)
    sub.add_argument("--beta", type=float, default=1.0, help="weight on the KL term")
    sub.add_argument("--epochs", type=int, default=30)
    sub.add_argument("--batch_size", type=int, default=32)
    sub.add_argument("--lr", type=float, default=1e-3)
    sub.add_argument("--save_interval", type=int, default=25)
    sub.add_argument("--resume", action="store_true")
    sub.add_argument("--checkpoint", type=str, default=None)
    sub.add_argument("--output_dir", type=str, default="outputs/classical_vae")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/classical_vae_<dataset>_normcls<norm_cls>_results.csv")
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--checkpoint_dir", type=str, default="outputs/classical_vae",
                         help="where the vae subcommand's checkpoints live")

    sub.add_argument("--original_results_csv", type=str, default=None,
                         help="the ORIGINAL clean-OOD run's results CSV to read beta from, e.g. "
                              "outputs/Main-results/classical_vae/classical_vae_<dataset>_normcls<norm_cls>_"
                              "results.csv")
    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")

    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                         help="each seed reproduces its OWN reparameterization draw, "
                              "the only source of seed-to-seed variance here since the checkpoint is frozen")
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    sub = subparsers.add_parser("density", help="Classical density-based detectors (DMKDE-mixed, IndepGaussian, MVGaussian) on a frozen pretrained backbone. Natural-shift sweep by default; pass --adversarial_dir to evaluate on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--norm_cls", type=int, default=0)
    sub.add_argument("--ood_cls", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--train_data_scale", type=float, default=0.2)
    sub.add_argument("--test_data_scale", type=float, default=1.0)
    sub.add_argument("--n_val_per_cls", type=int, default=20)
    sub.add_argument("--img_size", type=int, default=16)
    sub.add_argument("--backbone_img_size", type=int, default=32)
    sub.add_argument("--backbone", type=str, default="resnet18",
                         choices=["resnet18", "resnet34", "vgg16", "mobilenet_v2"])
    sub.add_argument("--detectors", nargs="+", type=str, default=DENSITY_ALL_DETECTORS, choices=DENSITY_ALL_DETECTORS)
    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                         help="each seed reloads its own val/test split")
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    sub.add_argument("--output_dir", type=str, default="outputs/classical_density")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/classical_density_<dataset>_normcls<norm_cls>_results.csv")

    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")

    sub = subparsers.add_parser("distance", help="Distance-based detectors (DeepKNN, Deep-Mean, Deep-Medoids, DeepSVDD) on a native pretrained backbone or a 6D learned projection (--settings native|6d|both). Natural-shift sweep by default; pass --adversarial_dir and --original_results_csv to evaluate on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--norm_cls", type=int, default=0)
    sub.add_argument("--ood_cls", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--train_data_scale", type=float, default=0.2,
                         help="data_scale for the training pool AND the fitting/val split")
    sub.add_argument("--test_data_scale", type=float, default=1.0,
                         help="data_scale for the FINAL reported test set only (separate call to "
                              "load_ood_split_scaled_raw, same seed so val stays identical/disjoint "
                              "between the two calls)")
    sub.add_argument("--n_val_per_cls", type=int, default=20)
    sub.add_argument("--img_size", type=int, default=16, help="raw pixel size, matches this project's quantum circuits")
    sub.add_argument("--backbone_img_size", type=int, default=32)
    sub.add_argument("--proj_dim", type=int, default=6, help="learned embedding dimensionality (the '6D' setting)")
    sub.add_argument("--settings", type=str, default="both", choices=["native", "6d", "both"],
                         help="which of the two final-comparison settings to run. 'native' skips projector "
                              "training entirely (and the ablation phase, since it only exists to pick the "
                              "projector's loss) -- use it for a quick native-dim-only sweep across backbones/"
                              "classes/seeds before committing to the full 6D comparison.")
    sub.add_argument("--skip_ablation", action="store_true",
                         help="skip phase 1 and use --best_loss directly for the final comparison")
    sub.add_argument("--best_loss", type=str, default=None, choices=ALL_LOSSES,
                         help="required if --skip_ablation is set")
    sub.add_argument("--ablation_losses", nargs="+", type=str, default=ALL_LOSSES, choices=ALL_LOSSES)
    sub.add_argument("--ablation_detectors", nargs="+", type=str, default=ALL_DETECTORS, choices=ALL_DETECTORS)
    sub.add_argument("--ablation_epochs", type=int, default=10)
    sub.add_argument("--ablation_lr", type=float, default=1e-2)
    sub.add_argument("--ablation_batch_size", type=int, default=32)
    sub.add_argument("--ablation_seed", type=int, default=0)
    sub.add_argument("--ablation_k", type=int, default=5, help="fixed K for DeepKNN during ablation (no tuning)")
    sub.add_argument("--ablation_m", type=int, default=5, help="fixed M for Deep-Medoids during ablation (no tuning)")
    sub.add_argument("--final_backbones", nargs="+", type=str,
                         default=["resnet18", "mobilenet_v2", "vgg16"],
                         choices=["resnet18", "resnet34", "mobilenet_v2", "vgg16"])
    sub.add_argument("--final_epochs", type=int, default=10)
    sub.add_argument("--final_lr", type=float, default=1e-2)
    sub.add_argument("--final_batch_size", type=int, default=32)
    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    sub.add_argument("--detectors", nargs="+", type=str, default=ALL_DETECTORS, choices=ALL_DETECTORS)
    sub.add_argument("--k_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    sub.add_argument("--m_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    sub.add_argument("--aug_max_rotate", type=float, default=15.0)
    sub.add_argument("--aug_max_translate", type=int, default=2)
    sub.add_argument("--aug_noise_std", type=float, default=0.03)
    sub.add_argument("--vicreg_lambda_inv", type=float, default=25.0)
    sub.add_argument("--vicreg_lambda_var", type=float, default=25.0)
    sub.add_argument("--vicreg_lambda_cov", type=float, default=1.0)
    sub.add_argument("--vicreg_gamma", type=float, default=1.0)
    sub.add_argument("--svdd_proj_dim", type=int, default=6)
    sub.add_argument("--svdd_epochs", type=int, default=10)
    sub.add_argument("--svdd_lr", type=float, default=1e-3)
    sub.add_argument("--svdd_lambda", type=float, default=1e-3)
    sub.add_argument("--svdd_batch_size", type=int, default=32)

    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                         help="'auto' (default) uses CUDA if torch.cuda.is_available() else CPU, silently. "
                              "'cuda' forces GPU and raises an error immediately if none is visible, instead "
                              "of silently falling back to CPU.")
    sub.add_argument("--output_dir", type=str, default="outputs/classical_projection")
    sub.add_argument("--ablation_csv", type=str, default=None,
                         help="defaults to <output_dir>/ablation_results.csv")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/final_results.csv")
    sub.add_argument("--backbone", type=str, default="resnet18",
                         choices=["resnet18", "resnet34", "vgg16", "mobilenet_v2"])

    sub.add_argument("--original_results_csv", type=str, default=None,
                         help="the ORIGINAL clean-OOD run's results CSV to read already-selected K/M from, "
                              "e.g. outputs/Main-results/classical_distance/classical_native_"
                              "<dataset>_normcls<norm_cls>_results.csv")
    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")

    sub = subparsers.add_parser("gan", help="Classical GAN-based detectors (Classical-AnoGAN / Classical-WGAN-GP). Natural-shift training+eval by default; pass --adversarial_dir to evaluate an already-trained --checkpoint_dir on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--target_class", type=int, default=0)
    sub.add_argument("--ood_class", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--scale", type=float, default=0.2, help="Training data fraction")
    sub.add_argument("--test_scale", type=float, default=1.0, help="ID test fraction")
    sub.add_argument("--img_size", type=int, default=16)
    sub.add_argument("--gan_type", type=str, default="both", choices=["c_anogan", "cwgan_gp", "both"])
    sub.add_argument("--epochs", type=int, default=100)
    sub.add_argument("--batch_size", type=int, default=64)
    sub.add_argument("--lr_g", type=float, default=1e-3)
    sub.add_argument("--lr_d", type=float, default=1e-3)
    sub.add_argument("--n_critic", type=int, default=5)
    sub.add_argument("--lambda_gp", type=float, default=10.0)
    sub.add_argument("--latent_dim", type=int, default=6,
                   help="Generator noise dim (6 to match quantum 6-qubit; raise for classical advantage)")
    sub.add_argument("--hidden_g", type=int, default=128)
    sub.add_argument("--hidden_d", type=int, default=128)
    sub.add_argument("--z_iter", type=int, default=500)
    sub.add_argument("--z_lr", type=float, default=1e-2)
    sub.add_argument("--alpha", type=float, default=0.5)
    sub.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    sub.add_argument("--run_all_id", action="store_true")
    sub.add_argument("--save_dir", type=str, default="results/classical_gan_mnist")
    sub.add_argument("--grad_clip", type=float, default=0.0)
    sub.add_argument("--bce_eps", type=float, default=0.0)
    sub.add_argument("--fast_test", action="store_true")
    sub.add_argument("--norm_cls", type=int, default=0)

    sub.add_argument("--checkpoint_dir", type=str, default=None,
                         help="directory holding canogan_ood.py's per-run dirs for this dataset, e.g. "
                              "outputs/classical_gan_mnist")

    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")

    sub.add_argument("--gan_types", nargs="+", type=str, default=["c_anogan", "cwgan_gp"],
                         choices=["c_anogan", "cwgan_gp"])
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    sub.add_argument("--output_dir", type=str, default="outputs/adversarial_eval")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/classical_gan_adversarial_<dataset>_"
                              "normcls<norm_cls>_<attack>_results.csv")

    sub = subparsers.add_parser("ganomaly", help="Classical GANomaly detector (C-GANomaly). Natural-shift training+eval by default; pass --adversarial_dir to evaluate an already-trained --checkpoint_dir on the adversarial OOD split instead.")
    sub.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    sub.add_argument("--data_dir", type=str, default="./data")
    sub.add_argument("--target_class", type=int, default=0)
    sub.add_argument("--ood_class", type=int, default=1)
    sub.add_argument("--setting", type=int, default=1, choices=[1, 2])
    sub.add_argument("--run_all_id", action="store_true")
    sub.add_argument("--scale", type=float, default=0.2)
    sub.add_argument("--test_scale", type=float, default=1.0)
    sub.add_argument("--img_size", type=int, default=16)
    sub.add_argument("--classical_latent_dim", type=int, default=32)
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
    sub.add_argument("--checkpoint_dir", type=str, default=None,
                         help="directory holding this subcommand's per-run dirs for this dataset")
    sub.add_argument("--adversarial_dir", type=str, default=None)
    sub.add_argument("--attack", type=str, default=None, choices=ATTACK_ORDER,
                         help="required when --adversarial_dir is given")
    sub.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    sub.add_argument("--output_dir", type=str, default="outputs/adversarial_eval")
    sub.add_argument("--results_csv", type=str, default=None,
                         help="defaults to <output_dir>/ganomaly_adversarial_<dataset>_"
                              "normcls<norm_cls>_<attack>_results.csv")

    return parser


def main():
    args = build_parser().parse_args()
    dispatch = {
        "sae": run_sae,
        "vae": run_vae,
        "density": run_density,
        "distance": run_distance,
        "gan": run_gan,
        "ganomaly": run_ganomaly,
    }
    dispatch[args.detector_family](args)


if __name__ == "__main__":
    main()
