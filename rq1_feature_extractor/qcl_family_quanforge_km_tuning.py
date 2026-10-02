"""k/m hyperparameter tuning for the quantum feature-extractor distance detectors."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main_experiments"))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import datasets, transforms

from scoring_pipelines import load_ood_split_scaled  # noqa: E402
from qcl_family_quanforge_ood import (QuantumFeatureExtractor, kmedoids_cosine,
                                       knn_scores, medoid_scores)


def select_k(train_bank, val_embs, val_labels, k_candidates):
    """K selected via the small ID+OOD validation split (disjoint from
    train_pool and test), scored with plain knn_scores (val points are not
    in the memory bank, so no leave-one-out self-match issue) and ranked by
    AUC-ROC."""
    records = {}
    best_k, best_auc = None, -np.inf
    for k in k_candidates:
        scores = knn_scores(train_bank, val_embs, k)
        auc = roc_auc_score(val_labels, scores)
        records[k] = round(auc, 4)
        if auc > best_auc:
            best_k, best_auc = k, auc
    print(f"QKNN K selection (validation AUC-ROC per K): {records} -> chose K={best_k}")
    return best_k, records


def select_m(train_bank, val_embs, val_labels, m_candidates, seed=0):
    """Same rationale as select_k, for QMedoids' number of medoids M."""
    records = {}
    best_m, best_auc = None, -np.inf
    for m in m_candidates:
        scores = medoid_scores(train_bank, val_embs, m, seed=seed)
        auc = roc_auc_score(val_labels, scores)
        records[m] = round(auc, 4)
        if auc > best_auc:
            best_m, best_auc = m, auc
    print(f"QMedoids M selection (validation AUC-ROC per M): {records} -> chose M={best_m}")
    return best_m, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--norm_cls", type=int, default=0)
    parser.add_argument("--ood_cls", type=int, default=1)
    parser.add_argument("--setting", type=int, default=1, choices=[1, 2])
    parser.add_argument("--train_data_scale", type=float, default=0.2,
                         help="MUST match whatever --data_scale the checkpoint was trained with, so the "
                              "memory bank/medoids are built on the exact same train_pool")
    parser.add_argument("--test_data_scale", type=float, default=1.0,
                         help="data_scale for the FINAL reported test set only (separate call to "
                              "load_ood_split_scaled, same --seed so val stays identical/disjoint between "
                              "the two calls)")
    parser.add_argument("--n_val_per_cls", type=int, default=20)
    parser.add_argument("--num_latent", type=int, default=6)
    parser.add_argument("--num_trash", type=int, default=2)
    parser.add_argument("--circuit", type=str, required=True, choices=["QCL", "QCNN", "HCQC", "DRNN"])
    parser.add_argument("--readout", type=str, default="probs", choices=["probs", "expval"])
    parser.add_argument("--hcqc_unitary", type=str, default="U_SU4",
                         choices=["U_TTN", "U_5", "U_6", "U_9", "U_13", "U_14", "U_15", "U_SO4", "U_SU4"])
    parser.add_argument("--drnn_ent_train", action="store_true")
    parser.add_argument("--drnn_scaling", type=float, default=1.5)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="path to a qcl_family_quanforge_ood.py checkpoint (any --feature_loss); "
                              "--circuit/--readout/--hcqc_unitary/--drnn_ent_train/--drnn_scaling must "
                              "match how it was trained, for the state_dict shapes to line up")
    parser.add_argument("--k_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    parser.add_argument("--m_candidates", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    parser.add_argument("--seed", type=int, default=0, help="QMedoids' k-medoids init seed")
    args = parser.parse_args()

    extractor = QuantumFeatureExtractor.from_checkpoint(
        args.checkpoint, args.circuit, args.readout, args.hcqc_unitary, args.drnn_ent_train, args.drnn_scaling
    )
    print(f"Loaded {args.circuit} extractor from '{args.checkpoint}'")

    # Same --data_scale -> n_train derivation as qcl_family_quanforge_ood.py
    # (int(scale * len(class_idx))), so passing the same --data_scale/
    # --dataset/--norm_cls reproduces the exact train_pool the checkpoint
    # was trained on.
    if args.dataset == "mnist":
        _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    else:
        _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
    n_train = int(args.train_data_scale * _class_count)
    print(f"train_data_scale={args.train_data_scale} -> n_train={n_train} "
          f"(of {_class_count} available {args.norm_cls}-class training images)")

    data_num_latent, data_num_trash = args.num_latent, args.num_trash

    # train_pool/val at train_data_scale; test at test_data_scale -- same
    # seed so val is identical/disjoint between the two calls.
    train_pool_imgs, val_imgs, val_labels, _unused_test_imgs, _unused_test_labels, img_shape, feat_dim, ood_str = \
        load_ood_split_scaled(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                               n_train, args.n_val_per_cls, args.train_data_scale, data_num_latent,
                               data_num_trash, seed=args.seed)
    _unused_train_pool, _unused_val_imgs, _unused_val_labels, test_imgs, test_labels, _, _, _ = \
        load_ood_split_scaled(args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
                               n_train, args.n_val_per_cls, args.test_data_scale, data_num_latent,
                               data_num_trash, seed=args.seed)

    with torch.no_grad():
        train_embs = extractor.get_embedding(torch.as_tensor(train_pool_imgs, dtype=torch.float64)).numpy()
        val_embs = extractor.get_embedding(torch.as_tensor(val_imgs, dtype=torch.float64)).numpy()
        test_embs = extractor.get_embedding(torch.as_tensor(test_imgs, dtype=torch.float64)).numpy()

    best_k, k_records = select_k(train_embs, val_embs, val_labels, args.k_candidates)
    best_m, m_records = select_m(train_embs, val_embs, val_labels, args.m_candidates, seed=args.seed)

    knn_test_s = knn_scores(train_embs, test_embs, best_k)
    med_test_s = medoid_scores(train_embs, test_embs, best_m, seed=args.seed)

    print(f"\n{'Circuit':<10} {'Detector':<10} {'Best HP':<10} {'Test AUC-ROC':<14} {'Test AUC-PR':<14}")
    print(f"{args.circuit:<10} {'QKNN':<10} {'K='+str(best_k):<10} "
          f"{roc_auc_score(test_labels, knn_test_s):<14.4f} {average_precision_score(test_labels, knn_test_s):<14.4f}")
    print(f"{args.circuit:<10} {'QMedoids':<10} {'M='+str(best_m):<10} "
          f"{roc_auc_score(test_labels, med_test_s):<14.4f} {average_precision_score(test_labels, med_test_s):<14.4f}")
