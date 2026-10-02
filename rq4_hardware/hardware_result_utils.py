"""Shared sample-selection and result-saving utilities for the hardware evaluation driver."""
import os
from datetime import datetime, timezone

import numpy as np


def select_fixed_test_samples(dataset, data_dir, id_class, test_scale=0.1, seed=42, target_n_id=100):
    """Deterministic ID+OOD raw dataset index selection, independent of any
    detector family's own resize/normalize preprocessing -- every detector
    family applies its OWN preprocessing to these SAME underlying test-set
    images, so results stay comparable across detectors and backends.

    Returns a dict: dataset, id_class, test_scale, seed, indices (int
    array, ID first then OOD), labels (0=ID/1=OOD, same order), source_class
    (int array, same order -- id_class for ID rows, the actual OOD class for
    OOD rows), n_id, n_ood.
    """
    from torchvision import datasets as tv_datasets

    if dataset == "mnist":
        test_set = tv_datasets.MNIST(data_dir, train=False, download=True)
    elif dataset == "fashion_mnist":
        test_set = tv_datasets.FashionMNIST(data_dir, train=False, download=True)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    targets = test_set.targets.numpy() if hasattr(test_set.targets, "numpy") else np.asarray(test_set.targets)

    rng = np.random.RandomState(seed)

    id_idx_all = np.where(targets == id_class)[0]
    id_idx_all = id_idx_all[:int(test_scale * len(id_idx_all))]
    n_id = min(target_n_id, len(id_idx_all))
    if n_id < target_n_id:
        print(f"WARNING: only {n_id} ID samples available for {dataset}/class{id_class}/"
              f"test_scale={test_scale} (wanted {target_n_id})")
    id_idx = np.sort(rng.choice(id_idx_all, size=n_id, replace=False))

    ood_classes = [c for c in range(10) if c != id_class]
    k = len(ood_classes)
    base, rem = divmod(n_id, k)
    extra = set(rng.choice(ood_classes, size=rem, replace=False).tolist()) if rem > 0 else set()
    ood_idx_parts, ood_source_parts = [], []
    for c in ood_classes:
        c_idx_all = np.where(targets == c)[0]
        c_idx_all = c_idx_all[:int(test_scale * len(c_idx_all))]
        n_c = min(base + (1 if c in extra else 0), len(c_idx_all))
        if n_c == 0:
            continue
        chosen = rng.choice(c_idx_all, size=n_c, replace=False)
        ood_idx_parts.append(chosen)
        ood_source_parts.append(np.full(n_c, c))
    ood_idx = np.concatenate(ood_idx_parts) if ood_idx_parts else np.array([], dtype=np.int64)
    ood_source = np.concatenate(ood_source_parts) if ood_source_parts else np.array([], dtype=np.int64)
    n_ood = len(ood_idx)
    if n_ood != n_id:
        print(f"WARNING: n_ood={n_ood} != n_id={n_id} for {dataset}/class{id_class}/test_scale={test_scale} "
              f"(exact 50/50 balance not achievable at this test_scale)")

    indices = np.concatenate([id_idx, ood_idx])
    labels = np.concatenate([np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)])
    source_class = np.concatenate([np.full(n_id, id_class), ood_source])

    return {
        "dataset": dataset, "id_class": id_class, "test_scale": test_scale, "seed": seed,
        "indices": indices, "labels": labels, "source_class": source_class,
        "n_id": int(n_id), "n_ood": int(n_ood),
    }


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def run_dir_for(output_dir, dataset, backend, detector):
    d = os.path.join(output_dir, "runs", dataset, backend, detector)
    os.makedirs(os.path.join(d, "raw_results"), exist_ok=True)
    return d
