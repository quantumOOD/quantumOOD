"""Shared data loading, feature extraction, and evaluation utilities for the quantum and classical OOD detectors."""
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit_machine_learning.circuit.library import raw_feature_vector
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

# ============================================================================
# Quantum autoencoder feature extraction and amplitude-encoding preprocessing
# ============================================================================

def ansatz(num_qubits, reps=5):
    return real_amplitudes(num_qubits, reps=reps)


def auto_encoder_circuit(num_latent, num_trash):
    """Training circuit: ansatz(num_latent+num_trash), then a SWAP test between
    the trash qubits and freshly-initialized reference qubits, read out through
    one auxiliary qubit. Reused unmodified by ReconstructionScorer below."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    auxiliary_qubit = num_latent + 2 * num_trash
    circuit.h(auxiliary_qubit)
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary_qubit)
    circuit.measure(auxiliary_qubit, cr[0])
    return circuit


def feature_dim_to_img_shape(num_latent, num_trash):
    """Square image shape whose pixel count equals 2**(num_latent+num_trash)."""
    feat_dim = 2 ** (num_latent + num_trash)
    img_size = int(round(np.sqrt(feat_dim)))
    assert img_size * img_size == feat_dim, (
        f"2**(num_latent+num_trash)={feat_dim} is not a perfect square -- "
        f"pick num_latent+num_trash even, or resize to a non-square shape manually."
    )
    return (img_size, img_size)


def normalize_images(images):
    # L2-normalize each flattened image so it is a valid amplitude-encoding input
    images = images.reshape(images.shape[0], -1).astype(np.float64)
    for i in range(len(images)):
        sum_sq = np.sum(images[i] ** 2)
        images[i] = images[i] / np.sqrt(sum_sq)
    return images


def resize_and_flatten(dataset, idx, img_shape):
    imgs = []
    for i in idx:
        img, _ = dataset[i]  # img: (1, 28, 28) tensor in [0, 1]
        img = TF.resize(img, list(img_shape), antialias=True)
        imgs.append(img.squeeze(0).numpy())
    return np.stack(imgs, axis=0)


def prepare_images(dataset, idx, num_latent, num_trash):
    """resize_and_flatten + normalize_images using the img_shape implied by
    (num_latent, num_trash)."""
    img_shape = feature_dim_to_img_shape(num_latent, num_trash)
    images = resize_and_flatten(dataset, idx, img_shape)
    return normalize_images(images)


def load_real_dataset(name, dir, target_class, scale=0.2, img_shape=(8, 4), draw=True):
    """ID-class-only train/val/test image loader for the quantum reconstruction
    detectors' training stage (no OOD data involved at training time)."""
    if name == 'mnist':
        train_set = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST(dir, train=False, download=True, transform=transforms.ToTensor())
    elif name == 'fashion_mnist':
        train_set = datasets.FashionMNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.FashionMNIST(dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f'unknown dataset {name}')

    train_targets = train_set.targets
    test_targets = test_set.targets

    train_class_idx = torch.where(train_targets == target_class)[0]
    test_class_idx = torch.where(test_targets == target_class)[0]

    train_pool_size = int(scale * len(train_class_idx))
    train_pool_idx = train_class_idx[:train_pool_size]

    train_size = int(0.8 * train_pool_size)
    train_idx = train_pool_idx[:train_size]
    val_idx = train_pool_idx[train_size:]

    test_size = int(scale * len(test_class_idx))
    test_idx = test_class_idx[:test_size]

    print(f'load {name} data for class {target_class} only, scale: {scale * 100}%, '
          f'train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}')

    train_images = resize_and_flatten(train_set, train_idx, img_shape)
    val_images = resize_and_flatten(train_set, val_idx, img_shape)
    test_images = resize_and_flatten(test_set, test_idx, img_shape)

    if draw:
        plt.title(f'{name} class {target_class} (resized {img_shape[0]}x{img_shape[1]})')
        plt.imshow(train_images[0], cmap="gray")
        plt.savefig(f'sample_{name}_{target_class}.png')

    train_images = normalize_images(train_images)
    val_images = normalize_images(val_images)
    test_images = normalize_images(test_images)

    return train_images, val_images, test_images


class LatentFeatureExtractor:
    """Frozen quantum encoder-only inference, used by the QKNN/QMean/QMedoids
    distance detectors (no fine-tuning)."""

    def __init__(self, num_latent, num_trash, weights):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.weights = np.asarray(weights)

        self.fm = raw_feature_vector(2 ** (num_latent + num_trash))
        self.ansatz_qc = ansatz(num_latent + num_trash)

        circuit = QuantumCircuit(num_latent + num_trash)
        circuit = circuit.compose(self.fm)
        circuit = circuit.compose(self.ansatz_qc)
        self.circuit = circuit

    @classmethod
    def from_checkpoint(cls, checkpoint_path, num_latent, num_trash):
        ckpt = np.load(checkpoint_path)
        return cls(num_latent, num_trash, ckpt['weights'])

    @classmethod
    def from_weights_file(cls, weights_path, num_latent, num_trash):
        return cls(num_latent, num_trash, np.load(weights_path))

    def get_latent_embedding(self, image, weights=None):
        """Encoder-only inference: fm(image) -> trained ansatz -> <Z> on the
        latent qubits. Returns a shape-(num_latent,) array."""
        if weights is None:
            weights = self.weights
        param_values = np.concatenate((image, weights))
        bound_qc = self.circuit.assign_parameters(param_values)
        sv = Statevector(bound_qc)
        return np.array([sv.expectation_value(Pauli('Z'), qargs=[q]).real for q in range(self.num_latent)])

    def get_embedding(self, images, weights=None):
        return np.stack([self.get_latent_embedding(image, weights) for image in images])


class TrainableLatentExtractor(torch.nn.Module):
    """Differentiable version of LatentFeatureExtractor, used by QSVDD so theta
    can be fine-tuned by gradient descent."""

    def __init__(self, num_latent, num_trash, initial_weights=None):
        super().__init__()
        self.num_latent = num_latent
        self.num_trash = num_trash

        fm = raw_feature_vector(2 ** (num_latent + num_trash))
        ansatz_qc = ansatz(num_latent + num_trash)
        circuit = QuantumCircuit(num_latent + num_trash)
        circuit = circuit.compose(fm)
        circuit = circuit.compose(ansatz_qc)

        observables = [
            SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=num_latent + num_trash)
            for q in range(num_latent)
        ]

        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=fm.parameters,
            weight_params=ansatz_qc.parameters,
            estimator=StatevectorEstimator(),
        )
        initial_weights = None if initial_weights is None else np.asarray(initial_weights)
        self.connector = TorchConnector(qnn, initial_weights=initial_weights)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, num_latent, num_trash):
        ckpt = np.load(checkpoint_path)
        return cls(num_latent, num_trash, initial_weights=ckpt['weights'])

    def forward(self, images):
        return self.connector(images).double()

    def get_embedding(self, images):
        with torch.no_grad():
            images_t = torch.as_tensor(np.asarray(images), dtype=torch.float64)
            return self.forward(images_t).numpy()


class ReconstructionScorer:
    """Reconstruction-based OOD scoring: s(x) = P(SWAP test measures 1), low
    for well-compressed (in-distribution) inputs."""

    def __init__(self, num_latent, num_trash, weights):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.weights = np.asarray(weights)

        fm = raw_feature_vector(2 ** (num_latent + num_trash))
        ae = auto_encoder_circuit(num_latent, num_trash)
        qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
        qc = qc.compose(fm, range(num_latent + num_trash))
        qc = qc.compose(ae)

        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=fm.parameters,
            weight_params=ae.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=StatevectorSampler(),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path, num_latent, num_trash):
        ckpt = np.load(checkpoint_path)
        return cls(num_latent, num_trash, ckpt['weights'])

    def score(self, images, weights=None):
        if weights is None:
            weights = self.weights
        probabilities = self.qnn.forward(np.asarray(images), weights)
        return probabilities[:, 1]


# ============================================================================
# Shared OOD split loading and evaluation (natural-shift, distance/density families)
# ============================================================================

RESULT_ROOT = "results/modified_ood_results"
CSV_COLUMNS = [
    "dataset", "setting", "detector", "seed", "id_class", "ood_class",
    "num_latent", "num_trash", "hyperparam", "infer_time",
    "auc_roc", "auc_pr", "f1_70", "f1_80", "f1_90", "f1_95", "f1_99"
]


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_ood_split(dataset, data_dir, norm_cls, ood_cls, setting,
                    n_train, n_val_per_cls, n_test_per_cls, num_latent, num_trash):
    """
    Three disjoint groups, shared by every detector script:
      - train_pool_imgs: ID-only. Fits the detector, and its own scores
        (leave-one-out where applicable) calibrate the F1/tau threshold.
      - val_imgs/val_labels: SMALL, ID+OOD, disjoint from train_pool and
        test_imgs. Used only for hyperparameter selection (e.g. QKNN's K,
        QMedoids' M) via AUC-ROC -- mimics real deployment where ID/OOD is
        unknown ahead of time. Detectors with no hyperparameter to tune
        (QMean, QSVDD, the reconstruction-based detector) can ignore it.
      - test_imgs/test_labels: ID+OOD, held out from both of the above.
        Used solely for the final reported AUC-ROC/AUC-PR/F1 metrics.
    val and test are both carved from the official test split, in disjoint
    index ranges (val filled first, then test, in dataset order).
    """
    img_shape = feature_dim_to_img_shape(num_latent, num_trash)
    feat_dim = 2 ** (num_latent + num_trash)

    if dataset == "mnist":
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

    train_targets = train_set.targets
    train_pool_idx = torch.where(train_targets == norm_cls)[0][:n_train].tolist()

    val_idx, val_labels = [], []
    test_idx, test_labels = [], []
    n_val_id, n_val_ood = 0, 0
    n_id, n_ood = 0, 0
    test_targets = test_set.targets
    for i in range(len(test_set)):
        label = int(test_targets[i])
        is_id = (label == norm_cls)
        if setting == 1:
            is_ood = (label != norm_cls)
        else:
            is_ood = (label == ood_cls)

        if is_id and n_val_id < n_val_per_cls:
            val_idx.append(i); val_labels.append(0); n_val_id += 1
        elif is_ood and n_val_ood < n_val_per_cls:
            val_idx.append(i); val_labels.append(1); n_val_ood += 1
        elif is_id and n_id < n_test_per_cls:
            test_idx.append(i); test_labels.append(0); n_id += 1
        elif is_ood and n_ood < n_test_per_cls:
            test_idx.append(i); test_labels.append(1); n_ood += 1

        if (n_val_id >= n_val_per_cls and n_val_ood >= n_val_per_cls
                and n_id >= n_test_per_cls and n_ood >= n_test_per_cls):
            break
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    train_pool_imgs = prepare_images(train_set, train_pool_idx, num_latent, num_trash)
    val_imgs = prepare_images(test_set, val_idx, num_latent, num_trash)
    test_imgs = prepare_images(test_set, test_idx, num_latent, num_trash)

    ood_str = f"all_except_{norm_cls}" if setting == 1 else str(ood_cls)
    print(f"Loaded: img={img_shape[0]}x{img_shape[1]}, num_latent={num_latent}, "
          f"num_trash={num_trash}, feature_dim={feat_dim}")
    print(f"Setting={setting}, ID={norm_cls}, OOD={ood_str}")
    print(f"Train pool (ID)={len(train_pool_idx)}, "
          f"Val ID={n_val_id}/OOD={n_val_ood}, Test ID={n_id}/OOD={n_ood}")
    return train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, img_shape, feat_dim, ood_str


def load_ood_split_scaled(dataset, data_dir, norm_cls, ood_cls, setting,
                           n_train, n_val_per_cls, data_scale, num_latent, num_trash, seed=0):
    """
    Alternate test-set construction to load_ood_split's fixed --n_test_per_cls,
    used by the RQ1 quantum feature-extractor scripts. Same train_pool/val
    contract as load_ood_split (val: n_val_per_cls ID + n_val_per_cls OOD,
    fixed small size). TEST is sized differently: the ID half = data_scale *
    (norm_cls's own test-split image count), e.g. data_scale=1.0 means "use
    ALL of norm_cls's test images" -- NOT a fraction of the full 10-class test
    split. The OOD half is set to exactly match the ID half's actual size, so
    the split is always exactly 50/50 regardless of data_scale.

    OOD is drawn evenly across all remaining classes for setting=1 (each
    contributing ~1/9 of the OOD target, randomly sampled within its own
    class), or from --ood_cls alone for setting=2. All sampling is randomized
    via a seeded RNG, unlike load_ood_split's deterministic dataset-order
    approach.
    """
    img_shape = feature_dim_to_img_shape(num_latent, num_trash)
    feat_dim = 2 ** (num_latent + num_trash)

    if dataset == "mnist":
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

    train_targets = train_set.targets
    train_pool_idx = torch.where(train_targets == norm_cls)[0][:n_train].tolist()

    test_targets = test_set.targets
    all_classes = sorted(set(int(c) for c in test_targets.tolist()))
    ood_classes = [c for c in all_classes if c != norm_cls] if setting == 1 else [ood_cls]

    rng = np.random.default_rng(seed)

    id_pool = torch.where(test_targets == norm_cls)[0].tolist()
    rng.shuffle(id_pool)
    ood_pools = {}
    for c in ood_classes:
        pool = torch.where(test_targets == c)[0].tolist()
        rng.shuffle(pool)
        ood_pools[c] = pool

    def take_evenly(pools, total, offsets):
        classes = list(pools.keys())
        n_classes = len(classes)
        base, remainder = divmod(total, n_classes)
        taken = []
        new_offsets = dict(offsets)
        for i, c in enumerate(classes):
            want = base + (1 if i < remainder else 0)
            start = offsets[c]
            chunk = pools[c][start:start + want]
            if len(chunk) < want:
                print(f"WARNING: OOD class {c} only has {len(chunk)} test images left "
                      f"(wanted {want}) -- using all of them")
            taken.extend(chunk)
            new_offsets[c] = start + len(chunk)
        return taken, new_offsets

    val_id_idx = id_pool[:n_val_per_cls]
    if len(val_id_idx) < n_val_per_cls:
        print(f"WARNING: norm_cls {norm_cls} only has {len(val_id_idx)} test images available for val "
              f"(wanted {n_val_per_cls})")
    id_offset = len(val_id_idx)
    ood_offsets = {c: 0 for c in ood_classes}
    val_ood_idx, ood_offsets = take_evenly(ood_pools, n_val_per_cls, ood_offsets)

    val_idx = val_id_idx + val_ood_idx
    val_labels = np.array([0] * len(val_id_idx) + [1] * len(val_ood_idx))

    n_test_target = int(data_scale * len(id_pool))  # per side (ID or OOD)
    available_id = len(id_pool) - id_offset
    n_test_id = min(n_test_target, available_id)
    if n_test_id < n_test_target:
        print(f"WARNING: norm_cls {norm_cls} only has {available_id} test images left after val "
              f"(wanted {n_test_target} for the ID half) -- capping BOTH halves to {n_test_id} "
              f"to keep the split exactly 50/50")
    test_id_idx = id_pool[id_offset: id_offset + n_test_id]

    test_ood_idx, ood_offsets = take_evenly(ood_pools, n_test_id, ood_offsets)

    test_idx = test_id_idx + test_ood_idx
    test_labels = np.array([0] * len(test_id_idx) + [1] * len(test_ood_idx))

    train_pool_imgs = prepare_images(train_set, train_pool_idx, num_latent, num_trash)
    val_imgs = prepare_images(test_set, val_idx, num_latent, num_trash)
    test_imgs = prepare_images(test_set, test_idx, num_latent, num_trash)

    ood_str = f"all_except_{norm_cls}" if setting == 1 else str(ood_cls)
    print(f"Loaded (scaled/balanced): img={img_shape[0]}x{img_shape[1]}, num_latent={num_latent}, "
          f"num_trash={num_trash}, feature_dim={feat_dim}")
    print(f"Setting={setting}, ID={norm_cls}, OOD={ood_str}")
    print(f"Train pool (ID)={len(train_pool_idx)}, "
          f"Val ID={len(val_id_idx)}/OOD={len(val_ood_idx)}, "
          f"Test ID={len(test_id_idx)}/OOD={len(test_ood_idx)} (target was {n_test_target}/side, "
          f"data_scale={data_scale} * {len(id_pool)} norm_cls={norm_cls} test images)")
    return train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, img_shape, feat_dim, ood_str


def evaluate(scores, labels, id_train_scores, method_name, infer_time=0.0):
    """
    AUC-ROC / AUC-PR are threshold-free, computed on the test set (ID+OOD).
    Threshold calibration (F1@70/80/90/95/99): thresholds are the
    70/80/90/95/99th percentiles of `id_train_scores` -- ID-only scores
    computed on the SAME ID training pool used to fit the detector (for
    QKNN/QMedoids, via leave-one-out scoring so a point can't trivially
    match itself). NEVER computed from the test set's own scores.
    """
    auc_roc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)
    threshs = np.quantile(id_train_scores, [0.7, 0.8, 0.9, 0.95, 0.99])
    f1s = [f1_score(labels, (scores > th).astype(int)) for th in threshs]
    res = {
        "auc_roc": auc_roc, "auc_pr": auc_pr,
        "f1_70": f1s[0], "f1_80": f1s[1], "f1_90": f1s[2], "f1_95": f1s[3], "f1_99": f1s[4]
    }
    print(f"{method_name} | AUC-ROC:{auc_roc:.4f} | AUC-PR:{auc_pr:.4f} | F1@95:{f1s[3]:.4f} | F1@99:{f1s[4]:.4f} | InferTime:{infer_time:.4f}s")
    return res


def ensure_result_csv():
    import pandas as pd
    os.makedirs(RESULT_ROOT, exist_ok=True)
    result_csv = os.path.join(RESULT_ROOT, "all_seed_results.csv")
    if not os.path.exists(result_csv):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(result_csv, index=False)
        print(f"Results saved: {result_csv}")
    return result_csv


# ============================================================================
# Classical CNN architecture, raw-pixel OOD split loading, and distance functions
# ============================================================================

def load_ood_split_scaled_raw(dataset, data_dir, norm_cls, ood_cls, setting,
                               n_train, n_val_per_cls, data_scale, img_size=16, seed=0):
    if dataset == "mnist":
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())

    train_targets = train_set.targets
    train_pool_idx = torch.where(train_targets == norm_cls)[0][:n_train].tolist()

    test_targets = test_set.targets
    all_classes = sorted(set(int(c) for c in test_targets.tolist()))
    ood_classes = [c for c in all_classes if c != norm_cls] if setting == 1 else [ood_cls]

    rng = np.random.default_rng(seed)
    id_pool = torch.where(test_targets == norm_cls)[0].tolist()
    rng.shuffle(id_pool)
    ood_pools = {}
    for c in ood_classes:
        pool = torch.where(test_targets == c)[0].tolist()
        rng.shuffle(pool)
        ood_pools[c] = pool

    def take_evenly(pools, total, offsets):
        classes = list(pools.keys())
        n_classes = len(classes)
        base, remainder = divmod(total, n_classes)
        taken = []
        new_offsets = dict(offsets)
        for i, c in enumerate(classes):
            want = base + (1 if i < remainder else 0)
            start = offsets[c]
            chunk = pools[c][start:start + want]
            if len(chunk) < want:
                print(f"WARNING: OOD class {c} only has {len(chunk)} test images left "
                      f"(wanted {want}) -- using all of them")
            taken.extend(chunk)
            new_offsets[c] = start + len(chunk)
        return taken, new_offsets

    val_id_idx = id_pool[:n_val_per_cls]
    if len(val_id_idx) < n_val_per_cls:
        print(f"WARNING: norm_cls {norm_cls} only has {len(val_id_idx)} test images available for val "
              f"(wanted {n_val_per_cls})")
    id_offset = len(val_id_idx)
    ood_offsets = {c: 0 for c in ood_classes}
    val_ood_idx, ood_offsets = take_evenly(ood_pools, n_val_per_cls, ood_offsets)

    val_idx = val_id_idx + val_ood_idx
    val_labels = np.array([0] * len(val_id_idx) + [1] * len(val_ood_idx))

    n_test_target = int(data_scale * len(id_pool))
    available_id = len(id_pool) - id_offset
    n_test_id = min(n_test_target, available_id)
    if n_test_id < n_test_target:
        print(f"WARNING: norm_cls {norm_cls} only has {available_id} test images left after val "
              f"(wanted {n_test_target} for the ID half) -- capping BOTH halves to {n_test_id} "
              f"to keep the split exactly 50/50")
    test_id_idx = id_pool[id_offset: id_offset + n_test_id]
    test_ood_idx, ood_offsets = take_evenly(ood_pools, n_test_id, ood_offsets)

    test_idx = test_id_idx + test_ood_idx
    test_labels = np.array([0] * len(test_id_idx) + [1] * len(test_ood_idx))

    def prep(dataset_obj, idx):
        imgs = []
        for i in idx:
            img, _ = dataset_obj[i]  # (1, 28, 28) in [0, 1]
            img = TF.resize(img, [img_size, img_size], antialias=True)
            imgs.append(img.numpy())
        return np.stack(imgs, axis=0).astype(np.float32) if imgs else np.zeros((0, 1, img_size, img_size), dtype=np.float32)

    train_pool_imgs = prep(train_set, train_pool_idx)
    val_imgs = prep(test_set, val_idx)
    test_imgs = prep(test_set, test_idx)

    ood_str = f"all_except_{norm_cls}" if setting == 1 else str(ood_cls)
    print(f"Loaded (raw pixels, scaled/balanced): img={img_size}x{img_size}")
    print(f"Setting={setting}, ID={norm_cls}, OOD={ood_str}")
    print(f"Train pool (ID)={len(train_pool_idx)}, Val ID={len(val_id_idx)}/OOD={len(val_ood_idx)}, "
          f"Test ID={len(test_id_idx)}/OOD={len(test_ood_idx)} (target was {n_test_target}/side, "
          f"data_scale={data_scale} * {len(id_pool)} norm_cls={norm_cls} test images)")
    return train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, ood_str


class SimpleCNNEncoder(nn.Module):
    """16x16 -> 8x8 -> 4x4 (32ch) -> flatten(512) -> FC(latent_dim)."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(32 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class SimpleCNNDecoder(nn.Module):
    """Mirror of SimpleCNNEncoder: latent_dim -> 4x4 (32ch) -> 8x8 -> 16x16, sigmoid output."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.reshape(h.size(0), 32, 4, 4)
        return self.deconv(h)


class Autoencoder(nn.Module):
    """SAE (Simple AutoEncoder) -- plain reconstruction, no variational
    machinery. Its encoder is reused (frozen) as the shared embedding for
    DeepKNN/Deep-Mean/Deep-Medoids."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = SimpleCNNEncoder(latent_dim)
        self.decoder = SimpleCNNDecoder(latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, latent_dim=32):
        model = cls(latent_dim)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


class VAE(nn.Module):
    """Same conv trunk as SimpleCNNEncoder, but two FC heads (mu, logvar)
    instead of one, and a reparameterized decode -- standard VAE."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(32 * 4 * 4, latent_dim)
        self.decoder = SimpleCNNDecoder(latent_dim)

    def encode(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @classmethod
    def from_checkpoint(cls, checkpoint_path, latent_dim=32):
        model = cls(latent_dim)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


def vae_loss_per_sample(recon_x, x, mu, logvar, beta=1.0):
    """MSE averaged over pixels (not summed) + beta * KL divergence, both per sample."""
    recon = torch.nn.functional.mse_loss(recon_x, x, reduction='none').reshape(x.size(0), -1).mean(dim=1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    per_sample = recon + beta * kl
    return per_sample, per_sample.mean()


def euclidean_dist_np(a, b):
    return float(np.linalg.norm(a - b))


def knn_scores_euclidean(train_bank, test_embs, k):
    scores = []
    for x in test_embs:
        dists = np.linalg.norm(train_bank - x, axis=-1)
        scores.append(np.mean(np.sort(dists)[:k]))
    return np.array(scores)


def mean_scores_euclidean(train_bank, test_embs):
    center = train_bank.mean(axis=0)
    return np.linalg.norm(test_embs - center, axis=-1)


def kmedoids_euclidean(embeddings, n_medoids, max_iter=50, seed=0):
    """Minimal PAM-style K-medoids using Euclidean distance."""
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    n_medoids = min(n_medoids, n)

    dist = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)

    medoid_idx = rng.choice(n, size=n_medoids, replace=False)
    for _ in range(max_iter):
        assign = np.argmin(dist[:, medoid_idx], axis=1)
        new_medoid_idx = medoid_idx.copy()
        changed = False
        for c in range(n_medoids):
            members = np.where(assign == c)[0]
            if len(members) == 0:
                continue
            sub_dist = dist[np.ix_(members, members)]
            best_local = members[np.argmin(sub_dist.sum(axis=1))]
            if best_local != medoid_idx[c]:
                new_medoid_idx[c] = best_local
                changed = True
        medoid_idx = new_medoid_idx
        if not changed:
            break
    return embeddings[medoid_idx], medoid_idx


def medoid_scores_euclidean(train_bank, test_embs, m, seed=0):
    medoids, _ = kmedoids_euclidean(train_bank, m, seed=seed)
    return np.array([min(np.linalg.norm(x - med) for med in medoids) for x in test_embs])


def select_k_euclidean(train_bank, val_embs, val_labels, k_candidates):
    records = {}
    best_k, best_auc = None, -np.inf
    for k in k_candidates:
        scores = knn_scores_euclidean(train_bank, val_embs, k)
        auc = roc_auc_score(val_labels, scores)
        records[k] = round(auc, 4)
        if auc > best_auc:
            best_k, best_auc = k, auc
    print(f"DeepKNN K selection (validation AUC-ROC per K): {records} -> chose K={best_k}")
    return best_k, records


def select_m_euclidean(train_bank, val_embs, val_labels, m_candidates, seed=0):
    records = {}
    best_m, best_auc = None, -np.inf
    for m in m_candidates:
        scores = medoid_scores_euclidean(train_bank, val_embs, m, seed=seed)
        auc = roc_auc_score(val_labels, scores)
        records[m] = round(auc, 4)
        if auc > best_auc:
            best_m, best_auc = m, auc
    print(f"Deep-Medoids M selection (validation AUC-ROC per M): {records} -> chose M={best_m}")
    return best_m, records


# ============================================================================
# Pretrained ImageNet / LeNet-5 backbones as frozen classical feature extractors
# ============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MNIST_NORM = {"mnist": (0.1307, 0.3081), "fashion_mnist": (0.2860, 0.3530)}

BACKBONE_FEATURE_DIM = {
    "resnet18": 512,
    "resnet34": 512,
    "vgg16": 4096,
    "mobilenet_v2": 1280,
    "lenet5": 84,
}


class LeNet5(nn.Module):
    """LeNet-5 (LeCun et al. 1998), for 32x32 grayscale input. Feature = the
    84-dim layer just before the final 10-class Linear."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def features(self, x):
        h = self.pool(torch.relu(self.conv1(x)))  # 32 -> 28 -> 14
        h = self.pool(torch.relu(self.conv2(h)))  # 14 -> 10 -> 5
        h = h.reshape(h.size(0), -1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return h  # (N, 84)

    def forward(self, x):
        return self.fc3(self.features(x))


def preprocess_for_lenet(imgs, dataset, img_size=32):
    imgs = TF.resize(imgs, [img_size, img_size], antialias=True)
    mean, std = MNIST_NORM[dataset]
    return (imgs - mean) / std


def pretrain_lenet_classifier(dataset, data_dir, img_size=32, epochs=5, lr=1e-3, batch_size=128,
                               checkpoint_path=None, force_retrain=False, seed=0, device=None):
    """Pretrains LeNet-5 via ordinary 10-class digit classification on the
    full train split, then its penultimate layer is reused as an embedding
    for the downstream OOD task. Cached to checkpoint_path if given."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path and os.path.exists(checkpoint_path) and not force_retrain:
        print(f"Loading cached pretrained LeNet-5 from '{checkpoint_path}'")
        model = LeNet5().to(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    torch.manual_seed(seed)
    if dataset == "mnist":
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    else:
        train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())

    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = preprocess_for_lenet(x, dataset, img_size)
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)
        print(f"LeNet-5 pretrain epoch {ep + 1}/{epochs}: loss={total_loss / n:.4f}, acc={correct / n:.4f}")
    model.eval()

    if checkpoint_path:
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        print(f"Pretrained LeNet-5 saved to '{checkpoint_path}'")
    return model


class LeNet5FeatureExtractor(nn.Module):
    """Wraps a pretrained (frozen) LeNet-5 to match PretrainedFeatureExtractor's
    get_embedding interface."""

    def __init__(self, dataset, data_dir, img_size=32, epochs=5, lr=1e-3, batch_size=128,
                 checkpoint_path=None, force_retrain=False, seed=0, device=None):
        super().__init__()
        self.backbone_name = "lenet5"
        self.dataset = dataset
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = BACKBONE_FEATURE_DIM["lenet5"]
        self.lenet = pretrain_lenet_classifier(dataset, data_dir, img_size, epochs, lr, batch_size,
                                                checkpoint_path, force_retrain, seed, device=self.device)
        for p in self.lenet.parameters():
            p.requires_grad = False

    def forward(self, imgs):
        imgs = imgs.to(self.device)
        x = preprocess_for_lenet(imgs, self.dataset, self.img_size)
        return self.lenet.features(x)

    def get_embedding(self, imgs):
        with torch.no_grad():
            return self.forward(imgs)


def build_backbone(name):
    """Returns a frozen nn.Module mapping (N,3,H,W) -> (N, feature_dim)."""
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier = model.classifier[:-1]
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"unknown backbone {name}, choose from {list(BACKBONE_FEATURE_DIM)}")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def preprocess_for_backbone(imgs, img_size=32):
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    imgs = TF.resize(imgs, [img_size, img_size], antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, dtype=imgs.dtype, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=imgs.dtype, device=imgs.device).view(1, 3, 1, 1)
    return (imgs - mean) / std


class PretrainedFeatureExtractor(nn.Module):
    def __init__(self, backbone_name, img_size=32, device=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = build_backbone(backbone_name).to(self.device)
        self.feature_dim = BACKBONE_FEATURE_DIM[backbone_name]

    def forward(self, imgs):
        imgs = imgs.to(self.device)
        x = preprocess_for_backbone(imgs, self.img_size)
        return self.backbone(x)

    def get_embedding(self, imgs):
        with torch.no_grad():
            return self.forward(imgs)


class PretrainedDeepSVDDDetector:
    """Frozen backbone + a small trainable linear projection head, fine-tuned
    via the hypersphere objective."""

    def __init__(self, backbone, proj_dim=32):
        self.backbone = backbone
        self.device = backbone.device
        self.proj_dim = proj_dim
        self.head = None
        self.center = None
        self.R = None
        self.final_loss = None

    def initialize_center(self, train_pool_imgs, eps=0.1):
        self.head = nn.Linear(self.backbone.feature_dim, self.proj_dim).to(self.device)
        with torch.no_grad():
            feats = self.backbone.get_embedding(torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=self.device))
            embs = self.head(feats)
        c = embs.mean(dim=0)
        near_zero = c.abs() < eps
        c = torch.where(near_zero & (c >= 0), torch.full_like(c, eps), c)
        c = torch.where(near_zero & (c < 0), torch.full_like(c, -eps), c)
        self.center = c.detach()
        return self.center

    def train_svdd(self, train_pool_imgs, epochs, lr, batch_size, lambda_reg, save_path=None, force_retrain=False):
        assert self.center is not None, "call initialize_center() before train_svdd()"

        if save_path and os.path.exists(save_path) and not force_retrain:
            print(f"Loading cached fine-tuned projection head from '{save_path}'")
            ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
            self.head = nn.Linear(self.backbone.feature_dim, self.proj_dim).to(self.device)
            self.head.load_state_dict(ckpt["head_state_dict"])
            self.head.eval()
            self.R = ckpt["R"].to(self.device)
            self.final_loss = ckpt.get("final_loss")
            print(f"DeepSVDD training outcome (cached): final_loss={self.final_loss}, R={self.R.item():.4f}")
            return []

        images_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feats = self.backbone.get_embedding(images_t)
        R = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam(list(self.head.parameters()) + [R], lr=lr)
        loader = DataLoader(TensorDataset(feats), batch_size=batch_size, shuffle=True)

        self.head.train()
        loss_rec = []
        for ep in range(epochs):
            total_loss = 0.0
            for (f,) in tqdm(loader, desc=f"PretrainedDeepSVDD fine-tune epoch {ep + 1}/{epochs}"):
                optimizer.zero_grad()
                emb = self.head(f)
                dist_sq = torch.sum((emb - self.center) ** 2, dim=-1)
                loss = torch.mean(dist_sq) + lambda_reg * R ** 2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            loss_rec.append(avg_loss)
            print(f"PretrainedDeepSVDD fine-tune epoch {ep + 1}/{epochs}: loss={avg_loss:.6f}, R={R.item():.4f}")
        self.R = R.detach()
        self.final_loss = loss_rec[-1] if loss_rec else None
        self.head.eval()

        if save_path:
            torch.save({"head_state_dict": self.head.state_dict(), "R": self.R,
                        "final_loss": self.final_loss}, save_path)
            print(f"Fine-tuned projection head saved to '{save_path}'")
        return loss_rec

    def predict_score(self, imgs):
        self.head.eval()
        with torch.no_grad():
            feats = self.backbone.get_embedding(torch.as_tensor(imgs, dtype=torch.float32, device=self.device))
            emb = self.head(feats)
            scores = torch.sum((emb - self.center) ** 2, dim=-1).cpu().numpy()
        return scores


# ============================================================================
# Trainable projection head and feature-learning objectives for classical backbones
# ============================================================================

class LinearProjector(nn.Module):
    """native_dim -> proj_dim, a single trainable Linear layer."""

    def __init__(self, native_dim, proj_dim=6):
        super().__init__()
        self.native_dim = native_dim
        self.proj_dim = proj_dim
        self.proj = nn.Linear(native_dim, proj_dim)

    def forward(self, feats):
        return self.proj(feats)


class ComposedEmbedding:
    """Wraps (backbone, optional projector) into one object exposing
    get_embedding/feature_dim/device"""

    def __init__(self, backbone, projector=None):
        self.backbone = backbone
        self.projector = projector
        self.device = backbone.device
        self.feature_dim = projector.proj_dim if projector is not None else backbone.feature_dim

    def get_embedding(self, imgs_t):
        with torch.no_grad():
            feats = self.backbone.get_embedding(imgs_t)
            if self.projector is not None:
                feats = self.projector(feats)
        return feats


def augment_image(image_chw, rng, max_rotate=15.0, max_translate=2, noise_std=0.03):
    """Rotation/translation/noise augmentation for a (C,H,W) raw pixel-in-[0,1]
    classical image."""
    img = torch.as_tensor(image_chw, dtype=torch.float32).unsqueeze(0)  # (1,C,H,W)
    angle = float(rng.uniform(-max_rotate, max_rotate))
    tx = int(rng.integers(-max_translate, max_translate + 1))
    ty = int(rng.integers(-max_translate, max_translate + 1))
    img = TF.affine(img, angle=angle, translate=[tx, ty], scale=1.0, shear=0.0)
    img = img + torch.randn_like(img) * noise_std
    img = torch.clamp(img, 0.0, 1.0)
    return img.squeeze(0).numpy()


def augment_batch(images, rng, **kwargs):
    return np.stack([augment_image(img, rng, **kwargs) for img in images])


def cosine_dist_pt(p1, p2):
    if p1.dim() == 1:
        p1 = p1.unsqueeze(0)
    if p2.dim() == 1:
        p2 = p2.unsqueeze(0)
    if p1.shape[0] != p2.shape[0] and p1.shape[0] == 1:
        p1 = p1.expand_as(p2)
    if p2.shape[0] != p1.shape[0] and p2.shape[0] == 1:
        p2 = p2.expand_as(p1)
    dot = torch.sum(p1 * p2, dim=-1)
    norm1 = torch.norm(p1, dim=-1)
    norm2 = torch.norm(p2, dim=-1)
    sim = dot / (norm1 * norm2 + 1e-8)
    return 1 - sim


def train_random(projector):
    """No training at all -- the projector stays at its random init."""
    print("feature_loss=random -> projector left at random initialization, no training performed")
    return []


def train_compact(backbone, projector, train_pool_imgs, epochs, lr, batch_size, device):
    """Compactness loss: pull embeddings toward their own running mean. No
    augmentation needed -- backbone features are extracted once and reused
    every epoch."""
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    images_t = torch.as_tensor(train_pool_imgs, dtype=torch.float32, device=device)
    with torch.no_grad():
        feats = backbone.get_embedding(images_t)
    loader = DataLoader(TensorDataset(feats), batch_size=batch_size, shuffle=True)
    loss_rec = []
    for ep in range(epochs):
        projector.eval()
        all_embs = []
        with torch.no_grad():
            for (f,) in loader:
                all_embs.append(projector(f))
        center = torch.mean(torch.cat(all_embs, dim=0), dim=0)

        projector.train()
        total_loss = 0.0
        for (f,) in tqdm(loader, desc=f"compact epoch {ep + 1}/{epochs}"):
            optimizer.zero_grad()
            emb = projector(f)
            loss = torch.mean(cosine_dist_pt(emb, center))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_rec.append(avg_loss)
        print(f"compact epoch {ep + 1}/{epochs}: loss={avg_loss:.4f}")
    return loss_rec


def train_cosine_pair(backbone, projector, train_pool_imgs, epochs, lr, batch_size, device, aug_kwargs, seed=0):
    """Positive-pair cosine loss: two augmented views of the same ID image,
    pulled together via 1 - cosine_similarity."""
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(train_pool_imgs)
    loss_rec = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        projector.train()
        total_loss = 0.0
        n_batches = 0
        for start in tqdm(range(0, n, batch_size), desc=f"cosine epoch {ep + 1}/{epochs}"):
            idx = perm[start:start + batch_size]
            batch = train_pool_imgs[idx]
            x1 = torch.as_tensor(augment_batch(batch, rng, **aug_kwargs), dtype=torch.float32, device=device)
            x2 = torch.as_tensor(augment_batch(batch, rng, **aug_kwargs), dtype=torch.float32, device=device)
            with torch.no_grad():
                f1 = backbone.get_embedding(x1)
                f2 = backbone.get_embedding(x2)

            optimizer.zero_grad()
            z1 = projector(f1)
            z2 = projector(f2)
            loss = torch.mean(cosine_dist_pt(z1, z2))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        loss_rec.append(avg_loss)
        print(f"cosine epoch {ep + 1}/{epochs}: loss={avg_loss:.4f}")
    return loss_rec


def vicreg_loss(z1, z2, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0, gamma=1.0, eps=1e-4):
    """Bardes et al. 2022 VICReg loss."""
    l_inv = torch.mean((z1 - z2) ** 2)

    def variance_term(z):
        std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
        return torch.mean(torch.relu(gamma - std))

    l_var = variance_term(z1) + variance_term(z2)

    def covariance_term(z):
        zc = z - z.mean(dim=0, keepdim=True)
        b, d = zc.shape
        if b < 2:
            return torch.zeros((), dtype=z.dtype, device=z.device)
        cov = (zc.t() @ zc) / (b - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / d

    l_cov = covariance_term(z1) + covariance_term(z2)

    total = lambda_inv * l_inv + lambda_var * l_var + lambda_cov * l_cov
    return total, l_inv, l_var, l_cov


def train_vicreg(backbone, projector, train_pool_imgs, epochs, lr, batch_size, device, aug_kwargs,
                  lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0, gamma=1.0, seed=0):
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(train_pool_imgs)
    loss_rec = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        projector.train()
        total_loss = 0.0
        n_batches = 0
        last_terms = (0.0, 0.0, 0.0)
        for start in tqdm(range(0, n, batch_size), desc=f"vicreg epoch {ep + 1}/{epochs}"):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue  # covariance term needs >=2 samples
            batch = train_pool_imgs[idx]
            x1 = torch.as_tensor(augment_batch(batch, rng, **aug_kwargs), dtype=torch.float32, device=device)
            x2 = torch.as_tensor(augment_batch(batch, rng, **aug_kwargs), dtype=torch.float32, device=device)
            with torch.no_grad():
                f1 = backbone.get_embedding(x1)
                f2 = backbone.get_embedding(x2)

            optimizer.zero_grad()
            z1 = projector(f1)
            z2 = projector(f2)
            loss, l_inv, l_var, l_cov = vicreg_loss(z1, z2, lambda_inv, lambda_var, lambda_cov, gamma)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            last_terms = (l_inv.item(), l_var.item(), l_cov.item())
        avg_loss = total_loss / max(1, n_batches)
        loss_rec.append(avg_loss)
        print(f"vicreg epoch {ep + 1}/{epochs}: loss={avg_loss:.4f} "
              f"(inv={last_terms[0]:.4f}, var={last_terms[1]:.4f}, cov={last_terms[2]:.4f})")
    return loss_rec


def train_projector(feature_loss, backbone, projector, train_pool_imgs, epochs, lr, batch_size, device,
                     aug_kwargs, vicreg_kwargs, seed=0):
    """Dispatcher matching the quantum feature-extractor's --feature_loss switch."""
    if feature_loss == "random":
        return train_random(projector)
    elif feature_loss == "compact":
        return train_compact(backbone, projector, train_pool_imgs, epochs, lr, batch_size, device)
    elif feature_loss == "cosine":
        return train_cosine_pair(backbone, projector, train_pool_imgs, epochs, lr, batch_size, device,
                                  aug_kwargs, seed=seed)
    elif feature_loss == "vicreg":
        return train_vicreg(backbone, projector, train_pool_imgs, epochs, lr, batch_size, device,
                             aug_kwargs, seed=seed, **vicreg_kwargs)
    else:
        raise ValueError(f"unknown feature_loss {feature_loss}")
