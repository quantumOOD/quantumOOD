"""QCL/QCNN/HCQC/DRNN quantum feature extractors for distance-based OOD detection."""
import argparse
import glob
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main_experiments"))

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from scoring_pipelines import load_ood_split_scaled, set_seed


# ====================== QuanForge-fixed circuit constants ======================
CIRCUIT_N_QUBITS = {"QCL": 8, "QCNN": 8, "HCQC": 8, "DRNN": 8}
QCL_DEPTH = 5
HCQC_DEPTH = 3
DRNN_DEPTH = 4
DRNN_TARGET_DIM = 72  # body/drnn.py's own default; our 256-dim images are
                       # already >= this, so QuanForge's pad_input() is a
                       # no-op (it only pads up, never truncates down)


# ====================== HCQC: the 9 two-qubit unitary blocks (circuits.py, verbatim) ======================
def _U_TTN(params, wires):  # 2 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def _U_5(params, wires):  # 10 params
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRZ(params[4], wires=[wires[1], wires[0]])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def _U_6(params, wires):  # 10 params
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])
    qml.RZ(params[2], wires=wires[0])
    qml.RZ(params[3], wires=wires[1])
    qml.CRX(params[4], wires=[wires[1], wires[0]])
    qml.CRX(params[5], wires=[wires[0], wires[1]])
    qml.RX(params[6], wires=wires[0])
    qml.RX(params[7], wires=wires[1])
    qml.RZ(params[8], wires=wires[0])
    qml.RZ(params[9], wires=wires[1])


def _U_9(params, wires):  # 2 params
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.CZ(wires=[wires[0], wires[1]])
    qml.RX(params[0], wires=wires[0])
    qml.RX(params[1], wires=wires[1])


def _U_13(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRZ(params[2], wires=[wires[0], wires[1]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRZ(params[5], wires=[wires[0], wires[1]])


def _U_14(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CRX(params[2], wires=[wires[1], wires[0]])
    qml.RY(params[3], wires=wires[0])
    qml.RY(params[4], wires=wires[1])
    qml.CRX(params[5], wires=[wires[0], wires[1]])


def _U_15(params, wires):  # 4 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def _U_SO4(params, wires):  # 6 params
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[4], wires=wires[0])
    qml.RY(params[5], wires=wires[1])


def _U_SU4(params, wires):  # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


HCQC_UNITARIES = {
    "U_TTN": _U_TTN, "U_5": _U_5, "U_6": _U_6, "U_9": _U_9, "U_13": _U_13,
    "U_14": _U_14, "U_15": _U_15, "U_SO4": _U_SO4, "U_SU4": _U_SU4,
}
HCQC_UNITARY_PARAMS = {
    "U_TTN": 2, "U_5": 10, "U_6": 10, "U_9": 2, "U_13": 6,
    "U_14": 6, "U_15": 4, "U_SO4": 6, "U_SU4": 15,
}


# ====================== QCNN: conv/pool/fc blocks (layers.py, verbatim for n_qubits=8) ======================
def _qcnn_conv1(n_qubits, w):
    for i in range(0, n_qubits, 2):
        qml.U3(w[i, 0], w[i, 1], w[i, 2], wires=i)
        qml.U3(w[i, 3], w[i, 4], w[i, 5], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
        qml.RY(w[i, 6], wires=i)
        qml.RZ(w[i, 7], wires=i + 1)
        qml.CNOT(wires=[i + 1, i])
        qml.RY(w[i, 8], wires=i)
        qml.CNOT(wires=[i, i + 1])
        qml.U3(w[i, 9], w[i, 10], w[i, 11], wires=i)
        qml.U3(w[i, 12], w[i, 13], w[i, 14], wires=i + 1)
    for i in range(1, n_qubits - 1, 2):
        qml.U3(w[i, 0], w[i, 1], w[i, 2], wires=i)
        qml.U3(w[i, 3], w[i, 4], w[i, 5], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
        qml.RY(w[i, 6], wires=i)
        qml.RZ(w[i, 7], wires=i + 1)
        qml.CNOT(wires=[i + 1, i])
        qml.RY(w[i, 8], wires=i)
        qml.CNOT(wires=[i, i + 1])
        qml.U3(w[i, 9], w[i, 10], w[i, 11], wires=i)
        qml.U3(w[i, 12], w[i, 13], w[i, 14], wires=i + 1)
    qml.U3(w[n_qubits - 1, 0], w[n_qubits - 1, 1], w[n_qubits - 1, 2], wires=0)
    qml.U3(w[n_qubits - 1, 3], w[n_qubits - 1, 4], w[n_qubits - 1, 5], wires=n_qubits - 1)
    qml.CNOT(wires=[0, n_qubits - 1])
    qml.RY(w[n_qubits - 1, 6], wires=0)
    qml.RZ(w[n_qubits - 1, 7], wires=n_qubits - 1)
    qml.CNOT(wires=[n_qubits - 1, 0])
    qml.RY(w[n_qubits - 1, 8], wires=0)
    qml.CNOT(wires=[0, n_qubits - 1])
    qml.U3(w[n_qubits - 1, 9], w[n_qubits - 1, 10], w[n_qubits - 1, 11], wires=0)
    qml.U3(w[n_qubits - 1, 12], w[n_qubits - 1, 13], w[n_qubits - 1, 14], wires=n_qubits - 1)


def _qcnn_pool1(n_qubits, w):
    for idx, i in enumerate(range(0, n_qubits, 2)):
        qml.CRZ(w[idx, 0], wires=[i + 1, i])
        qml.PauliX(wires=i + 1)
        qml.CRX(w[idx, 1], wires=[i + 1, i])


def _qcnn_conv2(n_qubits, w):
    for idx, i in enumerate(range(0, n_qubits - 2, 2)):
        qml.U3(w[idx, 0], w[idx, 1], w[idx, 2], wires=i)
        qml.U3(w[idx, 3], w[idx, 4], w[idx, 5], wires=i + 2)
        qml.CNOT(wires=[i, i + 2])
        qml.RY(w[idx, 6], wires=i)
        qml.RZ(w[idx, 7], wires=i + 2)
        qml.CNOT(wires=[i + 2, i])
        qml.RY(w[idx, 8], wires=i)
        qml.CNOT(wires=[i, i + 2])
        qml.U3(w[idx, 9], w[idx, 10], w[idx, 11], wires=i)
        qml.U3(w[idx, 12], w[idx, 13], w[idx, 14], wires=i + 2)


def _qcnn_pool2(n_qubits, w):
    for idx, i in enumerate(range(0, n_qubits - 2, 4)):
        qml.CRZ(w[idx, 0], wires=[i + 2, i])
        qml.PauliX(wires=i + 2)
        qml.CRX(w[idx, 1], wires=[i + 2, i])
    # QuanForge's n_qubits==10 extra block is dropped: this file fixes
    # QCNN's n_qubits=8, matching QuanForge's own qubit_dict['qcnn'].


def _qcnn_fc(n_qubits, w):
    # QuanForge's n_qubits==8 branch (the only one relevant here)
    qml.CNOT(wires=[0, 4])
    qml.CNOT(wires=[2, 4])
    qml.CNOT(wires=[4, 0])
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=2)
    qml.RX(w[2], wires=4)


# ====================== Per-circuit param counts (QuanForge-fixed, not user-configurable) ======================
def count_circuit_params(circuit_name, hcqc_unitary="U_SU4", drnn_ent_train=False):
    if circuit_name == "QCL":
        n_qubits = CIRCUIT_N_QUBITS["QCL"]
        return QCL_DEPTH * n_qubits * 3
    elif circuit_name == "QCNN":
        n_qubits = CIRCUIT_N_QUBITS["QCNN"]
        n_conv1 = n_qubits * 15                       # (8, 15)
        n_conv2 = math.ceil((n_qubits - 2) / 2) * 15   # (3, 15)
        n_pool1 = math.ceil(n_qubits / 2) * 2          # (4, 2)
        n_pool2 = math.ceil(n_qubits / 4) * 2          # (2, 2)
        n_fc = 3
        return n_conv1 + n_conv2 + n_pool1 + n_pool2 + n_fc
    elif circuit_name == "HCQC":
        n_qubits = CIRCUIT_N_QUBITS["HCQC"]
        u_params = HCQC_UNITARY_PARAMS[hcqc_unitary]
        return n_qubits * u_params  # Hierarchical_structure's params_all has n_qubits slots
    elif circuit_name == "DRNN":
        n_qubits = CIRCUIT_N_QUBITS["DRNN"]
        tuple_size = 2 if drnn_ent_train else 1
        n_input = DRNN_DEPTH * n_qubits * 3
        n_var = DRNN_DEPTH * n_qubits * tuple_size
        return n_input + n_var
    else:
        raise ValueError(f"unknown circuit {circuit_name}")


def get_qnode(circuit_name, readout="probs", hcqc_unitary="U_SU4", drnn_ent_train=False, drnn_scaling=1.5):
    n_qubits = CIRCUIT_N_QUBITS[circuit_name]
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(x, params):
        if circuit_name == "QCL":
            w = params.reshape(QCL_DEPTH, n_qubits, 3)
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True, pad_with=0.0)
            for d in range(QCL_DEPTH):
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
                for i in range(n_qubits):
                    qml.RX(w[d, i, 0], wires=i)
                    qml.RZ(w[d, i, 1], wires=i)
                    qml.RX(w[d, i, 2], wires=i)

        elif circuit_name == "QCNN":
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True, pad_with=0.0)
            n_conv1, n_conv2 = n_qubits * 15, math.ceil((n_qubits - 2) / 2) * 15
            n_pool1, n_pool2 = math.ceil(n_qubits / 2) * 2, math.ceil(n_qubits / 4) * 2
            idx = 0
            w_conv1 = params[idx:idx + n_conv1].reshape(n_qubits, 15); idx += n_conv1
            w_conv2 = params[idx:idx + n_conv2].reshape(math.ceil((n_qubits - 2) / 2), 15); idx += n_conv2
            w_pool1 = params[idx:idx + n_pool1].reshape(math.ceil(n_qubits / 2), 2); idx += n_pool1
            w_pool2 = params[idx:idx + n_pool2].reshape(math.ceil(n_qubits / 4), 2); idx += n_pool2
            w_fc = params[idx:idx + 3]
            _qcnn_conv1(n_qubits, w_conv1)
            _qcnn_pool1(n_qubits, w_pool1)
            _qcnn_conv2(n_qubits, w_conv2)
            _qcnn_pool2(n_qubits, w_pool2)
            _qcnn_fc(n_qubits, w_fc)

        elif circuit_name == "HCQC":
            U = HCQC_UNITARIES[hcqc_unitary]
            u_params = HCQC_UNITARY_PARAMS[hcqc_unitary]
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True, pad_with=0.0)
            params_all = [params[i * u_params:(i + 1) * u_params] for i in range(n_qubits)]
            # ported exactly, including QuanForge's own index arithmetic
            # (params_all[4] unused, params_all[-1] reused across layers)
            l = 0
            for idx_pair, i in enumerate(range(0, n_qubits, 2)):
                U(params_all[idx_pair], wires=[i, i + 1])
                l += 1
            for idx_pair, i in enumerate(range(1, n_qubits - 1, 2)):
                U(params_all[idx_pair + 1 + l], wires=[i, i + 2])
            t = n_qubits // 2 - 1 if (n_qubits // 2) % 2 == 0 else n_qubits // 2
            U(params_all[-1], wires=[int(t), n_qubits - 1])

        elif circuit_name == "DRNN":
            tuple_size = 2 if drnn_ent_train else 1
            n_input = DRNN_DEPTH * n_qubits * 3
            w_input = params[:n_input].reshape(DRNN_DEPTH, n_qubits, 3)
            w_var = params[n_input:].reshape(DRNN_DEPTH, n_qubits, tuple_size)
            # QuanForge's pad_input(x, target_dim=72): the 256-dim (16x16)
            # image is already >= 72, so this zero-pad is a no-op.
            if x.shape[0] < DRNN_TARGET_DIM:
                x = torch.nn.functional.pad(x, (0, DRNN_TARGET_DIM - x.shape[0]))
            n_features = x.shape[0]
            feature_per_layer = n_features // DRNN_DEPTH
            for l in range(DRNN_DEPTH):
                for q in range(n_qubits):
                    base = l * feature_per_layer + q * 3
                    qml.RX(drnn_scaling * x[base + 0] + w_input[l, q, 0], wires=q)
                    qml.RZ(drnn_scaling * x[base + 1] + w_input[l, q, 1], wires=q)
                    qml.RX(drnn_scaling * x[base + 2] + w_input[l, q, 2], wires=q)
                for q in range(n_qubits):
                    qml.RX(w_var[l, q, 0], wires=q)
                if drnn_ent_train:
                    for i in range(n_qubits - 1):
                        qml.CRZ(w_var[l, i, -1], wires=[i, i + 1])
                    qml.CRZ(w_var[l, n_qubits - 1, -1], wires=[n_qubits - 1, 0])
                else:
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])

        if readout == "probs":
            return qml.probs(wires=range(n_qubits))
        else:
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def cosine_dist_pt(p1, p2):
    # NO positive floor here -- unlike probs (always >=0), expval embeddings
    # are signed (Pauli-Z expectations in [-1, 1]); clamping them to a
    # positive minimum would corrupt the cosine similarity's sign. The
    # +1e-8 on the denominator below is enough to avoid a zero-norm division.
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


def cosine_dist_np(p1, p2):
    sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)
    return 1 - sim


class QuantumFeatureExtractor(nn.Module):
    """QuanForge-faithful QCL/QCNN/HCQC/DRNN circuit, trained via one of
    --feature_loss's objectives then frozen. Two deliberate deviations from
    QuanForge's own circuits: multi-qubit readout instead of QuanForge's
    classification-specific one, and this project's L2-normalized pixel
    convention for DRNN's angle encoding.

    readout="probs": full computational-basis distribution (qml.probs),
        dim=2**n_qubits, renormalized after stacking as a defensive measure.
    readout="expval": per-qubit Pauli-Z expectation, dim=n_qubits, signed
        in [-1, 1] -- NOT renormalized.
    """

    def __init__(self, circuit_name, readout="probs", hcqc_unitary="U_SU4",
                 drnn_ent_train=False, drnn_scaling=1.5):
        super().__init__()
        self.circuit_name = circuit_name
        self.readout = readout
        self.hcqc_unitary = hcqc_unitary
        self.drnn_ent_train = drnn_ent_train
        self.drnn_scaling = drnn_scaling
        self.n_qubits = CIRCUIT_N_QUBITS[circuit_name]
        self.circuit = get_qnode(circuit_name, readout, hcqc_unitary, drnn_ent_train, drnn_scaling)
        n_params = count_circuit_params(circuit_name, hcqc_unitary, drnn_ent_train)
        self.enc_params = nn.Parameter(torch.randn(n_params) * 0.1)

    def get_embedding(self, x):
        raw_outputs = [self.circuit(xi, self.enc_params) for xi in x]
        if self.readout == "expval":
            raw_outputs = [torch.stack(list(o)) for o in raw_outputs]
        out = torch.stack(raw_outputs)
        if self.readout == "probs":
            out = torch.clamp(out, min=1e-12)
            out = out / torch.sum(out, dim=-1, keepdim=True)
        return out

    @classmethod
    def from_checkpoint(cls, checkpoint_path, circuit_name, readout="probs", hcqc_unitary="U_SU4",
                         drnn_ent_train=False, drnn_scaling=1.5):
        model = cls(circuit_name, readout, hcqc_unitary, drnn_ent_train, drnn_scaling)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


def train_shared_ansatz(extractor, train_pool_imgs, epochs, lr, batch_size):
    """Compactness Loss: each epoch, re-derive the ID center from the
    CURRENT embeddings, then pull embeddings toward that (fixed-for-the-
    epoch) center."""
    optimizer = torch.optim.Adam(extractor.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.as_tensor(train_pool_imgs, dtype=torch.float64)),
                         batch_size=batch_size, shuffle=True)
    loss_rec = []
    for ep in range(epochs):
        extractor.eval()
        all_embs = []
        with torch.no_grad():
            for (x,) in loader:
                all_embs.append(extractor.get_embedding(x))
        center = torch.mean(torch.cat(all_embs, dim=0), dim=0)

        extractor.train()
        total_loss = 0.0
        for (x,) in tqdm(loader, desc=f"{extractor.circuit_name} epoch {ep + 1}/{epochs}"):
            optimizer.zero_grad()
            emb = extractor.get_embedding(x)
            loss = torch.mean(cosine_dist_pt(emb, center))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_rec.append(avg_loss)
        print(f"{extractor.circuit_name} epoch {ep + 1}/{epochs}: compactness_loss={avg_loss:.4f}")
    return loss_rec


def augment_image(image_flat, img_shape, rng, max_rotate=15.0, max_translate=2, noise_std=0.03):
    """One random augmented view of a single flattened, L2-normalized
    amplitude-encoding image: spatial affine jitter (rotation + integer
    pixel translation) plus additive Gaussian noise, renormalized back to
    unit L2 norm afterward -- required for the result to still be a valid
    qml.AmplitudeEmbedding input. Also used as DRNN's augmented view (its
    angle encoding doesn't strictly need unit norm, but reusing the same
    augmentation keeps the positive-pair objectives circuit-agnostic).
    """
    img = torch.as_tensor(image_flat.reshape(1, *img_shape), dtype=torch.float32)
    angle = float(rng.uniform(-max_rotate, max_rotate))
    tx = int(rng.integers(-max_translate, max_translate + 1))
    ty = int(rng.integers(-max_translate, max_translate + 1))
    img = TF.affine(img, angle=angle, translate=[tx, ty], scale=1.0, shear=0.0)
    img_np = img.squeeze(0).numpy().astype(np.float64)
    img_np = img_np + rng.normal(0, noise_std, size=img_np.shape)
    img_np = np.clip(img_np, 0, None).reshape(-1)
    norm = np.linalg.norm(img_np)
    if norm < 1e-8:
        norm = 1e-8
    return img_np / norm


def augment_batch(images, img_shape, rng, **aug_kwargs):
    return np.stack([augment_image(img, img_shape, rng, **aug_kwargs) for img in images])


def train_cosine_pair(extractor, train_pool_imgs, img_shape, epochs, lr, batch_size, aug_kwargs, seed=0):
    """Negative-free positive-pair loss: two augmented views of the same ID
    image, pulled together via 1 - cosine_similarity. ID-only, no OOD
    samples/labels."""
    optimizer = torch.optim.Adam(extractor.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(train_pool_imgs)
    loss_rec = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        extractor.train()
        total_loss = 0.0
        n_batches = 0
        for start in tqdm(range(0, n, batch_size), desc=f"{extractor.circuit_name} epoch {ep + 1}/{epochs}"):
            idx = perm[start:start + batch_size]
            batch = train_pool_imgs[idx]
            x1 = torch.as_tensor(augment_batch(batch, img_shape, rng, **aug_kwargs), dtype=torch.float64)
            x2 = torch.as_tensor(augment_batch(batch, img_shape, rng, **aug_kwargs), dtype=torch.float64)

            optimizer.zero_grad()
            z1 = extractor.get_embedding(x1)
            z2 = extractor.get_embedding(x2)
            loss = torch.mean(cosine_dist_pt(z1, z2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        loss_rec.append(avg_loss)
        print(f"{extractor.circuit_name} epoch {ep + 1}/{epochs}: cosine_pair_loss={avg_loss:.4f}")
    return loss_rec


def vicreg_loss(z1, z2, lambda_inv, lambda_var, lambda_cov, gamma=1.0, eps=1e-4):
    """Bardes et al. 2022, Eq 6-9."""
    l_inv = torch.mean((z1 - z2) ** 2)

    def variance_term(z):
        std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
        return torch.mean(torch.relu(gamma - std))

    l_var = variance_term(z1) + variance_term(z2)

    def covariance_term(z):
        zc = z - z.mean(dim=0, keepdim=True)
        b, d = zc.shape
        if b < 2:
            return torch.zeros((), dtype=z.dtype)
        cov = (zc.t() @ zc) / (b - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / d

    l_cov = covariance_term(z1) + covariance_term(z2)

    total = lambda_inv * l_inv + lambda_var * l_var + lambda_cov * l_cov
    return total, l_inv, l_var, l_cov


def train_vicreg(extractor, train_pool_imgs, img_shape, epochs, lr, batch_size, aug_kwargs,
                  lambda_inv, lambda_var, lambda_cov, gamma, seed=0):
    optimizer = torch.optim.Adam(extractor.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(train_pool_imgs)
    loss_rec = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        extractor.train()
        total_loss = 0.0
        n_batches = 0
        last_terms = (0.0, 0.0, 0.0)
        for start in tqdm(range(0, n, batch_size), desc=f"{extractor.circuit_name} epoch {ep + 1}/{epochs}"):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            batch = train_pool_imgs[idx]
            x1 = torch.as_tensor(augment_batch(batch, img_shape, rng, **aug_kwargs), dtype=torch.float64)
            x2 = torch.as_tensor(augment_batch(batch, img_shape, rng, **aug_kwargs), dtype=torch.float64)

            optimizer.zero_grad()
            z1 = extractor.get_embedding(x1)
            z2 = extractor.get_embedding(x2)
            loss, l_inv, l_var, l_cov = vicreg_loss(z1, z2, lambda_inv, lambda_var, lambda_cov, gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            last_terms = (l_inv.item(), l_var.item(), l_cov.item())
        avg_loss = total_loss / max(1, n_batches)
        loss_rec.append(avg_loss)
        print(f"{extractor.circuit_name} epoch {ep + 1}/{epochs}: vicreg_loss={avg_loss:.4f} "
              f"(inv={last_terms[0]:.4f}, var={last_terms[1]:.4f}, cov={last_terms[2]:.4f})")
    return loss_rec


def train_random(extractor):
    print(f"{extractor.circuit_name}: --feature_loss random -> circuit parameters left at random "
          f"initialization, no training performed")
    return []


def knn_scores(train_bank, test_embs, k):
    scores = []
    for x in test_embs:
        dists = np.array([cosine_dist_np(x, t) for t in train_bank])
        scores.append(np.mean(np.sort(dists)[:k]))
    return np.array(scores)


def mean_scores(train_bank, test_embs):
    center = train_bank.mean(axis=0)
    return np.array([cosine_dist_np(x, center) for x in test_embs])


def kmedoids_cosine(embeddings, n_medoids, max_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    n_medoids = min(n_medoids, n)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_dist_np(embeddings[i], embeddings[j])
            dist[i, j] = d
            dist[j, i] = d

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


def medoid_scores(train_bank, test_embs, m, seed=0):
    medoids, _ = kmedoids_cosine(train_bank, m, seed=seed)
    return np.array([min(cosine_dist_np(x, med) for med in medoids) for x in test_embs])


def clone_extractor(extractor):
    """A fresh QuantumFeatureExtractor with the same circuit config and
    CURRENT weights as `extractor`, for QSVDD to fine-tune independently."""
    clone = QuantumFeatureExtractor(extractor.circuit_name, extractor.readout, extractor.hcqc_unitary,
                                     extractor.drnn_ent_train, extractor.drnn_scaling)
    clone.load_state_dict(extractor.state_dict())
    return clone


class QSVDDDetector:
    """Starts from a COPY of the shared extractor, then fine-tunes its own
    copy of the circuit params toward a one-class hypersphere objective."""

    def __init__(self, base_extractor):
        self.base_extractor = base_extractor
        self.center = None
        self.R = None
        self.final_loss = None
        self.extractor = None

    def initialize_center(self, train_pool_imgs):
        self.base_extractor.eval()
        with torch.no_grad():
            embs = self.base_extractor.get_embedding(torch.as_tensor(train_pool_imgs, dtype=torch.float64))
        self.center = embs.mean(dim=0)
        return self.center

    def train_svdd(self, train_pool_imgs, epochs, lr, batch_size, lambda_reg, save_path=None, force_retrain=False):
        """save_path, if given: loaded (fine-tuning skipped) when it already
        exists, else fine-tuned then written there afterward -- mirrors
        modified_distance_ood.py's QSVDD caching. Without this, every run
        re-fine-tuned from scratch and discarded the result on exit.
        force_retrain=True bypasses the cache-load check (still overwrites
        save_path with the fresh result afterward, if given)."""
        assert self.center is not None, "call initialize_center() before train_svdd()"

        if save_path and os.path.exists(save_path) and not force_retrain:
            print(f"Loading cached fine-tuned QSVDD extractor from '{save_path}'")
            # weights_only=False: self-produced checkpoint (R/final_loss
            # alongside the state_dict); PyTorch >=2.6 defaults
            # weights_only=True, which rejects that.
            ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
            self.extractor = clone_extractor(self.base_extractor)
            self.extractor.load_state_dict(ckpt["extractor_state_dict"])
            self.extractor.eval()
            self.R = ckpt["R"]
            self.final_loss = ckpt.get("final_loss")
            print(f"QSVDD training outcome (cached): final_loss={self.final_loss}, R={self.R.item():.4f}")
            return []

        self.extractor = clone_extractor(self.base_extractor)
        R = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        optimizer = torch.optim.Adam(list(self.extractor.parameters()) + [R], lr=lr)

        images_t = torch.as_tensor(train_pool_imgs, dtype=torch.float64)
        loader = DataLoader(TensorDataset(images_t), batch_size=batch_size, shuffle=True)

        self.extractor.train()
        loss_rec = []
        for ep in range(epochs):
            total_loss = 0.0
            for (x,) in tqdm(loader, desc=f"QSVDD fine-tune epoch {ep + 1}/{epochs}"):
                optimizer.zero_grad()
                emb = self.extractor.get_embedding(x)
                dist_sq = torch.sum((emb - self.center) ** 2, dim=-1)
                loss = torch.mean(dist_sq) + lambda_reg * R ** 2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            loss_rec.append(avg_loss)
            print(f"QSVDD fine-tune epoch {ep + 1}/{epochs}: loss={avg_loss:.6f}, R={R.item():.4f}")
        self.R = R.detach()
        self.final_loss = loss_rec[-1] if loss_rec else None

        if save_path:
            torch.save({
                "extractor_state_dict": self.extractor.state_dict(),
                "R": self.R,
                "final_loss": self.final_loss,
            }, save_path)
            print(f"QSVDD fine-tuned extractor saved to '{save_path}'")

        return loss_rec

    def predict_score(self, imgs, return_timing=False):
        self.extractor.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            emb = self.extractor.get_embedding(torch.as_tensor(imgs, dtype=torch.float64))
            circuit_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = torch.sum((emb - self.center) ** 2, dim=-1).numpy()
            classical_seconds = time.perf_counter() - t0
        if return_timing:
            return scores, circuit_seconds, classical_seconds
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--norm_cls", type=int, default=0)
    parser.add_argument("--ood_cls", type=int, default=1)
    parser.add_argument("--setting", type=int, default=1, choices=[1, 2])
    parser.add_argument("--data_scale", type=float, default=0.2,
                         help="fraction of the norm_cls training pool to use, same convention as "
                              "QAE_digital.py/load_real_dataset (int(scale * len(class_idx)))")
    parser.add_argument("--n_val_per_cls", type=int, default=20)
    parser.add_argument("--num_latent", type=int, default=6,
                         help="num_latent+num_trash sets the image feature dim (2**(num_latent+num_trash)); "
                              "MUST sum to 8 for QCL/QCNN/HCQC (QuanForge fixes their n_qubits=8, i.e. "
                              "AmplitudeEmbedding needs exactly 2**8=256 features -- the 16x16 default)")
    parser.add_argument("--num_trash", type=int, default=2)
    parser.add_argument("--circuit", type=str, default="QCL", choices=["QCL", "QCNN", "HCQC", "DRNN"])
    parser.add_argument("--hcqc_unitary", type=str, default="U_SU4",
                         choices=["U_TTN", "U_5", "U_6", "U_9", "U_13", "U_14", "U_15", "U_SO4", "U_SU4"],
                         help="HCQC only: which 2-qubit unitary block to use in the hierarchical tree, "
                              "matching QuanForge's body/hcqc.py param_num options (default matches its own default)")
    parser.add_argument("--drnn_ent_train", action="store_true",
                         help="DRNN only: use a trainable CRZ entangling ring instead of the fixed CNOT ring "
                              "(matches QuanForge's DRNN(ent_train=True) option)")
    parser.add_argument("--drnn_scaling", type=float, default=1.5,
                         help="DRNN only: scaling factor on the re-uploaded data features (QuanForge default 1.5)")
    parser.add_argument("--readout", type=str, default="probs", choices=["probs", "expval"],
                         help="probs: full qml.probs() distribution, dim=2**n_qubits, always >=0. "
                              "expval: per-qubit <Z> expectation, dim=n_qubits, signed in [-1,1]. "
                              "NOTE: this is a deliberate deviation from QuanForge's own per-circuit "
                              "classification-specific readout")
    parser.add_argument("--feature_loss", type=str, default="compact", choices=["random", "cosine", "vicreg", "compact"],
                         help="training objective for the circuit's parameters (enc_params); does not change "
                              "the embedding extraction interface (--readout still applies to all four)")
    parser.add_argument("--aug_max_rotate", type=float, default=15.0,
                         help="cosine/vicreg augmentation: max +/- rotation in degrees")
    parser.add_argument("--aug_max_translate", type=int, default=2,
                         help="cosine/vicreg augmentation: max +/- pixel translation")
    parser.add_argument("--aug_noise_std", type=float, default=0.03,
                         help="cosine/vicreg augmentation: additive Gaussian noise std, applied before renormalizing")
    parser.add_argument("--vicreg_lambda_inv", type=float, default=25.0, help="vicreg: weight on the invariance (MSE) term")
    parser.add_argument("--vicreg_lambda_var", type=float, default=25.0, help="vicreg: weight on the variance (anti-collapse) term")
    parser.add_argument("--vicreg_lambda_cov", type=float, default=1.0, help="vicreg: weight on the covariance (redundancy) term")
    parser.add_argument("--vicreg_gamma", type=float, default=1.0,
                         help="vicreg: target per-dimension std in the variance term. NOTE: the sensible scale "
                              "differs a lot between --readout expval (dims in [-1,1], gamma=1 is reasonable) and "
                              "--readout probs (dims sum to 1, per-dim std is typically tiny -- consider "
                              "0.01-0.05 for probs)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--k", type=int, default=5, help="fixed K for the quick QKNN check -- NOT tuned/searched here")
    parser.add_argument("--m", type=int, default=5, help="fixed number of medoids for the quick QMedoids check -- NOT tuned/searched here")
    parser.add_argument("--detectors", nargs="+", type=str, default=["QKNN", "QMean", "QMedoids", "QSVDD"],
                         choices=["QKNN", "QMean", "QMedoids", "QSVDD"],
                         help="which detector(s) to compute/print -- e.g. --detectors QSVDD skips QKNN/"
                              "QMean/QMedoids' scoring entirely, and skips QSVDD's fine-tuning/checkpoint "
                              "save if QSVDD is not requested")
    parser.add_argument("--svdd_epochs", type=int, default=10,
                         help="QSVDD fine-tuning epochs (separate from --epochs, which trains the shared extractor)")
    parser.add_argument("--svdd_lr", type=float, default=1e-3)
    parser.add_argument("--svdd_lambda", type=float, default=1e-3, help="weight on R^2 in the QSVDD loss")
    parser.add_argument("--svdd_checkpoint", type=str, default=None,
                         help="path to cache/load the QSVDD fine-tuned extractor (default: auto-named from "
                              "run_tag_base + the shared extractor's current epoch + svdd hyperparams). If it "
                              "already exists, QSVDD fine-tuning is SKIPPED and the cached extractor/R/"
                              "final_loss are loaded instead -- without this, every run re-fine-tuned QSVDD "
                              "from scratch and discarded the result on exit")
    parser.add_argument("--svdd_force_retrain", action="store_true",
                         help="ignore an existing cached QSVDD checkpoint and fine-tune from scratch anyway "
                              "(still overwrites it with the fresh result afterward)")
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/qcl_family_quanforge",
                         help="folder for checkpoints/loss curves from this script (kept separate from "
                              "qcl_family_ood.py's outputs/qcl_family/, since the circuits differ)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.circuit in ("QCL", "QCNN", "HCQC", "DRNN"):
        assert args.num_latent + args.num_trash == 8, (
            f"{args.circuit}'s n_qubits=8 (AmplitudeEmbedding needs exactly 2**8=256 features); "
            f"--num_latent+--num_trash must sum to 8, got {args.num_latent + args.num_trash}"
        )

    # Same scale semantics as QAE_digital.py/load_real_dataset: n_train is a
    # FRACTION of the norm_cls training pool, not a fixed count.
    if args.dataset == "mnist":
        _train_set_for_count = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    else:
        _train_set_for_count = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    _class_count = int((_train_set_for_count.targets == args.norm_cls).sum())
    n_train = int(args.data_scale * _class_count)
    print(f"data_scale={args.data_scale} -> n_train={n_train} (of {_class_count} available {args.norm_cls}-class training images)")

    # DRNN uses the SAME 16x16=256-pixel image as QCL/QCNN/HCQC (8-qubit convention).
    data_num_latent, data_num_trash = args.num_latent, args.num_trash

    train_pool_imgs, val_imgs, val_labels, test_imgs, test_labels, img_shape, feat_dim, ood_str = load_ood_split_scaled(
        args.dataset, args.data_dir, args.norm_cls, args.ood_cls, args.setting,
        n_train, args.n_val_per_cls, args.data_scale, data_num_latent, data_num_trash
    )

    run_tag_base = (f"{args.dataset}_{args.norm_cls}_{len(train_pool_imgs)}_scale{args.data_scale}_{args.circuit}"
                     f"_{args.readout}_{args.feature_loss}_lr{args.lr}_bs{args.batch_size}")
    if args.circuit == "HCQC":
        run_tag_base += f"_{args.hcqc_unitary}"
    if args.circuit == "DRNN":
        run_tag_base += f"_ent{args.drnn_ent_train}_scl{args.drnn_scaling}"
    if args.feature_loss == "vicreg":
        run_tag_base += (f"_li{args.vicreg_lambda_inv}_lv{args.vicreg_lambda_var}"
                          f"_lc{args.vicreg_lambda_cov}_g{args.vicreg_gamma}")


    def checkpoint_path_for(step):
        if args.checkpoint:
            return args.checkpoint
        return os.path.join(args.output_dir, f"qcl_family_quanforge_checkpoint_{run_tag_base}_epoch{step}.pt")


    def find_resume_checkpoint():
        if args.checkpoint:
            return args.checkpoint if os.path.exists(args.checkpoint) else None
        candidates = glob.glob(os.path.join(args.output_dir, f"qcl_family_quanforge_checkpoint_{run_tag_base}_epoch*.pt"))
        if not candidates:
            return None

        def extract_epoch(path):
            m = re.search(r'_epoch(\d+)\.pt$', os.path.basename(path))
            return int(m.group(1)) if m else -1

        return max(candidates, key=extract_epoch)


    extractor = QuantumFeatureExtractor(args.circuit, args.readout, args.hcqc_unitary,
                                         args.drnn_ent_train, args.drnn_scaling)
    print(f"{args.circuit} (QuanForge-faithful, n_qubits={extractor.n_qubits}) trainable params: "
          f"{count_circuit_params(args.circuit, args.hcqc_unitary, args.drnn_ent_train)}")

    resume_path = find_resume_checkpoint() if args.resume else None
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        extractor.load_state_dict(ckpt["model_state_dict"])
        loss_history = list(ckpt["loss_history"])
        prior_loss_str = f"{loss_history[-1]:.4f}" if loss_history else "n/a (--feature_loss random)"
        print(f"Resumed from '{resume_path}' at epoch {len(loss_history)} (prior loss: {prior_loss_str})")
    else:
        if args.resume:
            print(f"--resume given but no checkpoint found matching "
                  f"'qcl_family_quanforge_checkpoint_{run_tag_base}_epoch*.pt', starting from scratch")
        loss_history = []

    aug_kwargs = dict(max_rotate=args.aug_max_rotate, max_translate=args.aug_max_translate, noise_std=args.aug_noise_std)

    start = time.time()
    if args.feature_loss == "compact":
        new_losses = train_shared_ansatz(extractor, train_pool_imgs, args.epochs, args.lr, args.batch_size)
    elif args.feature_loss == "cosine":
        new_losses = train_cosine_pair(extractor, train_pool_imgs, img_shape, args.epochs, args.lr,
                                        args.batch_size, aug_kwargs)
    elif args.feature_loss == "vicreg":
        new_losses = train_vicreg(extractor, train_pool_imgs, img_shape, args.epochs, args.lr, args.batch_size,
                                   aug_kwargs, args.vicreg_lambda_inv, args.vicreg_lambda_var,
                                   args.vicreg_lambda_cov, args.vicreg_gamma)
    else:  # random
        new_losses = train_random(extractor)
    elapsed = time.time() - start
    print(f"Fit in {elapsed:0.2f} seconds ({args.epochs if args.feature_loss != 'random' else 0} epochs this run)")
    loss_history.extend(new_losses)
    current_epochs = len(loss_history)

    checkpoint_path = checkpoint_path_for(current_epochs)
    torch.save({
        "model_state_dict": extractor.state_dict(),
        "loss_history": np.array(loss_history),
        "circuit": args.circuit,
        "readout": args.readout,
        "hcqc_unitary": args.hcqc_unitary,
        "drnn_ent_train": args.drnn_ent_train,
        "drnn_scaling": args.drnn_scaling,
        "feature_loss": args.feature_loss,
    }, checkpoint_path)
    print(f"[epoch {current_epochs}] checkpoint saved to '{checkpoint_path}'")

    if loss_history:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(loss_history) + 1), loss_history)
        ax.set_title(f"{args.circuit} (QuanForge) {args.feature_loss} loss against epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"qcl_family_quanforge_loss_{run_tag_base}_epoch{current_epochs}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        print("No training performed (--feature_loss random or epochs=0) -- skipping loss curve plot")

    # -------------------- quick, fixed-K/M check (only the detectors requested via --detectors) --------------------
    extractor.eval()
    with torch.no_grad():
        train_embs = extractor.get_embedding(torch.as_tensor(train_pool_imgs, dtype=torch.float64)).numpy()
        test_embs = extractor.get_embedding(torch.as_tensor(test_imgs, dtype=torch.float64)).numpy()

    scored = {}
    if "QKNN" in args.detectors:
        scored["QKNN"] = (knn_scores(train_embs, test_embs, args.k), f"K={args.k}")
    if "QMean" in args.detectors:
        scored["QMean"] = (mean_scores(train_embs, test_embs), "")
    if "QMedoids" in args.detectors:
        scored["QMedoids"] = (medoid_scores(train_embs, test_embs, args.m), f"M={args.m}")

    if "QSVDD" in args.detectors:
        # -------------------- QSVDD (fine-tunes a COPY of the extractor + R) --------------------
        svdd_checkpoint_path = args.svdd_checkpoint or os.path.join(
            args.output_dir,
            f"qcl_family_quanforge_qsvdd_checkpoint_{run_tag_base}_epoch{current_epochs}"
            f"_svdd{args.svdd_epochs}ep_lr{args.svdd_lr}_lam{args.svdd_lambda}.pt"
        )
        svdd = QSVDDDetector(extractor)
        svdd.initialize_center(train_pool_imgs)
        svdd_loss = svdd.train_svdd(train_pool_imgs, args.svdd_epochs, args.svdd_lr, args.batch_size, args.svdd_lambda,
                                     save_path=svdd_checkpoint_path, force_retrain=args.svdd_force_retrain)
        svdd_s = svdd.predict_score(test_imgs)
        svdd_final_loss_str = f"{svdd.final_loss:.6f}" if svdd.final_loss is not None else "n/a"
        print(f"QSVDD fine-tune done: final_loss={svdd_final_loss_str}, R={svdd.R.item():.4f} "
              f"({args.svdd_epochs} epochs, lr={args.svdd_lr}, lambda={args.svdd_lambda})")

        if svdd_loss:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, len(svdd_loss) + 1), svdd_loss)
            ax.set_title(f"{args.circuit} (QuanForge) QSVDD fine-tuning loss against epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (dist_sq + lambda * R^2)")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, f"qcl_family_quanforge_qsvdd_loss_{run_tag_base}_svddepoch{len(svdd_loss)}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)

        scored["QSVDD"] = (svdd_s, f"lambda={args.svdd_lambda}")

    print(f"\n{'Circuit':<10} {'Detector':<10} {'Hyperparam':<14} {'AUC-ROC':<10} {'AUC-PR':<10}")
    for name, (scores, hp) in scored.items():
        print(f"{args.circuit:<10} {name:<10} {hp:<14} "
              f"{roc_auc_score(test_labels, scores):<10.4f} {average_precision_score(test_labels, scores):<10.4f}")
