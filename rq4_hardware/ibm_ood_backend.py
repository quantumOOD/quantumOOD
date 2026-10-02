"""Builds quantum circuits for IBM hardware submission and checks them against the simulated ground truth."""
import os

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

DRNN_DEPTH = 4  # qcl_family_quanforge_ood.py's DRNN_DEPTH


def build_drnn_circuit(n_qubits, enc_params, drnn_scaling=1.5, drnn_ent_train=False):
    """Gate-for-gate translation of qcl_family_quanforge_ood.py's
    get_qnode(..., circuit_name="DRNN") circuit, BEFORE transpilation.

    enc_params: the trained circuit weights (checkpoint's "enc_params",
    length DRNN_DEPTH*n_qubits*3 + DRNN_DEPTH*n_qubits*tuple_size) -- FIXED
    per checkpoint, bound as numeric constants (not Qiskit Parameters),
    since they never vary across test samples.

    Returns (qc, x_params): qc is a QuantumCircuit with n_qubits qubits and
    n_qubits*DRNN_DEPTH*3 free Qiskit Parameters (the per-sample pixel
    values actually consumed by the circuit -- DRNN only reads the first 3
    pixels of each (layer, qubit)'s slice of the image, not the full
    flattened image). x_params is a flat list of those Parameters in the
    exact order (layer, qubit, {0,1,2}) that drnn_x_values_for_sample()
    below produces values in, so dict(zip(x_params,
    drnn_x_values_for_sample(...))) is a valid binding.

    Readout: caller is responsible for measuring/estimating per-qubit
    PauliZ (this project's --readout expval convention) -- see
    drnn_z_observables().
    """
    enc_params = np.asarray(enc_params, dtype=np.float64)
    tuple_size = 2 if drnn_ent_train else 1
    n_input = DRNN_DEPTH * n_qubits * 3
    n_var = DRNN_DEPTH * n_qubits * tuple_size
    expected_len = n_input + n_var
    if enc_params.shape[0] != expected_len:
        raise ValueError(f"enc_params has length {enc_params.shape[0]}, expected {expected_len} "
                          f"for n_qubits={n_qubits}, drnn_ent_train={drnn_ent_train}")
    w_input = enc_params[:n_input].reshape(DRNN_DEPTH, n_qubits, 3)
    w_var = enc_params[n_input:].reshape(DRNN_DEPTH, n_qubits, tuple_size)

    qc = QuantumCircuit(n_qubits, name="DRNN")
    x_params = []
    for l in range(DRNN_DEPTH):
        for q in range(n_qubits):
            p0 = Parameter(f"x_{l}_{q}_0")
            p1 = Parameter(f"x_{l}_{q}_1")
            p2 = Parameter(f"x_{l}_{q}_2")
            x_params.extend([p0, p1, p2])
            qc.rx(drnn_scaling * p0 + float(w_input[l, q, 0]), q)
            qc.rz(drnn_scaling * p1 + float(w_input[l, q, 1]), q)
            qc.rx(drnn_scaling * p2 + float(w_input[l, q, 2]), q)
        for q in range(n_qubits):
            qc.rx(float(w_var[l, q, 0]), q)
        if drnn_ent_train:
            # qml.CRZ(w_var[l, i, -1], wires=[i, i+1]) ring -- not needed for
            # this project's checkpoints (all trained with drnn_ent_train=
            # False, "entFalse" in every checkpoint filename), left
            # unimplemented rather than silently wrong.
            raise NotImplementedError("drnn_ent_train=True (CRZ ring) not translated -- "
                                       "no checkpoint in this project uses it (all are 'entFalse')")
        else:
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(n_qubits - 1, 0)
    return qc, x_params


def drnn_x_values_for_sample(x_image, n_qubits, feature_per_layer=None):
    """Extracts the SAME 3-values-per-(layer,qubit) slice of a flattened,
    preprocessed image that the DRNN circuit reads (base = l*feature_per_layer
    + q*3, values at base, base+1, base+2), in the exact (layer, qubit,
    {0,1,2}) order build_drnn_circuit()'s x_params list uses. x_image must
    be the SAME preprocessed (padded/L2-normalized) array the original
    circuit(x, params) call receives -- this function does not repeat that
    preprocessing itself; feed it the identical tensor used for the
    ideal-simulation embedding.

    feature_per_layer defaults to len(x_image) // DRNN_DEPTH, which matches
    this project's 256-pixel 16x16 images (len(x_image) >= 72)."""
    x_image = np.asarray(x_image, dtype=np.float64)
    if feature_per_layer is None:
        feature_per_layer = len(x_image) // DRNN_DEPTH
    values = []
    for l in range(DRNN_DEPTH):
        for q in range(n_qubits):
            base = l * feature_per_layer + q * 3
            values.extend([x_image[base + 0], x_image[base + 1], x_image[base + 2]])
    return np.array(values)


def drnn_z_observables(n_qubits):
    """One single-qubit PauliZ SparsePauliOp per qubit, in qubit order 0..n_qubits-1
    -- matches qcl_family_quanforge_ood.py's readout="expval":
    [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]."""
    obs = []
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"  # Qiskit's Pauli labels are little-endian (qubit 0 = rightmost char)
        obs.append(SparsePauliOp("".join(label)))
    return obs


def drnn_expvals_statevector(qc_template, x_params, x_values, n_qubits):
    """Exact LOCAL analytic evaluation (no shots) of the DRNN circuit's
    per-qubit <Z> expectation values for one bound sample -- the Qiskit-side
    half of the equivalence check. Uses Statevector, matching this
    project's other exact-simulation conventions (no sampling noise)."""
    bound = qc_template.assign_parameters(dict(zip(x_params, x_values)))
    sv = Statevector(bound)
    return np.array([sv.expectation_value(o).real for o in drnn_z_observables(n_qubits)])


def drnn_equivalence_check(checkpoint_path, x_images, n_qubits=8, drnn_scaling=None,
                            drnn_ent_train=None, tol=1e-6):
    """Local equivalence check for the DRNN circuit: loads the SAME checkpoint
    via qcl_family_quanforge_ood.QuantumFeatureExtractor (PennyLane, ground
    truth) and via build_drnn_circuit() (Qiskit, the hardware-bound
    translation above), evaluates both on x_images (a small list/array of
    already-preprocessed test images), and asserts the per-qubit <Z> values
    agree within tol. Requires pennylane. Run this before trusting
    build_drnn_circuit() for real submission.

    Returns (max_abs_err, per_sample_errors) on success; raises AssertionError
    with a diagnostic breakdown on mismatch."""
    import sys as _sys
    import torch
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rq1_feature_extractor"))
    from qcl_family_quanforge_ood import QuantumFeatureExtractor

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    extractor = QuantumFeatureExtractor.from_checkpoint(
        checkpoint_path, "DRNN", readout=ckpt.get("readout", "expval"),
        drnn_ent_train=ckpt.get("drnn_ent_train", False) if drnn_ent_train is None else drnn_ent_train,
        drnn_scaling=ckpt.get("drnn_scaling", 1.5) if drnn_scaling is None else drnn_scaling,
    )
    extractor.eval()
    enc_params = extractor.enc_params.detach().numpy()

    qc_template, x_params = build_drnn_circuit(n_qubits, enc_params, extractor.drnn_scaling,
                                                extractor.drnn_ent_train)

    per_sample_errors = []
    for x_image in x_images:
        x_t = torch.as_tensor(np.asarray(x_image, dtype=np.float64))
        with torch.no_grad():
            pennylane_out = extractor.get_embedding(x_t.unsqueeze(0)).squeeze(0).numpy()

        x_vals = drnn_x_values_for_sample(np.asarray(x_image, dtype=np.float64), n_qubits)
        qiskit_out = drnn_expvals_statevector(qc_template, x_params, x_vals, n_qubits)

        err = np.max(np.abs(pennylane_out - qiskit_out))
        per_sample_errors.append(err)
        if err >= tol:
            raise AssertionError(
                f"DRNN circuit equivalence FAILED: max abs error {err:.2e} >= tol {tol:.1e}\n"
                f"  PennyLane <Z>: {pennylane_out}\n"
                f"  Qiskit    <Z>: {qiskit_out}\n"
                f"Do not submit hardware jobs built from build_drnn_circuit() until this passes."
            )

    max_abs_err = float(np.max(per_sample_errors))
    print(f"DRNN circuit equivalence check PASSED on {len(x_images)} samples: "
          f"max_abs_err={max_abs_err:.2e} (tol={tol:.1e})")
    return max_abs_err, per_sample_errors


# ====================== FastVQC (QAE/QVAE/QGANomaly's shared ansatz) ======================

def n_qubit_z_observables(n_qubits):
    """Generalizes drnn_z_observables() -- identical construction, reused
    for FastVQC's readout (also per-qubit PauliZ, all n_qubits of them,
    return_all_z=True; see the qae subcommand's FastVQC.forward())."""
    return drnn_z_observables(n_qubits)


def build_fastvqc_circuit(n_qubits, n_layers, weights, enc_gate="ry", rot_gate="RX+RZ",
                           entangle_gate="CRY", param_prefix="x"):
    """Gate-for-gate translation of the qae subcommand's FastVQC.forward() (shared
    verbatim by the qvae and ganomaly subcommands) BEFORE transpilation.

    weights: the trained VQC weights (checkpoint's "*_vqc.weights", length
    count_vqc_params(n_qubits, n_layers, rot_gate, entangle_gate)) -- FIXED,
    bound as numeric constants.

    Gate order per FastVQC.forward(), verified against source line-by-line:
      1. encoding: RY(x_i) [enc_gate="ry", QAE/QVAE's encoder+decoder VQCs
         and QGANomaly's QuantumEncoder] or RX(x_i) [enc_gate="rx",
         QGANomaly's QuantumDecoder] on each qubit i, in qubit order.
      2. n_layers times:
         a. rotation: for each qubit i (in order): RX(weights[idx]); idx+=1
            [rot_gate="RX+RZ", this project's only value in use] then
            RZ(weights[idx]); idx+=1 -- i.e. RX then RZ on qubit i BEFORE
            moving to qubit i+1, not "all RX's then all RZ's".
         b. entangling ring: for each qubit i: CRY(weights[idx], control=i,
            target=(i+1)%n_qubits); idx+=1 -- full ring including the
            wraparound edge (n_qubits-1 -> 0), ALL n_qubits edges are
            trainable (unlike DRNN's ring, which uses a plain untrained
            CNOT ring for this project's checkpoints).
      Readout: per-qubit <Z> on all n_qubits qubits (see n_qubit_z_observables()).

    Returns (qc, x_params): qc has n_qubits free Qiskit Parameters (one per
    qubit's encoding angle, in qubit order 0..n_qubits-1 -- x_params[i]
    corresponds directly to angles[:, i] in the qae subcommand's encode()/decode()).
    """
    weights = np.asarray(weights, dtype=np.float64)
    rot_per_q = 2 if rot_gate == "RX+RZ" else 1
    n_rot = n_qubits * rot_per_q * n_layers
    n_ent = n_qubits * n_layers if entangle_gate == "CRY" else 0
    expected_len = n_rot + n_ent
    if weights.shape[0] != expected_len:
        raise ValueError(f"weights has length {weights.shape[0]}, expected {expected_len} "
                          f"for n_qubits={n_qubits}, n_layers={n_layers}, rot_gate={rot_gate}, "
                          f"entangle_gate={entangle_gate}")

    qc = QuantumCircuit(n_qubits, name="FastVQC")
    x_params = [Parameter(f"{param_prefix}_{i}") for i in range(n_qubits)]
    enc = qc.rx if enc_gate == "rx" else qc.ry
    for i in range(n_qubits):
        enc(x_params[i], i)

    idx = 0
    for _ in range(n_layers):
        if rot_gate == "RX":
            for i in range(n_qubits):
                qc.rx(float(weights[idx]), i); idx += 1
        elif rot_gate == "RY":
            for i in range(n_qubits):
                qc.ry(float(weights[idx]), i); idx += 1
        else:  # RX+RZ
            for i in range(n_qubits):
                qc.rx(float(weights[idx]), i); idx += 1
                qc.rz(float(weights[idx]), i); idx += 1
        for i in range(n_qubits):
            t = (i + 1) % n_qubits
            if entangle_gate == "CNOT":
                qc.cx(i, t)
            else:  # CRY
                qc.cry(float(weights[idx]), i, t); idx += 1
    assert idx == expected_len, f"consumed {idx} weights, expected {expected_len} -- gate/index-order bug"
    return qc, x_params


def fastvqc_expvals_statevector(qc_template, x_params, angle_values, n_qubits):
    """Exact LOCAL analytic <Z> per qubit for one bound FastVQC circuit --
    mirrors drnn_expvals_statevector()."""
    bound = qc_template.assign_parameters(dict(zip(x_params, angle_values)))
    sv = Statevector(bound)
    return np.array([sv.expectation_value(o).real for o in n_qubit_z_observables(n_qubits)])


def fastvqc_equivalence_check(n_qubits, n_layers, weights, angle_batches, enc_gate="ry",
                               rot_gate="RX+RZ", entangle_gate="CRY", tol=1e-4):
    """Local equivalence check for the FastVQC circuit: runs the qae
    subcommand's own FastVQC.forward()
    angle_batches: (B, n_qubits) array of encoding angles.
    Raises AssertionError with a diagnostic breakdown on mismatch."""
    import sys as _sys
    import torch
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main_experiments"))
    from quantum_OOD_detectors import FastVQC  # noqa: E402

    ref = FastVQC(n_qubits, n_layers, enc_gate=enc_gate, rot_gate=rot_gate,
                  entangle_gate=entangle_gate, return_all_z=True)
    with torch.no_grad():
        ref.weights.copy_(torch.as_tensor(np.asarray(weights, dtype=np.float32)))
    ref.eval()

    qc_template, x_params = build_fastvqc_circuit(n_qubits, n_layers, weights, enc_gate,
                                                   rot_gate, entangle_gate)

    per_sample_errors = []
    angle_batches = np.asarray(angle_batches, dtype=np.float64)
    for angles in angle_batches:
        with torch.no_grad():
            ref_out = ref(torch.as_tensor(angles, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        qiskit_out = fastvqc_expvals_statevector(qc_template, x_params, angles, n_qubits)
        err = np.max(np.abs(ref_out - qiskit_out))
        per_sample_errors.append(err)
        if err >= tol:
            raise AssertionError(
                f"FastVQC circuit equivalence FAILED: max abs error {err:.2e} >= tol {tol:.1e}\n"
                f"  angles: {angles}\n"
                f"  FastVQC (PyTorch) <Z>: {ref_out}\n"
                f"  Qiskit             <Z>: {qiskit_out}\n"
                f"Do not submit hardware jobs built from build_fastvqc_circuit() until this passes."
            )

    max_abs_err = float(np.max(per_sample_errors))
    print(f"FastVQC circuit equivalence check PASSED on {len(angle_batches)} samples "
          f"(enc_gate={enc_gate}, rot_gate={rot_gate}, entangle_gate={entangle_gate}): "
          f"max_abs_err={max_abs_err:.2e} (tol={tol:.1e})")
    return max_abs_err, per_sample_errors


# ====================== Real-hardware measurement and counts decoding ======================

def add_measurements(qc):
    """Returns a COPY of qc with measure_all() added -- equivalence checks
    (above) always run against the unmeasured original; only this measured
    copy is transpiled/submitted to real hardware."""
    measured = qc.copy()
    measured.measure_all()
    return measured


def z_expectations_from_counts(counts, n_qubits):
    """Per-qubit <Z> from measured shot counts, qubit order 0..n_qubits-1 --
    """
    total = sum(counts.values())
    if total == 0:
        raise ValueError("empty counts dict -- no shots?")
    z = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[::-1]  # rightmost char (qubit 0) -> bits[0]
        for q in range(n_qubits):
            z[q] += (1 - 2 * int(bits[q])) * count
    return z / total
