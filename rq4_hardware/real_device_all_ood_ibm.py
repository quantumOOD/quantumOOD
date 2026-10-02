#!/usr/bin/env python3
"""Submits OOD-detector circuits to real IBM quantum hardware and saves the raw measurement counts."""
import argparse
import json
import os
import time

import numpy as np
import torch

import ibm_ood_backend as ibmb
import hardware_result_utils as hru

ALL_DETECTORS = ["QKNN", "QMean", "QMedoids", "QSVDD", "DMKDE-mixed", "IndepGaussian",
                  "MVGaussian", "QAE", "QVAE", "QGANomaly", "Q-AnoGAN", "QWGAN-GP"]
SHARED_DRNN_GROUP = ["QKNN", "QMean", "QMedoids", "DMKDE-mixed", "IndepGaussian", "MVGaussian"]
DENSITY_OWN_STAGE = ["DMKDE-mixed", "IndepGaussian", "MVGaussian"]
ITERATIVE_UNSUPPORTED = ["Q-AnoGAN", "QWGAN-GP"]
ALL_BACKENDS = ["ibm_fez", "ibm_marrakesh", "ibm_kingston"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashion_mnist"])
    p.add_argument("--id_class", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, required=True, choices=ALL_BACKENDS + ["all"])
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--test_scale", type=float, default=0.1)
    p.add_argument("--target_n_id", type=int, default=100)
    p.add_argument("--hardware_batch_size", type=int, default=200)
    p.add_argument("--transpiler_optimization_level", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--checkpoint_config", type=str, default="checkpoint_config_seed0.json")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="outputs/ibm_hardware_ood")
    p.add_argument("--detectors", nargs="+", type=str, default=ALL_DETECTORS,
                    choices=ALL_DETECTORS + ["shared_drnn_group"])
    p.add_argument("--allow_expensive_iterative_gans", action="store_true")
    return p.parse_args()


def load_checkpoint_config(path):
    with open(path) as f:
        return json.load(f)


def resolve_backend_list(args):
    return ALL_BACKENDS if args.backend == "all" else [args.backend]


def resolve_detector_list(detectors):
    out = []
    for d in detectors:
        if d == "shared_drnn_group":
            out.extend(SHARED_DRNN_GROUP)
        else:
            out.append(d)
    seen = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def chunk_list(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


# ====================== Preprocessing (real data, no pennylane/qiskit needed) ======================

def load_preprocessed_images(dataset, data_dir, indices, img_size=16, train=False):
    """Loads the SAME raw dataset indices select_fixed_test_samples() chose,
    resized/L2-normalized to img_size x img_size and flattened."""
    from torchvision import datasets as tv_datasets, transforms
    import torch.nn.functional as F

    tv_cls = tv_datasets.MNIST if dataset == "mnist" else tv_datasets.FashionMNIST
    test_set = tv_cls(data_dir, train=train, download=True, transform=transforms.ToTensor())

    images = []
    for idx in indices:
        img, _ = test_set[int(idx)]
        img = F.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode="bilinear",
                             align_corners=False).squeeze()
        images.append(img.numpy().flatten())
    images = np.array(images, dtype=np.float64)
    norms = np.linalg.norm(images, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return images / norms


# ====================== Per-detector circuit-batch builders ======================

def build_shared_drnn_batch(checkpoint_config, dataset, images):
    group = checkpoint_config["shared_drnn_extractor"]
    checkpoint_path = group[dataset]["checkpoint_path"]
    n_qubits = group["n_qubits"]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    enc_params = ckpt["model_state_dict"]["enc_params"].numpy()
    qc_template, x_params = ibmb.build_drnn_circuit(n_qubits, enc_params, ckpt["drnn_scaling"],
                                                      ckpt["drnn_ent_train"])
    bound = [qc_template.assign_parameters(dict(zip(x_params, ibmb.drnn_x_values_for_sample(img, n_qubits))))
             for img in images]
    return bound, checkpoint_path


def build_qsvdd_batch(checkpoint_config, dataset, images):
    entry = checkpoint_config["QSVDD"]
    checkpoint_path = entry[dataset]["checkpoint_path"]
    n_qubits = entry["n_qubits"]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    enc_params = ckpt["extractor_state_dict"]["enc_params"].numpy()
    qc_template, x_params = ibmb.build_drnn_circuit(n_qubits, enc_params, drnn_scaling=1.5, drnn_ent_train=False)
    bound = [qc_template.assign_parameters(dict(zip(x_params, ibmb.drnn_x_values_for_sample(img, n_qubits))))
             for img in images]
    return bound, checkpoint_path


def build_fastvqc_detector_batch(checkpoint_config, detector, dataset, images):
    """Shared by QAE/QVAE/QGANomaly """
    group = checkpoint_config[detector]
    checkpoint_path = group[dataset]["checkpoint_path"]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    n_qubits, n_layers = group["n_qubits"], group["n_layers"]
    images_t = torch.as_tensor(images, dtype=torch.float32)

    stages = {}
    if detector in ("QAE", "QVAE"):
        sd = ckpt["state_dict"]
        compressed = torch.tanh(images_t @ sd["enc_compress.0.weight"].T + sd["enc_compress.0.bias"])
        enc_angles = torch.arccos(torch.clamp(compressed, -1.0, 1.0)).numpy()
        enc_key = "enc_vqc_mu.weights" if detector == "QVAE" else "enc_vqc.weights"
        qc_enc, xp_enc = ibmb.build_fastvqc_circuit(n_qubits, n_layers, sd[enc_key].numpy(), enc_gate="ry")
        stages["encoder"] = [qc_enc.assign_parameters(dict(zip(xp_enc, a))) for a in enc_angles]

    elif detector == "QGANomaly":
        sd = ckpt["G_state_dict"]
        c1 = torch.tanh(images_t @ sd["enc1.compress.0.weight"].T + sd["enc1.compress.0.bias"])
        angles1 = torch.arccos(torch.clamp(c1, -1.0, 1.0)).numpy()
        qc1, xp1 = ibmb.build_fastvqc_circuit(n_qubits, n_layers, sd["enc1.vqc.weights"].numpy(), enc_gate="ry")
        stages["enc1"] = [qc1.assign_parameters(dict(zip(xp1, a))) for a in angles1]

    return stages, checkpoint_path


# ====================== Real submission ======================

def submit_and_harvest_batch(sampler, circuits, shots, run_dir, batch_index):
    """Submits ONE batch of already-transpiled, already-measured circuits via
    job-mode SamplerV2, waits for the result, decodes counts, and saves the
    raw counts to disk. Returns counts_list."""
    print(f"    [SUBMIT] batch_index={batch_index}: {len(circuits)} circuits, {shots} shots")
    submit_ts = hru.utc_now_iso()
    job = sampler.run(circuits, shots=shots)
    job_id = job.job_id()
    print(f"      job_id={job_id}")

    t0 = time.perf_counter()
    result = job.result()
    wallclock = time.perf_counter() - t0
    result_ts = hru.utc_now_iso()

    counts_list = [result[i].data.meas.get_counts() for i in range(len(result))]

    raw_result_path = os.path.join(run_dir, "raw_results", f"job_{job_id}_counts.json")
    with open(raw_result_path, "w") as f:
        json.dump({"job_id": job_id, "batch_index": batch_index,
                   "local_submit_timestamp_utc": submit_ts, "local_result_timestamp_utc": result_ts,
                   "submit_to_result_wallclock_seconds": wallclock,
                   "counts_list": [dict(c) for c in counts_list]}, f, indent=2, default=str)
    print(f"      DONE in {wallclock:.2f}s -- saved to '{raw_result_path}'")
    return counts_list


def submit_stage(sampler, backend, stage_name, circuits, shots, opt_level, hardware_batch_size, run_dir):
    """Adds measurements, transpiles, splits into hardware-sized batches, and
    submits each batch for real. Returns the full list of decoded counts
    dicts, one per circuit, in input order."""
    from qiskit import transpile

    if not circuits:
        print(f"[{stage_name}] 0 circuits, skipping")
        return []
    measured = [ibmb.add_measurements(qc) for qc in circuits]
    transpiled = transpile(measured, backend=backend, optimization_level=opt_level)
    print(f"[{stage_name}] {len(transpiled)} circuits transpiled, submitting in "
          f"batches of {hardware_batch_size}")

    all_counts = []
    for batch_index, batch in enumerate(chunk_list(transpiled, hardware_batch_size)):
        all_counts.extend(submit_and_harvest_batch(sampler, batch, shots, run_dir, batch_index))
    return all_counts


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_config = load_checkpoint_config(args.checkpoint_config)
    detectors = resolve_detector_list(args.detectors)

    sample_info = hru.select_fixed_test_samples(args.dataset, args.data_dir, args.id_class,
                                                 args.test_scale, seed=42, target_n_id=args.target_n_id)
    print(f"Fixed test set: dataset={args.dataset} id_class={args.id_class} test_scale={args.test_scale} "
          f"n_id={sample_info['n_id']} n_ood={sample_info['n_ood']}")
    images = load_preprocessed_images(args.dataset, args.data_dir, sample_info["indices"])

    for d in [d for d in detectors if d in ITERATIVE_UNSUPPORTED]:
        detectors.remove(d)
        if args.allow_expensive_iterative_gans:
            print(f"{d}: --allow_expensive_iterative_gans given, but iterative hardware inference "
                  f"is not implemented -- skipping anyway.")
        else:
            print(f"{d}: hardware_supported=False (500-iteration per-sample latent optimization -> "
                  f"~1e4 circuit evals/sample, too expensive to run) -- skipping.")

    for d in [d for d in detectors if d in DENSITY_OWN_STAGE]:
        print(f"{d}: the shared DRNN embedding circuit will be submitted, but {d}'s OWN extra "
              f"circuit stage needs that embedding's real hardware output as input -- submitting "
              f"it requires a second submission phase that is not implemented here.")

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService()

    for backend_name in resolve_backend_list(args):
        print(f"\n{'#'*70}\n# BACKEND: {backend_name}\n{'#'*70}")
        try:
            backend = service.backend(backend_name)
        except Exception as e:
            print(f"FAILED to load backend '{backend_name}': {type(e).__name__}: {e} -- skipping this backend, "
                  f"other backends/results are unaffected.")
            continue
        sampler = SamplerV2(mode=backend)

        needs_shared_drnn = any(d in SHARED_DRNN_GROUP for d in detectors)
        if needs_shared_drnn:
            circuits, ckpt_path = build_shared_drnn_batch(checkpoint_config, args.dataset, images)
            print(f"shared DRNN extractor checkpoint: {ckpt_path}")
            run_dir = hru.run_dir_for(args.output_dir, args.dataset, backend_name, "shared_drnn_group")
            submit_stage(sampler, backend, "shared_drnn_embedding", circuits, args.shots,
                         args.transpiler_optimization_level, args.hardware_batch_size, run_dir)

        if "QSVDD" in detectors:
            circuits, ckpt_path = build_qsvdd_batch(checkpoint_config, args.dataset, images)
            print(f"QSVDD checkpoint: {ckpt_path}")
            run_dir = hru.run_dir_for(args.output_dir, args.dataset, backend_name, "QSVDD")
            submit_stage(sampler, backend, "QSVDD_own_extractor", circuits, args.shots,
                         args.transpiler_optimization_level, args.hardware_batch_size, run_dir)

        for detector in ("QAE", "QVAE", "QGANomaly"):
            if detector in detectors:
                stages, ckpt_path = build_fastvqc_detector_batch(checkpoint_config, detector, args.dataset, images)
                print(f"{detector} checkpoint: {ckpt_path}")
                run_dir = hru.run_dir_for(args.output_dir, args.dataset, backend_name, detector)
                for stage_name, circuits in stages.items():
                    submit_stage(sampler, backend, f"{detector}_{stage_name}", circuits, args.shots,
                                 args.transpiler_optimization_level, args.hardware_batch_size, run_dir)

    print("\nAll requested stages submitted. Raw measurement counts are saved under "
          f"'{args.output_dir}/runs/<dataset>/<backend>/<detector>/raw_results/'.")


if __name__ == "__main__":
    main()
