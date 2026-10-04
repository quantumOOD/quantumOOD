# Is Quantum Computing Ready for Image Out-of-Distribution Detection?

## Abstract

Deep learning has established mature Out-of-Distribution (OOD) detectors across different methodological families (e.g., distance-based, density-based, reconstruction-based, and GAN-based), whereas using quantum computing for such tasks remains at an early stage. Existing studies that apply quantum computing to OOD detection mainly focus on tabular and time-series data, and employ heterogeneous detectors and evaluation protocols. This fragmentation makes it difficult to systematically compare quantum OOD detectors from different families and assess their advantages over classical counterparts. In particular, quantum OOD detection for image data remains largely unexplored, with neither an established suite of quantum detectors nor a unified evaluation framework.

In this paper, we present the first unified and systematic study of quantum–classical OOD detection for image data, covering 12 detectors across four methodological families, with each detector implemented in both quantum and classical forms. We evaluate them on MNIST and Fashion-MNIST under natural and adversarial OOD shifts, using both ideal quantum simulation and three IBM quantum hardware platforms. Our results demonstrate the feasibility of quantum OOD detection under ideal simulation, although classical detectors generally achieve better performance, with the performance gap varying substantially across OOD scenarios and detector families. On real quantum hardware, most detectors experience considerable performance degradation, while reconstruction-based detectors remain comparatively robust. Our analysis suggests that this robustness stems from their larger ID–OOD score separation, which better withstands the perturbations introduced by hardware noise. These findings provide scenario-oriented and hardware-aware guidance for the design and deployment of quantum OOD detectors.

## Setup

```bash
conda env create -f environment.yml
conda activate quantum-ood
```

This project uses both Qiskit (hardware-facing circuits, IBM backend execution) and PennyLane (feature-extractor training) side by side. MNIST and Fashion-MNIST are downloaded automatically by `torchvision` on first run into `./data`.

Running the real-hardware scripts requires an IBM Quantum account configured locally:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<your token>")
```

No credentials are stored in this repository.

## Repository structure

```
rq1_feature_extractor/              RQ1.1/RQ1.2: quantum feature-extractor and VQC-config ablations
    qcl_family_quanforge_ood.py     QCL/QCNN/HCQC/DRNN quantum feature extractors
    qcl_family_quanforge_km_tuning.py  k/m hyperparameter tuning for the distance detectors
    run_quanforge_pipeline_mnist.py end-to-end training/tuning/evaluation pipeline for the extractor family
    run_rq1.sh                      launcher sweeping datasets and classes

main_experiments/                   RQ2 (quantum) and RQ3 (classical): ideal-simulation detectors
    quantum_OOD_detectors.py        all 12 quantum detectors (subcommands: qae, qvae, gan, ganomaly, distance, density)
    classical_OOD_detectors.py      all 12 classical detectors (subcommands: sae, vae, gan, ganomaly, distance, density)
    scoring_pipelines.py            shared data loading, feature extraction, and evaluation utilities
    adversarial_classifier.py       surrogate classifier used as the attack target
    adversarial_attacks.py          FGSM, PGD, SPSA, Salt-and-Pepper attack implementations
    build_adversarial_ood_split.py  assembles adversarial OOD test sets from the pools
    statistical_test/               Scott-Knott ESD significance testing: run_scott_knott.R runs the test
    run_main_experiments.sh         launcher sweeping datasets, classes, and attacks

rq4_hardware/                       RQ4: real IBM quantum hardware validation
    qae_ideal_eval.py / qvae_ideal_eval.py / quantum_gan_ideal_eval.py / ganomaly_ideal_eval.py
                                     ideal-simulation evaluation at the real-hardware test scale
    real_device_all_ood_ibm.py      submits and evaluates detectors on real IBM quantum hardware
    hardware_result_utils.py        sample-selection and result-saving utilities for the hardware driver
    ibm_ood_backend.py              builds hardware circuits and checks them against simulated ground truth
    run_rq4.sh                      launcher tying the ideal-simulation and hardware steps together
```

Each quantum/classical detector file is a single script with one subcommand per detector family; run `python quantum_OOD_detectors.py <subcommand> --help` (or the classical equivalent) to see that family's full option list. Every subcommand trains and evaluates on the natural OOD split by default; passing `--adversarial_dir outputs/adversarial_images` switches it to evaluating an already-trained `--checkpoint_dir`/`--checkpoint` on the adversarial split instead.

## Reproducing the results

### RQ1.1 — Selection of quantum feature extractors

Ablates training objective (random/compact/cosine/vicreg) x circuit (QCL/QCNN/HCQC/DRNN):

```bash
cd rq1_feature_extractor
python run_quanforge_pipeline_mnist.py --dataset mnist --circuits QCNN DRNN HCQC --feature_losses random cosine vicreg compact
python qcl_family_quanforge_ood.py --dataset mnist --circuit DRNN --feature_loss compact --epochs 20
```

or the full run: `bash run_rq1.sh`.

### RQ1.2 — Configuration of VQCs

Ablates rotation gate (RX/RY/RZ/RX+RZ) x entangling gate (CNOT/CRY) for the reconstruction/GAN detectors:

```bash
cd main_experiments
python quantum_OOD_detectors.py qae --dataset mnist --rot_gate RX+RZ --entangle_gate CRY --run_all_id --model both
```

### RQ2 — Effectiveness of quantum OOD detectors

Natural shifts (ideal simulation), from `main_experiments/`:

```bash
python quantum_OOD_detectors.py qae --dataset mnist --run_all_id --model both
python quantum_OOD_detectors.py qvae --dataset fashion_mnist --run_all_id --model both
python quantum_OOD_detectors.py gan --dataset mnist --run_all_id --gan_type both
python quantum_OOD_detectors.py ganomaly --dataset mnist --run_all_id --model both
python quantum_OOD_detectors.py distance --dataset mnist --circuit DRNN --detectors QKNN QMean QMedoids QSVDD
python quantum_OOD_detectors.py density --dataset mnist --circuit DRNN --detectors DMKDE-mixed IndepGaussian MVGaussian
```

Adversarial shifts:

```bash
python generate_adversarial_images.py --dataset mnist --norm_cls 0 --attacks fgsm pgd spsa salt_pepper
python build_adversarial_ood_split.py --dataset mnist --norm_cls 0 --attack fgsm
python quantum_OOD_detectors.py qae --dataset mnist --norm_cls 0 --checkpoint_dir results/qae_nq8_cry \
    --adversarial_dir outputs/adversarial_images --attack fgsm
```

(the other quantum subcommands take the same `--checkpoint_dir`/`--adversarial_dir`/`--attack` pattern, one of `fgsm`/`pgd`/`spsa`/`salt_pepper`), or the full run: `bash run_main_experiments.sh`. `Rscript run_scott_knott.R` runs the Scott-Knott ESD test.

### RQ3 — Comparison with classical OOD detectors

Mirrors RQ2 with the classical counterparts, from `main_experiments/`:

```bash
python classical_OOD_detectors.py sae --dataset mnist --norm_cls 0
python classical_OOD_detectors.py vae --dataset mnist --norm_cls 0
python classical_OOD_detectors.py gan --dataset mnist --target_class 0 --gan_type both
python classical_OOD_detectors.py ganomaly --dataset mnist --target_class 0
python classical_OOD_detectors.py density --dataset mnist --norm_cls 0
python classical_OOD_detectors.py distance --dataset mnist --norm_cls 0 --settings both --detectors DeepKNN DeepSVDD
```

and the same `--checkpoint_dir`/`--adversarial_dir`/`--attack` pattern for the adversarial counterparts, or the full run in `run_main_experiments.sh`.

### RQ4 — Real hardware validation

Ideal-simulation baseline matched to the hardware test scale, from `rq4_hardware/`:

```bash
python qae_ideal_eval.py --dataset mnist --norm_cls 0 --checkpoint_dir ../main_experiments/results/qae_nq8_cry
```

(and `qvae_ideal_eval.py` / `quantum_gan_ideal_eval.py` / `ganomaly_ideal_eval.py`), or `bash run_rq4.sh`.

Real IBM hardware submission, given a trained checkpoint for each detector:

```bash
python real_device_all_ood_ibm.py --dataset mnist --backend ibm_fez
```

This builds each detector's circuits from the real test images and trained checkpoint weights, transpiles them for the chosen backend, submits them, and saves the raw measurement counts under `<output_dir>/runs/<dataset>/<backend>/<detector>/raw_results/`. It spends QPU time on the configured IBM account. `--checkpoint_config` takes a JSON file mapping each detector to its checkpoint path, for example:

```json
{
  "shared_drnn_extractor": {"circuit": "DRNN", "n_qubits": 8,
    "mnist": {"checkpoint_path": "../rq1_feature_extractor/outputs/qcl_family_quanforge/.../checkpoint.pt"}},
  "QAE": {"circuit": "FastVQC", "n_qubits": 8, "n_layers": 3,
    "mnist": {"checkpoint_path": "../main_experiments/results/qae_nq8_cry/.../qae_model.pt"}}
}
```
