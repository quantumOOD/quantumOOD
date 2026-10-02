#!/usr/bin/env bash
# RQ4: ideal-simulation sweep at the real-hardware test scale (test_scale=0.1,
# id_class=0, seeds 0-4), followed by the real IBM QPU submission driver, so
# the two sides are directly comparable. Real job submission requires an IBM
# Quantum account configured locally (see the top-level README) and spends
# real QPU time.

set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

DATASETS=(mnist fashion_mnist)
NORM_CLS=0
SEEDS=(0 1 2 3 4)
TEST_SCALE=0.1
BACKENDS=(ibm_fez ibm_marrakesh ibm_kingston)

LOG_DIR="pipeline_logs/rq4"
RESULT_DIR="outputs/ideal_sim_eval"
EXTRACTOR_DIR="../rq1_feature_extractor/outputs/qcl_family_quanforge"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

run() {
    local name="$1"; shift
    local log_file="${LOG_DIR}/${name}.txt"
    echo "============================================================"
    echo "Starting: ${name}"
    echo "============================================================"
    python3 "$@" > "$log_file" 2>&1
    status=$?
    if [ $status -eq 0 ]; then echo "Finished: ${name}"; else echo "FAILED: ${name} (exit $status) -- see ${log_file}"; fi
}

# ---- ideal-simulation baseline, matched to the hardware test scale ----
for dataset in "${DATASETS[@]}"; do
    run "distance_${dataset}" ../main_experiments/quantum_OOD_detectors.py distance \
        --dataset "$dataset" --norm_cls "$NORM_CLS" --setting 1 \
        --circuit DRNN --feature_loss compact --extractor_seed 0 \
        --seeds "${SEEDS[@]}" --test_data_scale "$TEST_SCALE" \
        --extractor_dir "$EXTRACTOR_DIR" --output_dir "$RESULT_DIR"

    run "density_${dataset}" ../main_experiments/quantum_OOD_detectors.py density \
        --dataset "$dataset" --norm_cls "$NORM_CLS" --setting 1 \
        --circuit DRNN --feature_loss compact --extractor_seed 0 \
        --seeds "${SEEDS[@]}" --test_data_scale "$TEST_SCALE" --no_verify_mvg \
        --extractor_dir "$EXTRACTOR_DIR" --output_dir "$RESULT_DIR"

    run "qae_${dataset}" qae_ideal_eval.py \
        --dataset "$dataset" --norm_cls "$NORM_CLS" --test_scale "$TEST_SCALE" \
        --checkpoint_dir "../main_experiments/results/qae_nq8_cry" --seeds "${SEEDS[@]}" --output_dir "$RESULT_DIR"

    run "qvae_${dataset}" qvae_ideal_eval.py \
        --dataset "$dataset" --norm_cls "$NORM_CLS" --test_scale "$TEST_SCALE" \
        --checkpoint_dir "../main_experiments/results/qvae_nq8_cry" --seeds "${SEEDS[@]}" --output_dir "$RESULT_DIR"

    run "ganomaly_${dataset}" ganomaly_ideal_eval.py \
        --dataset "$dataset" --norm_cls "$NORM_CLS" --test_scale "$TEST_SCALE" \
        --checkpoint_dir "../main_experiments/results/qganomaly_nq8_cry" --seeds "${SEEDS[@]}" --output_dir "$RESULT_DIR"

    # Q-AnoGAN/QWGAN-GP: effectiveness only -- too expensive to run on real hardware
    # (500-step per-sample latent optimization), excluded from the submission driver below.
    run "gan_${dataset}" quantum_gan_ideal_eval.py \
        --dataset "$dataset" --norm_cls "$NORM_CLS" --test_scale "$TEST_SCALE" \
        --checkpoint_dir "../main_experiments/results/quantum_gan_ood_nq8_cry" --seeds "${SEEDS[@]}" --output_dir "$RESULT_DIR"
done

# ---- real IBM QPU submission (see real_device_all_ood_ibm.py --help) ----
for dataset in "${DATASETS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        run "hardware_${dataset}_${backend}" real_device_all_ood_ibm.py \
            --dataset "$dataset" --id_class "$NORM_CLS" --backend "$backend" \
            --test_scale "$TEST_SCALE" --output_dir outputs/ibm_hardware_ood
    done
done

echo "All RQ4 runs completed. Results under: ${RESULT_DIR}/ and outputs/ibm_hardware_ood/"
