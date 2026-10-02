#!/usr/bin/env bash
# RQ1.1/RQ1.2: trains and tunes the quantum feature-extractor family (QCL/QCNN/HCQC/DRNN)
# across feature-learning objectives, then sweeps the distance/density detectors on top.

set -e
cd "$(dirname "$0")"

LOG_DIR="pipeline_logs"
RESULT_DIR="outputs/qcl_family_quanforge"
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

DATASETS=(mnist fashion_mnist)
CLASSES=(0 1 2 3 4 5 6 7 8 9)

# ---- RQ1.1: feature-learning objective ablation (random/compact/cosine/vicreg) x circuit ----
for dataset in "${DATASETS[@]}"; do
    for norm_cls in "${CLASSES[@]}"; do
        run "pipeline_${dataset}_${norm_cls}" run_quanforge_pipeline_mnist.py \
            --dataset "$dataset" --norm_cls "$norm_cls" --setting 1 \
            --circuits QCNN DRNN HCQC --feature_losses random cosine vicreg compact \
            --num_seeds 5 \
            --results_csv "${RESULT_DIR}/full_pipeline_${dataset}_normcls${norm_cls}_results.csv"
    done
done

# ---- RQ2 ideal-simulation drivers that depend on the extractor trained above:
# distance (QKNN/QMean/QMedoids/QSVDD) and density (DMKDE/IndepGaussian/MVGaussian)
# detectors, both implemented as subcommands of quantum_OOD_detectors.py ----
for dataset in "${DATASETS[@]}"; do
    for norm_cls in "${CLASSES[@]}"; do
        run "distance_${dataset}_${norm_cls}" ../main_experiments/quantum_OOD_detectors.py distance \
            --dataset "$dataset" --norm_cls "$norm_cls" --setting 1 \
            --circuit DRNN --feature_loss compact \
            --extractor_dir "$RESULT_DIR" --output_dir "$RESULT_DIR"

        run "density_${dataset}_${norm_cls}" ../main_experiments/quantum_OOD_detectors.py density \
            --dataset "$dataset" --norm_cls "$norm_cls" --setting 1 \
            --circuit DRNN --feature_loss compact \
            --extractor_dir "$RESULT_DIR" --output_dir outputs/density_quanforge
    done
done

echo "All RQ1 runs completed. Results under: ${RESULT_DIR}/"
