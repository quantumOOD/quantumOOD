#!/usr/bin/env bash
# RQ2 (quantum) and RQ3 (classical): natural-shift training+eval for all 12
# quantum and 12 classical OOD detectors, then adversarial-shift generation
# and evaluation for the same detectors, via quantum_OOD_detectors.py and
# classical_OOD_detectors.py.

set -e
cd "$(dirname "$0")"

LOG_DIR="pipeline_logs"
mkdir -p "$LOG_DIR"

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
ATTACKS=(fgsm pgd spsa salt_pepper)

# ============================================================================
# Part 1 -- natural-shift training+eval, quantum detectors (RQ2)
# ============================================================================
for dataset in "${DATASETS[@]}"; do
    for norm_cls in "${CLASSES[@]}"; do
        run "qae_${dataset}_${norm_cls}" quantum_OOD_detectors.py qae \
            --dataset "$dataset" --target_class "$norm_cls" --model both --save_dir results/qae_nq8_cry
        run "qvae_${dataset}_${norm_cls}" quantum_OOD_detectors.py qvae \
            --dataset "$dataset" --target_class "$norm_cls" --model both --save_dir results/qvae_nq8_cry
        run "gan_${dataset}_${norm_cls}" quantum_OOD_detectors.py gan \
            --dataset "$dataset" --target_class "$norm_cls" --gan_type both --save_dir results/quantum_gan_ood_nq8_cry
        run "ganomaly_${dataset}_${norm_cls}" quantum_OOD_detectors.py ganomaly \
            --dataset "$dataset" --target_class "$norm_cls" --model both --save_dir results/qganomaly_nq8_cry
    done
done
# distance/density (QKNN/QMean/QMedoids/QSVDD, DMKDE-mixed/IndepGaussian/MVGaussian) reuse the
# quantum feature extractor trained in rq1_feature_extractor/run_rq1.sh -- see that script.

# ============================================================================
# Part 2 -- natural-shift training+eval, classical detectors (RQ3)
# ============================================================================
for dataset in "${DATASETS[@]}"; do
    for norm_cls in "${CLASSES[@]}"; do
        run "classical_sae_${dataset}_${norm_cls}" classical_OOD_detectors.py sae \
            --dataset "$dataset" --norm_cls "$norm_cls" --output_dir outputs/classical_sae \
            --results_csv "outputs/classical_sae/classical_sae_${dataset}_normcls${norm_cls}_results.csv"
        run "classical_vae_${dataset}_${norm_cls}" classical_OOD_detectors.py vae \
            --dataset "$dataset" --norm_cls "$norm_cls" --output_dir outputs/classical_vae \
            --results_csv "outputs/classical_vae/classical_vae_${dataset}_normcls${norm_cls}_results.csv"
        run "classical_gan_${dataset}_${norm_cls}" classical_OOD_detectors.py gan \
            --dataset "$dataset" --target_class "$norm_cls" --gan_type both --save_dir results/classical_gan_mnist
        run "classical_ganomaly_${dataset}_${norm_cls}" classical_OOD_detectors.py ganomaly \
            --dataset "$dataset" --target_class "$norm_cls" --save_dir results/classical_ganomaly
        run "classical_density_${dataset}_${norm_cls}" classical_OOD_detectors.py density \
            --dataset "$dataset" --norm_cls "$norm_cls" --output_dir outputs/classical_density \
            --results_csv "outputs/classical_density/classical_density_${dataset}_normcls${norm_cls}_results.csv"
        run "classical_distance_${dataset}_${norm_cls}" classical_OOD_detectors.py distance \
            --dataset "$dataset" --norm_cls "$norm_cls" --settings native --final_backbones resnet18 \
            --output_dir outputs/classical_projection \
            --results_csv "outputs/classical_projection/classical_native_${dataset}_normcls${norm_cls}_results.csv"
    done
done

# ============================================================================
# Part 3 -- adversarial image generation (shared by RQ2 and RQ3's adversarial evaluation)
# ============================================================================
for dataset in "${DATASETS[@]}"; do
    for norm_cls in "${CLASSES[@]}"; do
        run "generate_adversarial_${dataset}_${norm_cls}" generate_adversarial_images.py \
            --dataset "$dataset" --norm_cls "$norm_cls" --attacks "${ATTACKS[@]}" \
            --output_dir outputs/adversarial_images
    done
done

# ============================================================================
# Part 4 -- adversarial-shift evaluation (reuses the checkpoints trained in Parts 1-2;
# --adversarial_dir switches each subcommand from natural-shift training into
# adversarial-split evaluation of an already-trained --checkpoint_dir)
# ============================================================================
for dataset in "${DATASETS[@]}"; do
    for norm_cls in "${CLASSES[@]}"; do
        for attack in "${ATTACKS[@]}"; do
            run "qae_adv_${dataset}_${norm_cls}_${attack}" quantum_OOD_detectors.py qae \
                --dataset "$dataset" --norm_cls "$norm_cls" --checkpoint_dir results/qae_nq8_cry \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
            run "qvae_adv_${dataset}_${norm_cls}_${attack}" quantum_OOD_detectors.py qvae \
                --dataset "$dataset" --norm_cls "$norm_cls" --checkpoint_dir results/qvae_nq8_cry \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
            run "gan_adv_${dataset}_${norm_cls}_${attack}" quantum_OOD_detectors.py gan \
                --dataset "$dataset" --norm_cls "$norm_cls" --checkpoint_dir results/quantum_gan_ood_nq8_cry \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
            run "ganomaly_adv_${dataset}_${norm_cls}_${attack}" quantum_OOD_detectors.py ganomaly \
                --dataset "$dataset" --norm_cls "$norm_cls" --checkpoint_dir results/qganomaly_nq8_cry \
                --adversarial_dir outputs/adversarial_images --attack "$attack"

            run "classical_sae_adv_${dataset}_${norm_cls}_${attack}" classical_OOD_detectors.py sae \
                --dataset "$dataset" --norm_cls "$norm_cls" --checkpoint_dir outputs/classical_sae \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
            run "classical_vae_adv_${dataset}_${norm_cls}_${attack}" classical_OOD_detectors.py vae \
                --dataset "$dataset" --norm_cls "$norm_cls" --checkpoint_dir outputs/classical_vae \
                --original_results_csv "outputs/classical_vae/classical_vae_${dataset}_normcls${norm_cls}_results.csv" \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
            run "classical_gan_adv_${dataset}_${norm_cls}_${attack}" classical_OOD_detectors.py gan \
                --dataset "$dataset" --norm_cls "$norm_cls" --checkpoint_dir results/classical_gan_mnist \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
            run "classical_ganomaly_adv_${dataset}_${norm_cls}_${attack}" classical_OOD_detectors.py ganomaly \
                --dataset "$dataset" --norm_cls "$norm_cls" --checkpoint_dir results/classical_ganomaly \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
            run "classical_density_adv_${dataset}_${norm_cls}_${attack}" classical_OOD_detectors.py density \
                --dataset "$dataset" --norm_cls "$norm_cls" \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
            run "classical_distance_adv_${dataset}_${norm_cls}_${attack}" classical_OOD_detectors.py distance \
                --dataset "$dataset" --norm_cls "$norm_cls" \
                --original_results_csv "outputs/classical_projection/classical_native_${dataset}_normcls${norm_cls}_results.csv" \
                --adversarial_dir outputs/adversarial_images --attack "$attack"
        done
    done
done

echo "All main-experiment runs completed."
