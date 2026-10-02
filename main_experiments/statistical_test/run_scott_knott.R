# Runs the Scott-Knott ESD significance test on every master CSV extract_results.py produces.

options(expressions = 500000)
Sys.setenv("R_CSTACK_NOCHECK" = "true")
if (!requireNamespace("ScottKnottESD", quietly = TRUE)) install.packages("ScottKnottESD")
suppressMessages(library(ScottKnottESD))

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "."

DATASETS <- c("mnist", "fashion_mnist")
CLASSES <- 0:9
ATTACKS <- c("fgsm", "pgd", "spsa", "salt_pepper")
SEVERITY_TAG <- "mild"

NATURAL_COLUMNS <- c("Knn", "Mean", "Medoids", "SVDD", "DMKDE-Mixed", "IndepGaussian",
                      "MVGaussian", "AE", "VAE", "AnoGAN", "GANomaly", "WGAN-GP")

ADVERSARIAL_FAMILIES <- list(
  classical_distance       = c("DeepKNN", "Deep-Mean", "Deep-Medoids", "DeepSVDD"),
  classical_density        = c("DMKDE-mixed", "IndepGaussian", "MVGaussian"),
  classical_reconstruction = c("SAE-Recon", "VAE-Recon"),
  classical_gan            = c("Classical-AnoGAN", "C-GANomaly", "Classical-WGAN-GP"),
  quantum_distance         = c("QKNN", "QMean", "QMedoids", "QSVDD"),
  quantum_density          = c("DMKDE-mixed", "IndepGaussian", "MVGaussian"),
  quantum_reconstruction   = c("QAE-Recon", "QVAE-Recon"),
  quantum_gan              = c("Q-AnoGAN", "Q-GANomaly", "QWGAN-GP"),
  classical_all = c("DeepKNN", "Deep-Mean", "Deep-Medoids", "DeepSVDD",
                     "DMKDE-mixed", "IndepGaussian", "MVGaussian",
                     "SAE-Recon", "VAE-Recon",
                     "Classical-AnoGAN", "C-GANomaly", "Classical-WGAN-GP"),
  quantum_all = c("QKNN", "QMean", "QMedoids", "QSVDD",
                   "DMKDE-mixed", "IndepGaussian", "MVGaussian",
                   "QAE-Recon", "QVAE-Recon",
                   "Q-AnoGAN", "Q-GANomaly", "QWGAN-GP")
)

# Writes group/avg CSVs for one (column_order, row_labels, group_matrix, avg_matrix) result.
write_result <- function(row_labels, group_matrix, avg_matrix, column_order, group_out, avg_out) {
  if (length(row_labels) == 0) {
    cat(sprintf("skip: no complete rows for '%s'\n", group_out))
    return(invisible())
  }
  rownames(group_matrix) <- row_labels
  rownames(avg_matrix) <- row_labels
  colnames(group_matrix) <- column_order
  colnames(avg_matrix) <- column_order
  write.csv(data.frame(row = row_labels, group_matrix, check.names = FALSE), paste0(group_out, ".csv"), row.names = FALSE)
  write.csv(data.frame(row = row_labels, avg_matrix, check.names = FALSE), paste0(avg_out, ".csv"), row.names = FALSE)
  cat(sprintf("-> '%s.csv', '%s.csv' (%d rows)\n", group_out, avg_out, length(row_labels)))
}

# One Scott-Knott comparison per (dataset, class) row, 5 seeds per cell. Used
# for the natural-shift tables and the per-setting/per-family adversarial tables.
run_by_class <- function(column_order, master_path, group_out, avg_out) {
  if (!file.exists(master_path)) {
    cat(sprintf("skip: '%s' not found\n", master_path))
    return(invisible())
  }
  master <- read.csv(master_path, stringsAsFactors = FALSE)
  for (metric_name in c("auc_roc", "auc_pr")) {
    metric_df <- master[master$metric == metric_name, ]
    row_labels <- character(0)
    group_matrix <- matrix(nrow = 0, ncol = length(column_order))
    avg_matrix <- matrix(nrow = 0, ncol = length(column_order))

    for (dataset in DATASETS) {
      for (norm_cls in CLASSES) {
        row_df <- metric_df[metric_df$dataset == dataset & metric_df$norm_cls == norm_cls, ]
        if (nrow(row_df) == 0) next  # not run yet for this class -- skip, don't error

        mat <- sapply(column_order, function(col) {
          vals <- row_df$value[row_df$column == col]
          stopifnot(length(vals) == 5)
          vals
        })
        colnames(mat) <- column_order

        result <- tryCatch(sk_esd(mat, version = "p"), error = function(e) e)
        if (inherits(result, "error")) {
          # degenerate input (e.g. two+ columns tied/zero-variance across all
          # 5 seeds) can crash sk_esd()'s recursive partitioning -- skip this
          # row rather than error the whole run.
          cat(sprintf("skip %s_%s (sk_esd failed: %s)\n", dataset, norm_cls, conditionMessage(result)))
          next
        }
        # sk_esd() sanitizes column names via make.names() (e.g.
        # "DMKDE-Mixed" -> "DMKDE.Mixed"), so result$groups must be looked up
        # by the SAME sanitized names, not the originals.
        group_row <- result$groups[make.names(column_order)]
        avg_row <- colMeans(mat)[column_order]

        row_labels <- c(row_labels, paste0(dataset, "_", norm_cls))
        group_matrix <- rbind(group_matrix, group_row)
        avg_matrix <- rbind(avg_matrix, avg_row)
      }
    }
    write_result(row_labels, group_matrix, avg_matrix, column_order,
                  paste0(group_out, "_", metric_name), paste0(avg_out, "_", metric_name))
  }
}

# One Scott-Knott comparison per (dataset, attack) row, pooling all classes x
# seeds into one cell (up to 50 values). Used for the by-attack tables.
run_by_attack <- function(column_order, master_path_fn, group_out, avg_out) {
  for (metric_name in c("auc_roc", "auc_pr")) {
    row_labels <- character(0)
    group_matrix <- matrix(nrow = 0, ncol = length(column_order))
    avg_matrix <- matrix(nrow = 0, ncol = length(column_order))

    for (dataset in DATASETS) {
      for (attack in ATTACKS) {
        master_path <- master_path_fn(attack)
        if (!file.exists(master_path)) next
        master <- read.csv(master_path, stringsAsFactors = FALSE)
        sub <- master[master$metric == metric_name & master$dataset == dataset, ]
        if (nrow(sub) == 0) next

        seeds_sorted <- sort(unique(sub$seed))
        # class varies fastest within a fixed seed: class 0..9 (seed 1), class 0..9 (seed 2), ...
        combos <- expand.grid(norm_cls = CLASSES, seed = seeds_sorted)
        combo_key <- paste(combos$seed, combos$norm_cls)

        mat <- sapply(column_order, function(col) {
          col_sub <- sub[sub$column == col, ]
          col_key <- paste(col_sub$seed, col_sub$norm_cls)
          vals <- col_sub$value[match(combo_key, col_key)]
          stopifnot(!any(is.na(vals)))
          vals
        })
        colnames(mat) <- column_order

        result <- tryCatch(sk_esd(mat, version = "p"), error = function(e) e)
        if (inherits(result, "error")) {
          cat(sprintf("skip %s_%s (sk_esd failed: %s)\n", dataset, attack, conditionMessage(result)))
          next
        }
        group_row <- result$groups[make.names(column_order)]
        avg_row <- colMeans(mat)[column_order]

        row_labels <- c(row_labels, paste0(dataset, "_", attack))
        group_matrix <- rbind(group_matrix, group_row)
        avg_matrix <- rbind(avg_matrix, avg_row)
      }
    }
    write_result(row_labels, group_matrix, avg_matrix, column_order,
                  paste0(group_out, "_", metric_name), paste0(avg_out, "_", metric_name))
  }
}

# ---- Scenario 1: natural shifts, all 12 detectors, classical/quantum/quantum_8qubit ----
for (table_name in c("classical", "quantum", "quantum_8qubit")) {
  run_by_class(NATURAL_COLUMNS,
               file.path(script_dir, paste0("master_", table_name, ".csv")),
               file.path(script_dir, paste0("group_", table_name)),
               file.path(script_dir, paste0("avg_", table_name)))
}

# ---- Scenario 2: adversarial shifts, every detector family, per attack, by class ----
for (family in names(ADVERSARIAL_FAMILIES)) {
  column_order <- ADVERSARIAL_FAMILIES[[family]]
  if (length(column_order) < 2) {
    cat(sprintf("skip %s: fewer than 2 detectors, NPSK comparison not meaningful\n", family))
    next
  }
  for (setting in ATTACKS) {
    run_by_class(column_order,
                 file.path(script_dir, paste0("master_adversarial_", SEVERITY_TAG, "_", family, "_", setting, ".csv")),
                 file.path(script_dir, paste0("group_adversarial_", SEVERITY_TAG, "_", family, "_", setting)),
                 file.path(script_dir, paste0("avg_adversarial_", SEVERITY_TAG, "_", family, "_", setting)))
    # classical_distance is also written without a family tag by extract_results.py --
    # keep that naming for backward-compatible output alongside the family-tagged one.
    if (family == "classical_distance") {
      run_by_class(column_order,
                   file.path(script_dir, paste0("master_adversarial_", SEVERITY_TAG, "_", setting, ".csv")),
                   file.path(script_dir, paste0("group_adversarial_", SEVERITY_TAG, "_", setting)),
                   file.path(script_dir, paste0("avg_adversarial_", SEVERITY_TAG, "_", setting)))
    }
  }
}

# ---- Scenario 3: adversarial shifts, every detector family, pooled by attack type ----
for (family in names(ADVERSARIAL_FAMILIES)) {
  column_order <- ADVERSARIAL_FAMILIES[[family]]
  if (length(column_order) < 2) next
  run_by_attack(column_order,
                function(attack) file.path(script_dir, paste0("master_adversarial_", SEVERITY_TAG, "_", family, "_", attack, ".csv")),
                file.path(script_dir, paste0("group_adversarial_", SEVERITY_TAG, "_", family, "_by_attack")),
                file.path(script_dir, paste0("avg_adversarial_", SEVERITY_TAG, "_", family, "_by_attack")))
  if (family == "classical_distance") {
    run_by_attack(column_order,
                  function(attack) file.path(script_dir, paste0("master_adversarial_", SEVERITY_TAG, "_", attack, ".csv")),
                  file.path(script_dir, paste0("group_adversarial_", SEVERITY_TAG, "_by_attack")),
                  file.path(script_dir, paste0("avg_adversarial_", SEVERITY_TAG, "_by_attack")))
  }
}
