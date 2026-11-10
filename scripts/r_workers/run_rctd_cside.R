#!/usr/bin/env Rscript
# Standalone Background R Subprocess Worker for spacexr (RCTD & CSIDE)
# Decoupled from Python memory via temporary Parquet/Matrix IPC files.

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
  library(Matrix)
  library(arrow)
})

option_list <- list(
  make_option(c("-w", "--workdir"), type = "character", default = NULL,
              help = "Path to temporary IPC directory containing mtx and parquet inputs",
              metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

if (is.null(opt$workdir)) {
  print_help(opt_parser)
  stop("Error: --workdir must be specified.", call. = FALSE)
}

workdir <- normalizePath(opt$workdir, mustWork = TRUE)
message(sprintf("[R-Worker] Initializing spacexr RCTD worker in: %s", workdir))

# 1. Read JSON parameters
param_file <- file.path(workdir, "params.json")
if (!file.exists(param_file)) {
  stop(sprintf("Parameters file not found at %s", param_file), call. = FALSE)
}
params <- jsonlite::fromJSON(param_file)

# 2. Check and load spacexr
if (!requireNamespace("spacexr", quietly = TRUE)) {
  stop("Bioconductor / GitHub package 'spacexr' is not installed in the R environment.", call. = FALSE)
}
library(spacexr)

# 3. Read Spatial Data
spatial_counts_file <- file.path(workdir, "spatial_counts.mtx")
spatial_coords_file <- file.path(workdir, "spatial_coords.parquet")
spatial_genes_file  <- file.path(workdir, "spatial_genes.parquet")

if (!file.exists(spatial_counts_file) || !file.exists(spatial_coords_file)) {
  stop("Missing required spatial input files (spatial_counts.mtx, spatial_coords.parquet).", call. = FALSE)
}

spatial_counts <- Matrix::readMM(spatial_counts_file)
coords_df <- arrow::read_parquet(spatial_coords_file)
spatial_genes <- arrow::read_parquet(spatial_genes_file)$gene

rownames(spatial_counts) <- spatial_genes
colnames(spatial_counts) <- rownames(coords_df)

coords_mat <- as.matrix(coords_df[, c("x", "y")])
rownames(coords_mat) <- rownames(coords_df)
colnames(coords_mat) <- c("x", "y")

# Construct SpatialRNA puck
puck <- SpatialRNA(
  coords = coords_mat,
  counts = spatial_counts,
  nUMI = colSums(spatial_counts)
)

# 4. Read Single-Cell Reference Data
ref_counts_file <- file.path(workdir, "ref_counts.mtx")
ref_meta_file   <- file.path(workdir, "ref_meta.parquet")
ref_genes_file  <- file.path(workdir, "ref_genes.parquet")

if (!file.exists(ref_counts_file) || !file.exists(ref_meta_file)) {
  stop("Missing required reference input files (ref_counts.mtx, ref_meta.parquet).", call. = FALSE)
}

ref_counts <- Matrix::readMM(ref_counts_file)
ref_meta   <- arrow::read_parquet(ref_meta_file)
ref_genes  <- arrow::read_parquet(ref_genes_file)$gene

cell_type_col <- ifelse(!is.null(params$cell_type_col), params$cell_type_col, "cell_type")
cell_types <- as.factor(ref_meta[[cell_type_col]])
names(cell_types) <- rownames(ref_meta)

rownames(ref_counts) <- ref_genes
colnames(ref_counts) <- rownames(ref_meta)

reference <- Reference(
  counts = ref_counts,
  cell_types = cell_types,
  nUMI = colSums(ref_counts)
)

# 5. Execute RCTD Deconvolution
message(sprintf("[R-Worker] Running RCTD in doublet_mode='%s' with max_cores=%d...",
                params$doublet_mode, params$max_cores))

myRCTD <- create.RCTD(
  spatialRNA = puck,
  reference = reference,
  max_cores = as.integer(params$max_cores),
  CELL_MIN_INSTANCE = as.integer(params$CELL_MIN_INSTANCE),
  UMI_min = as.integer(params$UMI_min)
)

myRCTD <- run.RCTD(myRCTD, doublet_mode = params$doublet_mode)

# 6. Extract and Normalize Spot Cell-Type Weights
weights <- as.matrix(myRCTD@results$weights)
norm_weights <- normalize_weights(weights)

norm_weights_df <- as.data.frame(norm_weights)
out_weights_file <- file.path(workdir, "rctd_weights.parquet")
arrow::write_parquet(norm_weights_df, out_weights_file)
message(sprintf("[R-Worker] Wrote deconvolved weights (%d spots x %d cell types) to %s",
                nrow(norm_weights_df), ncol(norm_weights_df), out_weights_file))

# 7. Write run summary
summary_info <- list(
  status = "SUCCESS",
  n_spots_deconvolved = nrow(norm_weights_df),
  n_cell_types = ncol(norm_weights_df),
  cell_types = colnames(norm_weights_df),
  doublet_mode = params$doublet_mode
)
write(jsonlite::toJSON(summary_info, auto_unbox = TRUE, pretty = TRUE),
      file = file.path(workdir, "summary.json"))

message("[R-Worker] RCTD deconvolution completed successfully.")

