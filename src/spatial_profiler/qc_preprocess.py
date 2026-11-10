"""Spatial QC, filtering, and normalization module for Visium and Visium HD data.

Preserves raw integer count layers for deep generative modeling (scvi-tools) and
downstream R Bioconductor workers (spacexr / RCTD).
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import scanpy as sc
import scipy.sparse as sp

from spatial_profiler.config import QCConfig

logger = logging.getLogger(__name__)


def load_visium_dataset(
    path: str | Path,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
) -> ad.AnnData:
    """Load 10x Genomics Visium / Visium HD dataset into an AnnData object.

    Args:
        path: Path to Space Ranger output directory or .h5ad file.
        count_file: 10x count matrix filename (default: filtered_feature_bc_matrix.h5).
        library_id: Identifier for the Visium library.

    Returns:
        AnnData containing spatial coordinates, count matrix, and uns metadata.
    """
    path_obj = Path(path)

    if path_obj.suffix in [".h5ad", ".zarr"]:
        logger.info("Loading pre-formatted AnnData from %s", path_obj)
        return sc.read_h5ad(path_obj) if path_obj.suffix == ".h5ad" else ad.read_zarr(path_obj)

    # Space Ranger directory format
    try:
        import squidpy as sq

        logger.info("Reading Visium Space Ranger directory with squidpy from %s", path_obj)
        adata = sq.read.visium(path_obj, count_file=count_file, library_id=library_id)
        return adata
    except Exception as exc:
        logger.warning(
            "Squidpy Visium reader encountered %s. Attempting fallback 10x H5 loader.", exc
        )
        h5_path = path_obj / count_file
        if not h5_path.is_file():
            raise FileNotFoundError(
                f"Count matrix {count_file} not found in directory: {path_obj}"
            ) from exc
        adata = sc.read_10x_h5(h5_path)
        return adata


def calculate_qc_metrics(
    adata: ad.AnnData,
    mitochondrial_prefix: str = "MT-",
    ribosomal_prefix: tuple[str, ...] = ("RPS", "RPL"),
) -> ad.AnnData:
    """Calculate comprehensive quality metrics for spatial barcodes and genes.

    Args:
        adata: Input spatial AnnData object.
        mitochondrial_prefix: Prefix identifying mitochondrial genes (default: MT-).
        ribosomal_prefix: Prefixes identifying ribosomal protein genes.

    Returns:
        AnnData with computed qc metrics stored in obs and var.
    """
    adata.var_names_make_unique()

    # Identify gene biotypes
    adata.var["mt"] = adata.var_names.str.upper().str.startswith(mitochondrial_prefix.upper())
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(ribosomal_prefix)

    # Use Scanpy's optimized QC calculator
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Standardize column naming across scanpy versions
    if "total_counts" in adata.obs:
        adata.obs["n_counts"] = adata.obs["total_counts"]
    if "n_genes_by_counts" in adata.obs:
        adata.obs["n_genes"] = adata.obs["n_genes_by_counts"]

    return adata


def filter_spatial_spots(
    adata: ad.AnnData,
    config: QCConfig,
) -> ad.AnnData:
    """Filter low-quality spatial spots and rare genes.

    Enforces:
    - Tissue localization: adata.obs["in_tissue"] == 1 (if column present)
    - Minimum UMI count per spot: config.min_counts
    - Minimum unique genes detected per spot: config.min_genes
    - Maximum mitochondrial fraction: config.max_pct_mt
    - Minimum spots expressing each gene: config.min_cells

    Args:
        adata: Raw AnnData object.
        config: QCConfig validation parameters.

    Returns:
        Filtered AnnData object.
    """
    n_spots_init = adata.n_obs

    # 1. Filter out-of-tissue barcodes if tissue positioning is annotated
    if config.filter_out_of_tissue and "in_tissue" in adata.obs:
        in_tissue_mask = adata.obs["in_tissue"].astype(int) == 1
        adata = adata[in_tissue_mask, :].copy()
        logger.info(
            "Filtered out-of-tissue spots: %d -> %d (%d dropped)",
            n_spots_init,
            adata.n_obs,
            n_spots_init - adata.n_obs,
        )

    # 2. Ensure QC metrics are populated
    if "pct_counts_mt" not in adata.obs or "n_genes" not in adata.obs:
        adata = calculate_qc_metrics(adata, mitochondrial_prefix=config.mitochondrial_prefix)

    # 3. Apply spot-level threshold masks
    valid_spots = (
        (adata.obs["n_counts"] >= config.min_counts)
        & (adata.obs["n_genes"] >= config.min_genes)
        & (adata.obs["pct_counts_mt"] <= config.max_pct_mt)
    )

    n_before_spot_filter = adata.n_obs
    adata = adata[valid_spots, :].copy()
    logger.info(
        "Filtered low-quality spots: %d -> %d (%d dropped, cutoff: min_counts=%d, min_genes=%d, max_mt=%.1f%%)",
        n_before_spot_filter,
        adata.n_obs,
        n_before_spot_filter - adata.n_obs,
        config.min_counts,
        config.min_genes,
        config.max_pct_mt,
    )

    if adata.n_obs == 0:
        raise ValueError(
            f"All spots were filtered out by QC thresholds (min_counts={config.min_counts}, "
            f"min_genes={config.min_genes}, max_pct_mt={config.max_pct_mt}%). "
            f"Ensure dataset sequencing depth matches configured thresholds or provide a custom config."
        )

    # 4. Filter rare genes
    n_before_gene_filter = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=config.min_cells)
    logger.info(
        "Filtered rare genes: %d -> %d (%d dropped, cutoff: min_cells=%d)",
        n_before_gene_filter,
        adata.n_vars,
        n_before_gene_filter - adata.n_vars,
        config.min_cells,
    )

    return adata


def normalize_and_select_hvgs(
    adata: ad.AnnData,
    config: QCConfig,
) -> ad.AnnData:
    """Normalize library sizes, perform log1p transform, and detect highly variable genes.

    Crucially preserves raw integer count matrix in `adata.layers["counts"]`
    required by scvi-tools (DestVI) and R Bioconductor workers (spacexr/RCTD).

    Args:
        adata: Filtered AnnData object.
        config: QCConfig parameters.

    Returns:
        AnnData with raw counts stored in layers["counts"], log-normalized data in X,
        and HVGs tagged in var["highly_variable"].
    """
    # 1. Preserve raw integer counts
    if "counts" not in adata.layers:
        if sp.issparse(adata.X):
            adata.layers["counts"] = adata.X.copy()
        else:
            adata.layers["counts"] = sp.csr_matrix(adata.X.copy())

    # 2. Select Highly Variable Genes (HVGs)
    # seurat_v3 requires raw count layer; gracefully fall back to seurat on log1p if needed
    hvg_success = False
    effective_n_top = min(config.n_top_genes, max(10, adata.n_vars - 1))

    if config.hvg_flavor == "seurat_v3":
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=effective_n_top,
                flavor="seurat_v3",
                layer="counts",
                subset=False,
            )
            hvg_success = True
        except Exception as exc:
            logger.warning("seurat_v3 HVG selection encountered %s. Falling back to 'seurat'.", exc)

    # 3. Log-normalize library size for visualization, PCA, and morphology fusion
    sc.pp.normalize_total(adata, target_sum=config.target_sum)
    sc.pp.log1p(adata)

    if not hvg_success:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=effective_n_top,
            flavor="seurat",
            subset=False,
        )

    logger.info(
        "Selected %d highly variable genes (HVG flavor: %s)",
        adata.var["highly_variable"].sum(),
        config.hvg_flavor if hvg_success else "seurat (fallback)",
    )

    return adata


def run_spatial_qc_pipeline(
    data_source: str | Path | ad.AnnData,
    config: QCConfig | None = None,
) -> ad.AnnData:
    """End-to-end execution of Phase 1 Spatial QC and Preprocessing.

    Args:
        data_source: Path to Visium directory / .h5ad or instantiated AnnData.
        config: Optional QCConfig instance (uses defaults if None).

    Returns:
        Fully preprocessed AnnData ready for morphology fusion and RCTD deconvolution.
    """
    if config is None:
        config = QCConfig()

    logger.info("Initializing Spatial QC pipeline...")
    if isinstance(data_source, ad.AnnData):
        adata = data_source.copy()
    else:
        adata = load_visium_dataset(data_source)

    initial_spots = adata.n_obs
    initial_genes = adata.n_vars

    adata = calculate_qc_metrics(adata, mitochondrial_prefix=config.mitochondrial_prefix)
    adata = filter_spatial_spots(adata, config=config)
    adata = normalize_and_select_hvgs(adata, config=config)

    # Store provenance metadata
    adata.uns["qc_provenance"] = {
        "initial_spots": int(initial_spots),
        "retained_spots": int(adata.n_obs),
        "initial_genes": int(initial_genes),
        "retained_genes": int(adata.n_vars),
        "hvg_count": int(adata.var["highly_variable"].sum()),
        "target_sum": float(config.target_sum),
        "max_pct_mt": float(config.max_pct_mt),
        "has_counts_layer": "counts" in adata.layers,
    }

    logger.info(
        "QC pipeline complete: %d spots × %d genes retained (%d HVGs). Counts layer preserved.",
        adata.n_obs,
        adata.n_vars,
        adata.var["highly_variable"].sum(),
    )
    return adata
