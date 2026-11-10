"""Unit tests for Phase 1 Spatial QC and Preprocessing pipeline."""

import anndata as ad
import scipy.sparse as sp

from spatial_profiler.config import QCConfig
from spatial_profiler.qc_preprocess import (
    calculate_qc_metrics,
    filter_spatial_spots,
    normalize_and_select_hvgs,
    run_spatial_qc_pipeline,
)


def test_calculate_qc_metrics(synthetic_visium_adata: ad.AnnData):
    """Verify calculation of mitochondrial, ribosomal, and library size metrics."""
    adata = synthetic_visium_adata.copy()
    adata = calculate_qc_metrics(adata, mitochondrial_prefix="MT-")

    assert "pct_counts_mt" in adata.obs
    assert "pct_counts_ribo" in adata.obs
    assert "n_counts" in adata.obs
    assert "n_genes" in adata.obs

    # MT percentage must be non-negative and finite
    assert (adata.obs["pct_counts_mt"] >= 0).all()
    assert (adata.obs["pct_counts_mt"] <= 100).all()
    assert (adata.obs["pct_counts_ribo"] >= 0).all()


def test_filter_spatial_spots(synthetic_visium_adata: ad.AnnData):
    """Verify that out-of-tissue, low-count, and high-MT spots are filtered out."""
    adata = synthetic_visium_adata.copy()
    initial_spots = adata.n_obs
    assert (adata.obs["in_tissue"] == 0).sum() == 20

    config = QCConfig(
        min_counts=200,
        min_genes=50,
        max_pct_mt=20.0,
        min_cells=5,
        filter_out_of_tissue=True,
    )
    filtered = filter_spatial_spots(adata, config=config)

    # Out of tissue spots must be gone
    assert (filtered.obs["in_tissue"] == 1).all()
    # Filtered spots count must be less than initial
    assert filtered.n_obs < initial_spots
    # All retained spots must meet thresholds
    assert (filtered.obs["n_counts"] >= config.min_counts).all()
    assert (filtered.obs["n_genes"] >= config.min_genes).all()
    assert (filtered.obs["pct_counts_mt"] <= config.max_pct_mt).all()


def test_normalize_and_select_hvgs(synthetic_visium_adata: ad.AnnData):
    """Verify raw counts preservation in layers['counts'] and HVG identification."""
    adata = synthetic_visium_adata.copy()
    config = QCConfig(target_sum=10000.0, n_top_genes=50, hvg_flavor="seurat_v3")

    processed = normalize_and_select_hvgs(adata, config=config)

    # layers["counts"] must exist and match raw integer format
    assert "counts" in processed.layers
    assert sp.issparse(processed.layers["counts"])

    # Processed X must be log-normalized (max values typically < 15)
    max_log_val = processed.X.max()
    assert max_log_val < 15.0

    # HVGs must be identified
    assert "highly_variable" in processed.var
    assert processed.var["highly_variable"].sum() > 0


def test_run_spatial_qc_pipeline_end_to_end(synthetic_visium_adata: ad.AnnData):
    """End-to-end integration test of the Phase 1 Spatial QC pipeline."""
    adata = synthetic_visium_adata.copy()
    config = QCConfig(
        min_counts=100,
        min_genes=30,
        max_pct_mt=25.0,
        min_cells=2,
        n_top_genes=60,
    )

    result = run_spatial_qc_pipeline(adata, config=config)

    assert result.n_obs > 0
    assert result.n_vars > 0
    assert "counts" in result.layers
    assert "qc_provenance" in result.uns
    assert result.uns["qc_provenance"]["has_counts_layer"] is True
    assert result.uns["qc_provenance"]["retained_spots"] == result.n_obs
