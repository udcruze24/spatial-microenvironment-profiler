"""Unit tests for decoupled R subprocess IPC bridge."""

import anndata as ad
import pytest

from spatial_profiler.r_bridge import is_rscript_available, run_r_rctd_deconvolution


def test_is_rscript_available():
    """Verify system check returns a boolean."""
    res = is_rscript_available()
    assert isinstance(res, bool)


def test_r_bridge_key_errors(
    synthetic_visium_adata: ad.AnnData, synthetic_reference_adata: ad.AnnData
):
    """Verify validation when required columns/layers are missing."""
    sp_adata = synthetic_visium_adata[:20].copy()
    ref_adata = synthetic_reference_adata.copy()

    # Missing cell_type column in ref
    with pytest.raises(KeyError, match="cell_type_col"):
        run_r_rctd_deconvolution(sp_adata, ref_adata, cell_type_col="non_existent_col")

    # Missing spatial coordinates
    del sp_adata.obsm["spatial"]
    with pytest.raises(KeyError, match="Spatial coordinates missing"):
        run_r_rctd_deconvolution(sp_adata, ref_adata, cell_type_col="cell_type")


def test_r_bridge_mock_mode(
    synthetic_visium_adata: ad.AnnData, synthetic_reference_adata: ad.AnnData
):
    """Verify offline mock execution populates cell weight proportions."""
    sp_adata = synthetic_visium_adata[:25].copy()
    ref_adata = synthetic_reference_adata.copy()

    weights_df = run_r_rctd_deconvolution(
        sp_adata,
        ref_adata,
        cell_type_col="cell_type",
        mock_if_missing_r=True,
    )

    assert weights_df.shape == (25, len(ref_adata.obs["cell_type"].unique()))
    assert "rctd_cell_weights" in sp_adata.obsm
    assert (sp_adata.obsm["rctd_cell_weights"].values >= 0).all()
