"""Unit tests for spatial neighborhood graphs, niche enrichment, and spatial statistics."""

import anndata as ad
import numpy as np

from spatial_profiler.config import SpatialStatsConfig
from spatial_profiler.spatial_statistics import (
    build_spatial_neighborhood_graph,
    compute_niche_enrichment,
    compute_spatial_autocorrelation,
)


def test_build_spatial_neighborhood_graph(synthetic_visium_adata: ad.AnnData):
    """Verify construction of Visium hexagonal spatial graph."""
    adata = synthetic_visium_adata[:50].copy()
    config = SpatialStatsConfig(coord_type="generic", n_neighs=4)

    adata = build_spatial_neighborhood_graph(adata, config=config)
    assert "spatial_connectivities" in adata.obsp
    assert "spatial_distances" in adata.obsp


def test_niche_enrichment(synthetic_visium_adata: ad.AnnData):
    """Verify Monte Carlo neighborhood enrichment calculation."""
    adata = synthetic_visium_adata[:50].copy()
    build_spatial_neighborhood_graph(
        adata, config=SpatialStatsConfig(coord_type="generic", n_neighs=4)
    )

    # Add mock spatial niches
    np.random.seed(42)
    adata.obs["spatial_niche"] = np.random.choice(
        ["Tumor_Core", "Stroma", "Immune_Infiltrate"], size=adata.n_obs
    )

    z_scores, counts = compute_niche_enrichment(adata, cluster_key="spatial_niche", n_perms=50)

    assert z_scores.shape == (3, 3)
    assert counts.shape == (3, 3)
    assert not np.isnan(z_scores).all()


def test_spatial_autocorrelation(synthetic_visium_adata: ad.AnnData):
    """Verify Moran's I spatial autocorrelation ranking."""
    adata = synthetic_visium_adata[:40].copy()
    build_spatial_neighborhood_graph(
        adata, config=SpatialStatsConfig(coord_type="generic", n_neighs=4)
    )

    moran_df = compute_spatial_autocorrelation(adata, mode="moran", n_perms=20)
    assert "I" in moran_df.columns
    assert len(moran_df) == adata.n_vars
