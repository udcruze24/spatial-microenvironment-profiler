"""Spatial neighborhood graphs, Moran's I autocorrelation, and Hotspot gene module discovery."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from spatial_profiler.config import HotspotConfig, SpatialStatsConfig

logger = logging.getLogger(__name__)


def build_spatial_neighborhood_graph(
    adata: ad.AnnData,
    config: SpatialStatsConfig | None = None,
) -> ad.AnnData:
    """Build a spatial coordinate neighbor graph using Squidpy.

    Supports hexagonal Visium grid topology or generic spatial coordinates.

    Args:
        adata: AnnData with obsm["spatial"] coordinates.
        config: SpatialStatsConfig with coord_type, n_neighs, and n_rings.

    Returns:
        AnnData with spatial graph stored in adata.obsp["spatial_connectivities"].
    """
    if config is None:
        config = SpatialStatsConfig()

    logger.info(
        "Building spatial neighbor graph (coord_type=%s, n_neighs=%d, n_rings=%d)",
        config.coord_type,
        config.n_neighs,
        config.n_rings,
    )
    sq.gr.spatial_neighbors(
        adata,
        coord_type=config.coord_type,
        n_neighs=config.n_neighs,
        n_rings=config.n_rings,
    )
    return adata


def compute_niche_enrichment(
    adata: ad.AnnData,
    cluster_key: str = "spatial_niche",
    n_perms: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute permutation-based spatial neighborhood enrichment between cell types or niches.

    Args:
        adata: AnnData with spatial neighbor graph and cluster labels in adata.obs.
        cluster_key: Column in obs representing discrete cell types or spatial clusters.
        n_perms: Number of Monte Carlo permutations for significance testing.

    Returns:
        Tuple of (enrichment_z_score_matrix, p_value_matrix).
    """
    if cluster_key not in adata.obs:
        # If no clusters are present, cluster using Leiden on spatial neighbors or transcriptomics
        logger.info(
            "Cluster key '%s' not found. Running Leiden clustering on connectivities...",
            cluster_key,
        )
        sc.tl.leiden(adata, key_added=cluster_key, resolution=0.8)

    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    logger.info(
        "Computing neighborhood enrichment for '%s' (%d permutations)...", cluster_key, n_perms
    )
    sq.gr.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
        seed=42,
    )

    z_scores = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    p_values = adata.uns[f"{cluster_key}_nhood_enrichment"]["count"]
    return z_scores, p_values


def compute_spatial_autocorrelation(
    adata: ad.AnnData,
    genes: list[str] | None = None,
    mode: str = "moran",
    n_perms: int = 100,
) -> pd.DataFrame:
    """Calculate Moran's I or Geary's C spatial autocorrelation for variable genes.

    Args:
        adata: AnnData with spatial neighbor graph.
        genes: Optional list of genes to evaluate (defaults to all or HVGs).
        mode: "moran" or "geary".
        n_perms: Permutations for significance estimation.

    Returns:
        DataFrame ranking genes by spatial autocorrelation statistic (Moran's I / Geary's C).
    """
    logger.info("Computing spatial autocorrelation (mode=%s)...", mode)
    kwargs = {"mode": mode, "n_perms": n_perms, "seed": 42}
    if genes is not None:
        kwargs["genes"] = genes
    sq.gr.spatial_autocorr(
        adata,
        **kwargs,
    )
    key = "moranI" if mode == "moran" else "gearyC"
    df = adata.uns[key].sort_values(by="I" if mode == "moran" else "C", ascending=(mode != "moran"))
    return df


def find_hotspot_gene_modules(
    adata: ad.AnnData,
    config: HotspotConfig | None = None,
) -> pd.DataFrame | None:
    """Identify spatially correlated gene co-expression modules using Hotspot.

    Uses Depth-Adjusted Negative Binomial (danb) model on raw integer counts.

    Args:
        adata: AnnData containing raw counts in layers["counts"] and spatial coords.
        config: HotspotConfig parameters.

    Returns:
        DataFrame mapping genes to modules, or None if hotspot is unavailable.
    """
    if config is None:
        config = HotspotConfig()

    try:
        import hotspot

        logger.info(
            "Running Hotspot module discovery (model=%s, neighbors=%d)...",
            config.model,
            config.n_neighbors,
        )

        counts = adata.layers["counts"].T.tocsc() if "counts" in adata.layers else adata.X.T.tocsc()
        coords = pd.DataFrame(
            adata.obsm["spatial"][:, :2], index=adata.obs_names, columns=["x", "y"]
        )

        hs = hotspot.Hotspot(
            counts,
            model=config.model,
            latent=None,
            umi_counts=np.array(counts.sum(axis=0)).flatten(),
            tree=None,
        )
        hs.create_knn_graph(
            weighted_graph=False, n_neighbors=config.n_neighbors, positional_data=coords
        )
        hs_results = hs.compute_autocorrelations(jobs=1)

        # Select significant genes
        sig_genes = hs_results.loc[hs_results.fdr < config.fdr_threshold].index
        if len(sig_genes) < config.min_gene_threshold:
            logger.warning(
                "Fewer than %d significant spatial genes found by Hotspot.",
                config.min_gene_threshold,
            )
            return None

        # Group into modules
        modules = hs.create_modules(
            min_gene_threshold=config.min_gene_threshold,
            core_only=False,
            fdr_threshold=config.fdr_threshold,
        )
        adata.uns["hotspot_modules"] = modules
        return modules
    except ImportError:
        logger.warning(
            "Package 'hotspot-sc' not installed. Ranking genes by Moran's I as fallback."
        )
        moran_df = compute_spatial_autocorrelation(adata, mode="moran", n_perms=50)
        return moran_df
