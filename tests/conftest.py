"""Pytest fixtures for spatial-microenvironment-profiler testing."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture(scope="session")
def synthetic_visium_adata() -> ad.AnnData:
    """Create a reproducible in-memory Visium AnnData fixture."""
    np.random.seed(42)
    n_spots = 300
    n_genes = 200

    barcodes = [f"SPOT_{i:04d}-1" for i in range(n_spots)]

    # 10 MT genes, 10 Ribo genes, 180 marker/HVG candidates
    mt_genes = (
        [f"MT-ND{i}" for i in range(1, 6)]
        + [f"MT-CO{i}" for i in range(1, 4)]
        + ["MT-ATP6", "MT-CYB"]
    )
    ribo_genes = [f"RPS{i}" for i in range(2, 7)] + [f"RPL{i}" for i in range(3, 8)]
    other_genes = [f"GENE_{i:04d}" for i in range(n_genes - len(mt_genes) - len(ribo_genes))]
    genes = mt_genes + ribo_genes + other_genes

    # Negative binomial counts
    spot_depths = np.random.gamma(shape=5.0, scale=200.0, size=n_spots)
    gene_factors = np.random.uniform(0.1, 2.0, size=n_genes)
    mu_matrix = np.outer(spot_depths, gene_factors) / np.sum(gene_factors)
    p_matrix = 0.5
    r_matrix = mu_matrix * p_matrix / (1.0 - p_matrix)
    counts = np.random.negative_binomial(np.clip(r_matrix, 0.1, 1e4), p_matrix).astype(np.int32)

    # Low-count dead spots (index 0 to 14)
    counts[0:15, :] = np.random.negative_binomial(1, 0.9, size=(15, n_genes))
    # High mitochondrial fraction spots (index 15 to 29)
    counts[15:30, 0 : len(mt_genes)] *= 10

    # Hexagonal grid coordinates
    cols = int(np.sqrt(n_spots * 1.5))
    rows = (n_spots + cols - 1) // cols
    coords = []
    for r in range(rows):
        for c in range(cols):
            if len(coords) < n_spots:
                x = c * 100.0 + (50.0 if r % 2 == 1 else 0.0)
                y = r * 86.6
                coords.append([x, y])
    spatial_coords = np.array(coords[:n_spots], dtype=np.float32)

    # In-tissue flag (last 20 spots out of tissue)
    in_tissue = np.ones(n_spots, dtype=int)
    in_tissue[280:] = 0

    obs_df = pd.DataFrame(
        {
            "in_tissue": in_tissue,
            "array_row": [int(y // 86.6) for _, y in spatial_coords],
            "array_col": [int(x // 100.0) for x, _ in spatial_coords],
        },
        index=barcodes,
    )

    var_df = pd.DataFrame(index=genes)
    var_df["gene_ids"] = [f"ENSG0000{i:07d}" for i in range(n_genes)]

    adata = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=obs_df,
        var=var_df,
    )
    adata.layers["counts"] = sp.csr_matrix(counts.copy())
    adata.obsm["spatial"] = spatial_coords

    # Synthetic RGB image
    synthetic_image = (np.random.rand(500, 500, 3) * 255).astype(np.uint8)
    adata.uns["spatial"] = {
        "sample_stub": {
            "images": {"hires": synthetic_image},
            "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": 55.0},
        }
    }
    return adata


@pytest.fixture(scope="session")
def synthetic_reference_adata() -> ad.AnnData:
    """Create a reproducible single-cell reference AnnData for deconvolution testing."""
    np.random.seed(99)
    n_cells = 250
    n_genes = 200
    cell_types = ["Tumor_Epithelial", "T_Cell", "Macrophage", "Fibroblast", "Endothelial"]

    cell_barcodes = [f"CELL_{i:04d}" for i in range(n_cells)]
    assigned_types = np.random.choice(cell_types, size=n_cells)

    counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.int32)
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]

    obs_df = pd.DataFrame({"cell_type": assigned_types}, index=cell_barcodes)
    var_df = pd.DataFrame(index=genes)

    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs_df, var=var_df)
    adata.layers["counts"] = sp.csr_matrix(counts.copy())
    return adata


@pytest.fixture
def temp_config_yaml(tmp_path: Path) -> Path:
    """Generate a valid YAML configuration file in a temporary folder."""
    yaml_content = """
project_name: "test_suite_crc"
seed: 123
qc:
  min_counts: 100
  min_genes: 50
  max_pct_mt: 25.0
  min_cells: 3
  target_sum: 5000.0
  n_top_genes: 50
  hvg_flavor: "seurat_v3"
morphology:
  patch_size: 64
  backbone: "resnet18"
  pretrained: false
  batch_size: 32
  pca_components: 8
  alpha_morphology: 0.30
deconvolution:
  method: "rctd"
  doublet_mode: "full"
  umi_min: 50
  cell_min_instance: 10
  max_cores: 2
spatial_statistics:
  coord_type: "grid"
  n_neighs: 6
  n_perms: 100
"""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml_content.strip())
    return cfg_file
