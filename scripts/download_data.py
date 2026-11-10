#!/usr/bin/env python3
"""Dataset downloader and fixture builder for spatial-microenvironment-profiler.

Fetches open-access 10x Genomics Visium HD CRC datasets, multi-cancer benchmarks,
or generates self-contained mini-stubs for CI/CD testing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_data")

DATASETS = {
    "visium_hd_crc": {
        "description": "10x Genomics Visium HD Human Colorectal Cancer (8um & 16um bins)",
        "url": "https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer/Visium_HD_Human_Colon_Cancer_feature_slice.h5",
        "filename": "visium_hd_human_crc_slice.h5",
    },
    "visium_v1_crc": {
        "description": "10x Genomics Visium Human Colorectal Cancer (Standard 55um spot array)",
        "url": "https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Colon_Cancer/Visium_FFPE_Human_Colon_Cancer_filtered_feature_bc_matrix.h5",
        "filename": "visium_ffpe_human_colon_cancer.h5",
    },
}


def generate_mini_visium_fixture(output_path: Path, n_spots: int = 300, n_genes: int = 200) -> Path:
    """Generate a synthetic mini Visium dataset for fast CI testing and local verification.

    Creates an AnnData object with:
    - Raw integer count matrix (negative binomial distribution)
    - Mitochondrial and ribosomal genes
    - Spatial 2D coordinates arranged on a hexagonal grid
    - Out-of-tissue and low-quality spots to test QC filters
    - Synthetic low-resolution pathology RGB image in uns["spatial"]
    """
    try:
        import anndata as ad
        import pandas as pd
    except ImportError:
        logger.error(
            "anndata and pandas are required to generate the fixture. Install with `pip install anndata pandas`."
        )
        sys.exit(1)

    np.random.seed(42)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Barcode names and gene symbols
    barcodes = [f"SPOT_{i:04d}-1" for i in range(n_spots)]

    # 15 mitochondrial genes, 15 ribosomal genes, remainder standard HVGs & markers
    mt_genes = (
        [f"MT-ND{i}" for i in range(1, 6)]
        + [f"MT-CO{i}" for i in range(1, 4)]
        + [f"MT-ATP{i}" for i in range(6, 9)]
        + ["MT-CYB"]
    )
    ribo_genes = [f"RPS{i}" for i in range(2, 10)] + [f"RPL{i}" for i in range(3, 10)]
    other_genes = [f"GENE_{i:04d}" for i in range(n_genes - len(mt_genes) - len(ribo_genes))]
    genes = mt_genes + ribo_genes + other_genes

    # 2. Expression counts (Negative Binomial sampling)
    # Mean counts varying by spot to simulate depth variation
    spot_depths = np.random.gamma(shape=5.0, scale=200.0, size=n_spots)  # Average ~1000 counts
    gene_factors = np.random.uniform(0.1, 2.0, size=n_genes)

    mu_matrix = np.outer(spot_depths, gene_factors) / np.sum(gene_factors)
    p_matrix = 0.5
    r_matrix = mu_matrix * p_matrix / (1.0 - p_matrix)

    # Generate integer count matrix
    counts = np.random.negative_binomial(np.clip(r_matrix, 0.1, 1e4), p_matrix).astype(np.int32)

    # Add a few intentionally low-quality and dead spots for QC testing
    counts[0:15, :] = np.random.negative_binomial(
        1, 0.9, size=(15, n_genes)
    )  # Ultra low counts (<50)
    counts[15:30, 0 : len(mt_genes)] *= 8  # High mitochondrial fraction (>30%)

    # 3. Spatial Coordinates (Hexagonal Grid)
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

    # 4. Spot Metadata (obs)
    in_tissue = np.ones(n_spots, dtype=int)
    in_tissue[280:] = 0  # 20 out-of-tissue spots

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

    # 5. Build AnnData
    adata = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=obs_df,
        var=var_df,
    )
    adata.layers["counts"] = sp.csr_matrix(counts.copy())
    adata.obsm["spatial"] = spatial_coords

    # Synthetic RGB pathology image
    synthetic_image = (np.random.rand(400, 400, 3) * 255).astype(np.uint8)
    adata.uns["spatial"] = {
        "library_stub": {
            "images": {"hires": synthetic_image, "lowres": synthetic_image[::2, ::2, :]},
            "scalefactors": {
                "tissue_hires_scalef": 0.1,
                "tissue_lowres_scalef": 0.05,
                "spot_diameter_fullres": 89.0,
                "fiducial_diameter_fullres": 144.0,
            },
        }
    }

    adata.write_h5ad(output_path)
    logger.info(
        "Successfully generated mini Visium fixture at %s (%d spots x %d genes)",
        output_path,
        n_spots,
        n_genes,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or generate spatial biology datasets.")
    parser.add_argument(
        "--dataset",
        choices=["mini_stub", "visium_hd_crc", "visium_v1_crc", "all"],
        default="mini_stub",
        help="Target dataset to fetch or generate (default: mini_stub for fast offline testing)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("tests/data/mini_visium_stub"),
        help="Destination directory for dataset storage",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ["mini_stub", "all"]:
        fixture_file = args.outdir / "mini_visium.h5ad"
        generate_mini_visium_fixture(fixture_file)

    if args.dataset in ["visium_hd_crc", "visium_v1_crc"]:
        meta = DATASETS[args.dataset]
        logger.info("Downloading %s from %s", meta["description"], meta["url"])
        try:
            import urllib.request

            dest = args.outdir / meta["filename"]
            urllib.request.urlretrieve(meta["url"], dest)
            logger.info("Downloaded %s to %s", meta["filename"], dest)
        except Exception as exc:
            logger.error(
                "Download failed (%s). You can manually place the file in %s", exc, args.outdir
            )


if __name__ == "__main__":
    main()
