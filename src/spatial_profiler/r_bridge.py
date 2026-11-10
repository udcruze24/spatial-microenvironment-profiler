"""Decoupled Python-to-R subprocess bridge for spatial deconvolution (spacexr/RCTD).

Avoids fragile in-memory C-bindings (e.g. rpy2) by using file-backed IPC
via Matrix Market (.mtx) and Apache Parquet (.parquet) files.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def is_rscript_available() -> bool:
    """Check if Rscript binary is executable on the current PATH."""
    return shutil.which("Rscript") is not None


def run_r_rctd_deconvolution(
    spatial_adata: ad.AnnData,
    ref_adata: ad.AnnData,
    cell_type_col: str = "cell_type",
    doublet_mode: str = "full",
    max_cores: int = 4,
    r_script_path: str | Path = "scripts/r_workers/run_rctd_cside.R",
    umi_min: int = 100,
    cell_min_instance: int = 25,
    mock_if_missing_r: bool = False,
) -> pd.DataFrame:
    """Execute spacexr (RCTD) deconvolution in a background R subprocess.

    Exports spatial counts and coordinates, formats single-cell reference annotations,
    invokes the standalone R worker script, and re-imports normalized cell weights.

    Args:
        spatial_adata: Spatial AnnData containing raw integer counts in layers["counts"]
                       or X, and 2D coordinates in obsm["spatial"].
        ref_adata: Single-cell reference AnnData with cell-type annotations in obs.
        cell_type_col: Column in ref_adata.obs with reference cell-type labels.
        doublet_mode: Mode for RCTD ("full", "doublet", "multi").
        max_cores: Maximum CPU threads to allocate to the R worker.
        r_script_path: Path to run_rctd_cside.R worker script.
        umi_min: Minimum spot UMI filter.
        cell_min_instance: Minimum occurrences per reference cell type.
        mock_if_missing_r: If True and Rscript is absent, returns a synthetic weights
                           matrix for offline unit testing without R installed.

    Returns:
        DataFrame of deconvolved cell-type proportions (spots x cell types),
        also stored in spatial_adata.obsm["rctd_cell_weights"].
    """
    if cell_type_col not in ref_adata.obs:
        raise KeyError(f"cell_type_col '{cell_type_col}' not found in ref_adata.obs")

    if "spatial" not in spatial_adata.obsm:
        raise KeyError("Spatial coordinates missing from spatial_adata.obsm['spatial']")

    # Check Rscript availability
    if not is_rscript_available():
        if mock_if_missing_r:
            logger.warning(
                "Rscript binary not found. Generating mock deconvolution weights for testing."
            )
            cell_types = sorted(ref_adata.obs[cell_type_col].unique())
            n_spots = spatial_adata.n_obs
            raw_weights = np.random.dirichlet(np.ones(len(cell_types)), size=n_spots)
            weights_df = pd.DataFrame(
                raw_weights, index=spatial_adata.obs_names, columns=cell_types
            )
            spatial_adata.obsm["rctd_cell_weights"] = weights_df
            return weights_df
        raise OSError(
            "Rscript executable not found on PATH. Install R 4.3+ with 'spacexr' "
            "or run inside the provided Docker container."
        )

    r_path = Path(r_script_path)
    if not r_path.is_file():
        raise FileNotFoundError(f"R worker script not found at: {r_path.resolve()}")

    # Extract count matrices (prefer raw integer counts)
    sp_counts = (
        spatial_adata.layers["counts"] if "counts" in spatial_adata.layers else spatial_adata.X
    )
    if not sp.issparse(sp_counts):
        sp_counts = sp.csr_matrix(sp_counts)

    ref_counts = ref_adata.layers["counts"] if "counts" in ref_adata.layers else ref_adata.X
    if not sp.issparse(ref_counts):
        ref_counts = sp.csr_matrix(ref_counts)

    with tempfile.TemporaryDirectory(prefix="rctd_ipc_") as tmpdir:
        tmp = Path(tmpdir)
        logger.info("Staging IPC data for R worker in %s", tmp)

        # 1. Export sparse matrices (transposed: genes x cells for Matrix Market)
        sio.mmwrite(str(tmp / "spatial_counts.mtx"), sp_counts.T.tocsc())
        sio.mmwrite(str(tmp / "ref_counts.mtx"), ref_counts.T.tocsc())

        # 2. Export spatial coordinates and gene lists
        coords_df = pd.DataFrame(
            spatial_adata.obsm["spatial"][:, :2],
            index=spatial_adata.obs_names,
            columns=["x", "y"],
        )
        coords_df.to_parquet(tmp / "spatial_coords.parquet")
        pd.DataFrame({"gene": list(spatial_adata.var_names)}).to_parquet(
            tmp / "spatial_genes.parquet"
        )
        pd.DataFrame({"gene": list(ref_adata.var_names)}).to_parquet(tmp / "ref_genes.parquet")

        # 3. Export reference metadata
        ref_adata.obs[[cell_type_col]].to_parquet(tmp / "ref_meta.parquet")

        # 4. Export parameter manifest
        params = {
            "cell_type_col": cell_type_col,
            "doublet_mode": doublet_mode,
            "max_cores": int(max_cores),
            "UMI_min": int(umi_min),
            "CELL_MIN_INSTANCE": int(cell_min_instance),
        }
        (tmp / "params.json").write_text(json.dumps(params, indent=2))

        # 5. Launch Rscript subprocess
        cmd = ["Rscript", "--vanilla", str(r_path.resolve()), "--workdir", str(tmp)]
        logger.info("Executing R command: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if proc.returncode != 0:
            logger.error(
                "Rscript worker execution failed.\nSTDOUT:\n%s\nSTDERR:\n%s",
                proc.stdout,
                proc.stderr,
            )
            raise RuntimeError(
                f"Rscript RCTD worker exited with code {proc.returncode}:\n{proc.stderr}"
            )

        # 6. Read back deconvolved weights
        out_parquet = tmp / "rctd_weights.parquet"
        if not out_parquet.is_file():
            raise FileNotFoundError(f"Expected R output file {out_parquet} was not generated.")

        weights_df = pd.read_parquet(out_parquet)
        weights_df.index = spatial_adata.obs_names
        spatial_adata.obsm["rctd_cell_weights"] = weights_df
        logger.info("RCTD deconvolution complete. Shape: %s (spots x cell types)", weights_df.shape)
        return weights_df
