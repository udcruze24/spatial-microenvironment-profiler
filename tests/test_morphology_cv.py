"""Unit tests for PyTorch histology patch extraction and morphology fusion."""

import anndata as ad
import numpy as np
import torch

from spatial_profiler.config import MorphologyConfig
from spatial_profiler.morphology_cv import (
    MorphologyEncoder,
    SpotPatchDataset,
    extract_morphology_embeddings,
    fuse_multimodal_graph,
)


def test_spot_patch_dataset():
    """Verify spot patch extraction handles bounds and returns tensor of correct shape."""
    img = np.random.randint(0, 255, size=(300, 300, 3), dtype=np.uint8)
    coords = np.array([[50, 50], [150, 150], [290, 290]])  # includes boundary spot

    dataset = SpotPatchDataset(image=img, spatial_coords=coords, patch_size=64)
    assert len(dataset) == 3

    patch0 = dataset[0]
    assert isinstance(patch0, torch.Tensor)
    assert patch0.shape == (3, 64, 64)

    # Boundary patch
    patch_boundary = dataset[2]
    assert patch_boundary.shape == (3, 64, 64)


def test_morphology_encoder():
    """Verify forward pass of feature extractor without classification head."""
    encoder = MorphologyEncoder(backbone="resnet18", pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    out = encoder(x)
    assert out.shape == (2, 512)


def test_extract_morphology_embeddings(synthetic_visium_adata: ad.AnnData):
    """Verify deep embedding extraction and PCA dimensionality reduction."""
    adata = synthetic_visium_adata[:30].copy()  # Subset for fast test
    image = adata.uns["spatial"]["sample_stub"]["images"]["hires"]

    config = MorphologyConfig(
        patch_size=64,
        backbone="resnet18",
        pretrained=False,
        batch_size=16,
        pca_components=8,
    )

    pca_feats = extract_morphology_embeddings(adata, image=image, config=config, device="cpu")

    assert pca_feats.shape == (30, 8)
    assert "X_morphology_pca" in adata.obsm
    assert "X_morphology_raw" in adata.obsm
    assert adata.obsm["X_morphology_raw"].shape == (30, 512)


def test_fuse_multimodal_graph(synthetic_visium_adata: ad.AnnData):
    """Verify multimodal fusion constructs joint latent space and neighbor graph."""
    adata = synthetic_visium_adata[:30].copy()

    # Mock RNA PCA and Morphology PCA
    np.random.seed(42)
    adata.obsm["X_pca"] = np.random.randn(30, 10).astype(np.float32)
    adata.obsm["X_morphology_pca"] = np.random.randn(30, 8).astype(np.float32)

    fused_adata = fuse_multimodal_graph(adata, alpha_morphology=0.35, n_neighbors=5)

    assert "X_fused" in fused_adata.obsm
    assert fused_adata.obsm["X_fused"].shape == (30, 18)
    assert "connectivities" in fused_adata.obsp
    assert "X_umap" in fused_adata.obsm
