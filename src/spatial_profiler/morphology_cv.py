"""PyTorch histology image feature extraction and multi-modal morphology fusion."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import models, transforms

from spatial_profiler.config import MorphologyConfig

logger = logging.getLogger(__name__)


class SpotPatchDataset(torch.utils.data.Dataset):
    """PyTorch Dataset extracting square image patches centered on Visium spots."""

    def __init__(
        self,
        image: np.ndarray | Image.Image,
        spatial_coords: np.ndarray,
        patch_size: int = 224,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image = np.array(image)
        self.spatial_coords = spatial_coords
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.spatial_coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        cx, cy = self.spatial_coords[idx]
        cx, cy = int(round(cx)), int(round(cy))

        H, W = self.image.shape[:2]
        x_min = max(0, cx - self.half_patch)
        x_max = min(W, cx + self.half_patch)
        y_min = max(0, cy - self.half_patch)
        y_max = min(H, cy + self.half_patch)

        patch = self.image[y_min:y_max, x_min:x_max]

        # Handle border padding if spot is near image boundary
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            padded = np.zeros((self.patch_size, self.patch_size, 3), dtype=self.image.dtype)
            h_p, w_p = patch.shape[:2]
            padded[:h_p, :w_p] = patch
            patch = padded

        pil_patch = Image.fromarray(patch.astype(np.uint8))
        return self.transform(pil_patch)


class MorphologyEncoder(nn.Module):
    """Backbone feature extractor for histological image tiles."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            # Remove classification head, keeping 2048-d global average pool
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
            self.feature_dim = 2048
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


def extract_morphology_embeddings(
    adata: ad.AnnData,
    image: np.ndarray | Image.Image,
    config: MorphologyConfig | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Extract deep convolutional feature embeddings for all spatial coordinates.

    Args:
        adata: Spatial AnnData object with obsm["spatial"] coordinates.
        image: High-resolution brightfield H&E image matching spatial coordinates.
        config: MorphologyConfig with patch size, backbone, and PCA parameters.
        device: "cuda", "cpu", or auto-detect if None.

    Returns:
        PCA-reduced morphology matrix stored in adata.obsm["X_morphology_pca"].
    """
    if config is None:
        config = MorphologyConfig()

    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required for patch extraction.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Extracting morphology features on %s using %s backbone...", device, config.backbone
    )

    dataset = SpotPatchDataset(
        image=image,
        spatial_coords=adata.obsm["spatial"][:, :2],
        patch_size=config.patch_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Cross-platform safe default
    )

    encoder = MorphologyEncoder(backbone=config.backbone, pretrained=config.pretrained).to(device)

    all_embeddings: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        feats = encoder(batch).cpu().numpy()
        all_embeddings.append(feats)

    raw_embeddings = np.vstack(all_embeddings)
    adata.obsm["X_morphology_raw"] = raw_embeddings

    # Reduce dimensionality with PCA
    n_components = min(config.pca_components, raw_embeddings.shape[0] - 1, raw_embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    morphology_pca = pca.fit_transform(raw_embeddings)

    adata.obsm["X_morphology_pca"] = morphology_pca
    adata.uns["morphology_pca_variance_ratio"] = pca.explained_variance_ratio_

    logger.info(
        "Extracted morphology embeddings: %d spots -> %d-d raw -> %d-d PCA",
        adata.n_obs,
        raw_embeddings.shape[1],
        morphology_pca.shape[1],
    )
    return morphology_pca


def fuse_multimodal_graph(
    adata: ad.AnnData,
    alpha_morphology: float = 0.35,
    n_neighbors: int = 15,
) -> ad.AnnData:
    """Fuse transcriptomic PCA and morphological image PCA into a joint latent space.

    Args:
        adata: AnnData containing obsm["X_pca"] and obsm["X_morphology_pca"].
        alpha_morphology: Weight for morphological features (0.0 = RNA only, 1.0 = morphology only).
        n_neighbors: Number of nearest neighbors for the joint neighborhood graph.

    Returns:
        AnnData with fused latent space stored in adata.obsm["X_fused"] and updated
        adata.obsp["connectivities"] and adata.obsp["distances"].
    """
    import scanpy as sc

    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, svd_solver="arpack")

    if "X_morphology_pca" not in adata.obsm:
        raise KeyError("obsm['X_morphology_pca'] missing. Run extract_morphology_embeddings first.")

    rna_pca = adata.obsm["X_pca"]
    morph_pca = adata.obsm["X_morphology_pca"]

    # L2 normalize both modalities before linear convex combination
    rna_norm = rna_pca / (np.linalg.norm(rna_pca, axis=1, keepdims=True) + 1e-8)
    morph_norm = morph_pca / (np.linalg.norm(morph_pca, axis=1, keepdims=True) + 1e-8)

    w_rna = 1.0 - alpha_morphology
    w_morph = alpha_morphology

    fused_matrix = np.hstack([w_rna * rna_norm, w_morph * morph_norm])
    adata.obsm["X_fused"] = fused_matrix

    # Compute joint neighbor graph and UMAP on fused latent space
    sc.pp.neighbors(adata, use_rep="X_fused", n_neighbors=n_neighbors)
    sc.tl.umap(adata)

    logger.info(
        "Constructed multimodal joint graph (alpha_morphology=%.2f, latent dim=%d)",
        alpha_morphology,
        fused_matrix.shape[1],
    )
    return adata
