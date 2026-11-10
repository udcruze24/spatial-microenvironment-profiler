"""Configuration schema and validation using Pydantic v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class QCConfig(BaseModel):
    """Quality control filtering and normalization parameters."""

    min_counts: int = Field(default=500, ge=1, description="Minimum UMI count per spatial spot")
    min_genes: int = Field(default=250, ge=1, description="Minimum unique genes detected per spot")
    max_pct_mt: float = Field(
        default=20.0, ge=0.0, le=100.0, description="Maximum mitochondrial expression percentage"
    )
    min_cells: int = Field(
        default=10, ge=1, description="Minimum spots expressing a gene to retain it"
    )
    target_sum: float = Field(
        default=10000.0, gt=0, description="Target scaling factor for library size normalization"
    )
    n_top_genes: int = Field(
        default=3000, ge=10, description="Number of highly variable genes (HVGs) to select"
    )
    hvg_flavor: Literal["seurat_v3", "seurat", "cell_ranger"] = Field(
        default="seurat_v3", description="HVG dispersion method"
    )
    filter_out_of_tissue: bool = Field(
        default=True, description="Filter barcodes where in_tissue != 1"
    )
    mitochondrial_prefix: str = Field(
        default="MT-", description="Prefix pattern for identifying mitochondrial genes"
    )


class MorphologyConfig(BaseModel):
    """Deep learning histology image feature extraction parameters."""

    patch_size: int = Field(
        default=224, ge=32, description="Square pixel dimension of spot image tile (e.g. 224x224)"
    )
    backbone: Literal["resnet50", "resnet18", "vit_base_patch16_224"] = Field(
        default="resnet50", description="Computer vision encoder architecture"
    )
    pretrained: bool = Field(default=True, description="Use ImageNet pretrained weights")
    batch_size: int = Field(default=128, ge=1, description="Mini-batch size for image inference")
    num_workers: int = Field(default=4, ge=0, description="PyTorch DataLoader worker count")
    pca_components: int = Field(
        default=32, ge=2, description="Target dimension for compressed morphological embedding"
    )
    alpha_morphology: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Multimodal weight for morphological embeddings relative to transcriptomic PCA",
    )


class DeconvolutionConfig(BaseModel):
    """Spatial cell-type deconvolution parameters (e.g. spacexr / RCTD)."""

    method: Literal["rctd", "destvi"] = Field(default="rctd", description="Deconvolution algorithm")
    doublet_mode: Literal["full", "doublet", "multi"] = Field(
        default="full", description="RCTD execution mode (full allows multi-cell mixtures)"
    )
    umi_min: int = Field(default=100, ge=1, description="Minimum spot UMI threshold for RCTD")
    cell_min_instance: int = Field(
        default=25, ge=1, description="Minimum cells per reference type required by RCTD"
    )
    max_cores: int = Field(default=4, ge=1, description="Maximum CPU cores for R worker bridge")
    r_script_path: str = Field(
        default="scripts/r_workers/run_rctd_cside.R",
        description="Path to standalone R worker script",
    )


class HotspotConfig(BaseModel):
    """Hotspot spatially variable gene module discovery configuration."""

    model: Literal["danb", "bernoulli", "normal"] = Field(
        default="danb",
        description="Gene expression model (danb = Depth-Adjusted Negative Binomial)",
    )
    n_neighbors: int = Field(default=30, ge=5, description="K-nearest neighbors in spatial space")
    fdr_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0, description="FDR significance cutoff for variable genes"
    )
    min_gene_threshold: int = Field(
        default=15, ge=2, description="Minimum genes per co-expression module"
    )


class SpatialStatsConfig(BaseModel):
    """Spatial graph and neighborhood statistics parameters."""

    coord_type: Literal["grid", "generic"] = Field(
        default="grid", description="Visium hexagonal grid or generic Euclidean coordinates"
    )
    n_neighs: int = Field(default=6, ge=1, description="Hexagonal neighbor degree")
    n_rings: int = Field(default=2, ge=1, description="Number of rings for spatial neighbor graph")
    n_perms: int = Field(
        default=1000, ge=50, description="Monte Carlo permutations for neighborhood enrichment"
    )
    cluster_key: str = Field(
        default="spatial_niche", description="Cluster column in adata.obs for spatial enrichment"
    )
    hotspot: HotspotConfig = Field(default_factory=HotspotConfig)


class PipelineConfig(BaseModel):
    """Master configuration for spatial-microenvironment-profiler."""

    project_name: str = Field(
        default="visium_hd_human_crc", description="Identifies analysis run and artifacts"
    )
    seed: int = Field(default=42, description="Global random seed for reproducibility")
    qc: QCConfig = Field(default_factory=QCConfig)
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)
    deconvolution: DeconvolutionConfig = Field(default_factory=DeconvolutionConfig)
    spatial_statistics: SpatialStatsConfig = Field(default_factory=SpatialStatsConfig)

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("project_name cannot be empty")
        return clean


def load_config(config_path: str | Path | None = None, **overrides: Any) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML file or dictionary.

    Args:
        config_path: Optional path to a YAML configuration file.
        **overrides: Optional key-value overrides for top-level config fields.

    Returns:
        Validated PipelineConfig instance.
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        p = Path(config_path)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {p.resolve()}")
        with open(p, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                data = loaded

    data.update(overrides)
    return PipelineConfig(**data)
