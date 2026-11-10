"""Unit tests for configuration schema validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from spatial_profiler.config import (
    MorphologyConfig,
    PipelineConfig,
    QCConfig,
    load_config,
)


def test_default_config_instantiation():
    """Verify default parameters match roadmap benchmark recommendations."""
    cfg = PipelineConfig()
    assert cfg.project_name == "visium_hd_human_crc"
    assert cfg.qc.min_counts == 500
    assert cfg.qc.min_genes == 250
    assert cfg.qc.max_pct_mt == 20.0
    assert cfg.qc.n_top_genes == 3000
    assert cfg.morphology.patch_size == 224
    assert cfg.morphology.backbone == "resnet50"
    assert cfg.morphology.alpha_morphology == 0.35
    assert cfg.deconvolution.doublet_mode == "full"
    assert cfg.spatial_statistics.n_neighs == 6


def test_load_config_from_yaml(temp_config_yaml: Path):
    """Test loading and overriding parameters from a YAML file."""
    cfg = load_config(temp_config_yaml)
    assert cfg.project_name == "test_suite_crc"
    assert cfg.qc.min_counts == 100
    assert cfg.morphology.patch_size == 64
    assert cfg.morphology.backbone == "resnet18"
    assert cfg.deconvolution.max_cores == 2


def test_config_overrides():
    """Verify runtime kwargs override YAML values."""
    cfg = load_config(None, project_name="custom_override_run")
    assert cfg.project_name == "custom_override_run"


def test_validation_errors():
    """Ensure invalid parameters raise Pydantic ValidationErrors."""
    # max_pct_mt must be <= 100.0
    with pytest.raises(ValidationError):
        QCConfig(max_pct_mt=150.0)

    # patch_size must be >= 32
    with pytest.raises(ValidationError):
        MorphologyConfig(patch_size=16)

    # alpha_morphology must be in [0.0, 1.0]
    with pytest.raises(ValidationError):
        MorphologyConfig(alpha_morphology=1.5)

    # empty project name should fail
    with pytest.raises(ValidationError):
        PipelineConfig(project_name="   ")
