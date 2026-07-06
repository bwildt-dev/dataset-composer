"""Unit tests for config.py: YAML loading, validation, defaults, round-trip."""

from __future__ import annotations

import pytest
import yaml

from dataset_composer.classifier import RegionSpec
from dataset_composer.config import (
    IdConfig,
    PathsConfig,
    TimepointConfig,
    config_to_dict,
    load_config,
)


def _write_yaml(tmp_path, data: dict, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_config_requires_output_root(tmp_path):
    path = _write_yaml(tmp_path, {"paths": {"source_dir": str(tmp_path)}})
    with pytest.raises(ValueError, match="output_root"):
        load_config(path)


def test_load_config_requires_source_or_nifti_dir(tmp_path):
    path = _write_yaml(tmp_path, {"paths": {"output_root": str(tmp_path / "out")}})
    with pytest.raises(ValueError, match="nifti_dir.*source_dir|source_dir.*nifti_dir"):
        load_config(path)


def test_load_config_full_pipeline_requires_scratch_dir(tmp_path):
    path = _write_yaml(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "source_dir": str(tmp_path)},
    })
    with pytest.raises(ValueError, match="scratch_dir"):
        load_config(path)


def test_load_config_classify_only_minimal(tmp_path):
    path = _write_yaml(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path / "niftis")},
    })
    cfg = load_config(path)
    assert cfg.is_classify_only is True
    assert cfg.classifier.backend == "boa"                 # default
    assert cfg.classifier.keep_phases == ["arterial", "venous"]
    assert cfg.keep_phases == ("arterial", "venous")
    assert cfg.timepoint.labels == {"t0": ["PRERAND"], "t1": ["EVAL"]}


def test_load_config_full_pipeline_minimal(tmp_path):
    path = _write_yaml(tmp_path, {
        "paths": {
            "output_root": str(tmp_path / "out"),
            "source_dir": str(tmp_path / "src"),
            "scratch_dir": str(tmp_path / "scratch"),
        },
    })
    cfg = load_config(path)
    assert cfg.is_classify_only is False
    assert cfg.paths.nifti_dir is None


def test_load_config_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPOSER_TEST_ROOT", str(tmp_path))
    path = _write_yaml(tmp_path, {
        "paths": {
            "output_root": "$COMPOSER_TEST_ROOT/out",
            "nifti_dir": "$COMPOSER_TEST_ROOT/niftis",
        },
    })
    cfg = load_config(path)
    assert str(cfg.paths.output_root) == str(tmp_path / "out")


def test_load_config_overrides_regions_and_scanner_patterns(tmp_path):
    path = _write_yaml(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path)},
        "classifier": {
            "backend": "dummy",
            "regions": [{"label": "chest", "structures": {"lung": 100}}],
            "keep_phases": ["venous"],
        },
        "scanner": {"reformat_patterns": ["FOO"], "topogram_patterns": ["BAR"]},
        "id": {"patterns": [r"SUBJ(\d+)"], "format": "SUBJ-{:03d}"},
        "keep_all": True,
    })
    cfg = load_config(path)
    assert cfg.classifier.backend == "dummy"
    assert cfg.classifier.regions == [RegionSpec("chest", {"lung": 100})]
    assert cfg.classifier.keep_phases == ["venous"]
    assert cfg.scanner.reformat_patterns == ["FOO"]
    assert cfg.scanner.topogram_patterns == ["BAR"]
    assert cfg.id.patterns == [r"SUBJ(\d+)"]
    assert cfg.keep_all is True


def test_build_extractors_uses_config_patterns(tmp_path):
    path = _write_yaml(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path)},
        "id": {"patterns": [r"SUBJ(\d+)"], "format": "SUBJ-{:03d}"},
        "timepoint": {"enabled": False},
    })
    cfg = load_config(path)
    id_ex, tp_ex = cfg.build_extractors()
    from pathlib import Path
    assert id_ex(Path("SUBJ7/scan")) == "SUBJ-007"
    assert tp_ex(Path("anything")) == ""


def test_config_to_dict_round_trips_key_fields(tmp_path):
    path = _write_yaml(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path)},
    })
    cfg = load_config(path)
    d = config_to_dict(cfg)
    assert d["paths"]["output_root"] == str(cfg.paths.output_root)
    assert d["classifier"]["backend"] == "boa"
    assert d["classifier"]["regions"] == [{"label": "abdomen", "structures": {"liver": 5000}}]


def test_paths_config_is_a_plain_dataclass():
    p = PathsConfig(output_root="/tmp/out")
    assert p.source_dir is None
    assert p.nifti_dir is None


def test_id_config_defaults_to_no_patterns():
    assert IdConfig().patterns == []
    assert IdConfig().format == "{}"


def test_timepoint_config_defaults():
    tc = TimepointConfig()
    assert tc.enabled is True
    assert tc.fallback == "unknown"
