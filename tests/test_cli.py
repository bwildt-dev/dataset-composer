"""Unit tests for cli.py: argument parsing, --verify, and main() dispatch."""

from __future__ import annotations

import json

import pytest
import yaml

from dataset_composer.cli import _parse_shard, _verify, build_parser, main
from dataset_composer.config import load_config

from conftest import make_fake_dcm2niix, make_nifti, make_series_dir


def _write_config(tmp_path, data, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_build_parser_requires_config():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_build_parser_defaults():
    args = build_parser().parse_args(["--config", "cfg.yaml"])
    assert args.shard is None
    assert args.convert_only is False
    assert args.classify_only is False
    assert args.merge_shards is None
    assert args.force is False
    assert args.verify is False


def test_build_parser_flags_parse():
    args = build_parser().parse_args([
        "--config", "cfg.yaml", "--shard", "1/4", "--convert-only", "--verify",
    ])
    assert args.shard == "1/4"
    assert args.convert_only is True
    assert args.verify is True


def test_parse_shard_valid():
    assert _parse_shard("1/4") == (1, 4)
    assert _parse_shard("0/1") == (0, 1)


@pytest.mark.parametrize("spec", ["4/4", "-1/4", "1/0", "not-a-spec", "1/2/3"])
def test_parse_shard_invalid_raises_systemexit(spec):
    with pytest.raises(SystemExit):
        _parse_shard(spec)


def test_verify_classify_only_missing_nifti_dir_raises(tmp_path):
    cfg = load_config(_write_config(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path / "missing")},
    }))
    with pytest.raises(SystemExit):
        _verify(cfg)


def test_verify_classify_only_existing_dir_passes(tmp_path):
    (tmp_path / "niftis").mkdir()
    cfg = load_config(_write_config(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path / "niftis")},
        "classifier": {"backend": "dummy"},
    }))
    _verify(cfg)  # must not raise


def test_verify_full_pipeline_missing_source_dir_raises(tmp_path):
    (tmp_path / "scratch").mkdir()
    cfg = load_config(_write_config(tmp_path, {
        "paths": {
            "output_root": str(tmp_path / "out"),
            "source_dir": str(tmp_path / "missing_src"),
            "scratch_dir": str(tmp_path / "scratch"),
        },
        "classifier": {"backend": "dummy"},
    }))
    with pytest.raises(SystemExit):
        _verify(cfg)


def test_verify_full_pipeline_missing_dcm2niix_raises(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "scratch").mkdir()
    monkeypatch.setattr("shutil.which", lambda name: None)
    cfg = load_config(_write_config(tmp_path, {
        "paths": {
            "output_root": str(tmp_path / "out"),
            "source_dir": str(tmp_path / "src"),
            "scratch_dir": str(tmp_path / "scratch"),
        },
        "classifier": {"backend": "dummy"},
    }))
    with pytest.raises(SystemExit):
        _verify(cfg)


def test_verify_bad_classifier_backend_raises(tmp_path):
    (tmp_path / "niftis").mkdir()
    cfg = load_config(_write_config(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path / "niftis")},
        "classifier": {"backend": "not-a-backend"},
    }))
    with pytest.raises(SystemExit):
        _verify(cfg)


def test_main_verify_only_exits_cleanly(tmp_path, capsys):
    (tmp_path / "niftis").mkdir()
    config_path = _write_config(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path / "niftis")},
        "classifier": {"backend": "dummy"},
    })
    main(["--config", str(config_path), "--verify"])  # must not raise or compose anything
    assert not (tmp_path / "out").exists()


def test_main_merge_shards_dispatches_to_merge(tmp_path):
    root = tmp_path / "out"
    shard_dir = root / "_state_shard0of1"
    shard_dir.mkdir(parents=True)
    (shard_dir / "classification_state.json").write_text(json.dumps({"a.nii.gz": {"phase": "venous"}}))

    config_path = _write_config(tmp_path, {
        "paths": {"output_root": str(root), "nifti_dir": str(tmp_path / "niftis")},
    })
    main(["--config", str(config_path), "--merge-shards", "1"])

    merged = json.loads((root / "_state" / "classification_state.json").read_text())
    assert "a.nii.gz" in merged


def test_main_classify_only_end_to_end(tmp_path):
    make_nifti(tmp_path / "niftis" / "P1__t0__aaaa1111.nii.gz")
    config_path = _write_config(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path / "niftis")},
        "classifier": {"backend": "dummy"},
    })
    main(["--config", str(config_path)])
    assert (tmp_path / "out" / "venous" / "P1" / "t0" / "P1__t0__aaaa1111.nii.gz").exists()


def test_main_full_pipeline_end_to_end(tmp_path):
    dcm2niix = make_fake_dcm2niix(tmp_path, "success")
    make_series_dir(tmp_path / "src" / "P1", "abdomen_series", n_slices=25)
    config_path = _write_config(tmp_path, {
        "paths": {
            "output_root": str(tmp_path / "out"),
            "source_dir": str(tmp_path / "src"),
            "scratch_dir": str(tmp_path / "raw"),
            "dcm2niix": str(dcm2niix),
        },
        "classifier": {"backend": "dummy"},
        "id": {"patterns": [r"(P\d+)"]},
        "unzip_archives": False,
    })
    main(["--config", str(config_path)])
    kept = list((tmp_path / "out").rglob("*.nii.gz"))
    assert len(kept) == 1


def test_main_classify_only_flag_overrides_full_pipeline_config(tmp_path):
    dcm2niix = make_fake_dcm2niix(tmp_path, "success")
    make_series_dir(tmp_path / "src", "abdomen_series", n_slices=25)
    config_path = _write_config(tmp_path, {
        "paths": {
            "output_root": str(tmp_path / "out"),
            "source_dir": str(tmp_path / "src"),
            "scratch_dir": str(tmp_path / "raw"),
            "dcm2niix": str(dcm2niix),
        },
        "classifier": {"backend": "dummy"},
        "unzip_archives": False,
    })
    # --convert-only first to populate scratch_dir with a NIfTI, no classification.
    main(["--config", str(config_path), "--convert-only"])
    assert not (tmp_path / "out" / "_state" / "classification_state.json").exists()

    # --classify-only on the same config should pick up scratch_dir as nifti_dir.
    main(["--config", str(config_path), "--classify-only"])
    assert (tmp_path / "out" / "_state" / "classification_state.json").exists()


def test_main_classify_only_with_shard_writes_shard_state(tmp_path):
    make_nifti(tmp_path / "niftis" / "P1__t0__aaaa1111.nii.gz")
    config_path = _write_config(tmp_path, {
        "paths": {"output_root": str(tmp_path / "out"), "nifti_dir": str(tmp_path / "niftis")},
        "classifier": {"backend": "dummy"},
        "wandb": {"enabled": False, "run_name": "myrun"},
    })
    main(["--config", str(config_path), "--shard", "0/1"])
    assert (tmp_path / "out" / "_state_shard0of1").exists()
