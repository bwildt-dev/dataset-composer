"""CLI entry point for the dataset composer.

The run is fully described by the YAML config file; only operational knobs
that vary per invocation of the same config live on the command line.

All runs are resumable; kill and rerun at any point.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from . import build_classifier, compose, compose_from_niftis, init_wandb, load_config, merge_shard_states
from .config import ComposerConfig, config_to_dict
from .converter import find_dcm2niix

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dataset-composer",
        description="AI-driven DICOM → NIfTI composer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    p.add_argument("--shard", type=str, default=None, metavar="K/N",
                   help="Classify-only: process NIfTIs where index %% N == K. "
                        "Run N of these in parallel (a SLURM array, or any "
                        "other way of launching N tasks); each shard has its "
                        "own state file and wandb run. Example: --shard 0/4.")
    p.add_argument("--convert-only", action="store_true", dest="convert_only",
                   help="Full pipeline: stop after dcm2niix conversion (CPU-only). "
                        "Follow up with --classify-only on the same config.")
    p.add_argument("--classify-only", action="store_true", dest="classify_only",
                   help="Full pipeline: skip discover+convert and classify "
                        "`paths.scratch_dir` directly, e.g. as the second step "
                        "after a --convert-only run with the same config. "
                        "Ignored if `paths.nifti_dir` is already set.")
    p.add_argument("--merge-shards", type=int, default=None, metavar="N",
                   help="Merge N shards' state files (from prior --shard K/N "
                        "runs) into the canonical state under `paths.output_root`, "
                        "then exit without composing.")
    p.add_argument("--force", action="store_true",
                   help="With --merge-shards: discard any existing canonical "
                        "state instead of merging into it.")
    p.add_argument("--verify", action="store_true",
                   help="Validate the config, paths, and classifier backend, then exit "
                        "without composing.")
    return p


def _parse_shard(spec: str) -> tuple:
    try:
        k_str, n_str = spec.split("/")
        k, n = int(k_str), int(n_str)
        if not (0 <= k < n and n >= 1):
            raise ValueError("require 0 <= K < N and N >= 1")
    except Exception as exc:
        raise SystemExit(f"--shard must be 'K/N' with 0 <= K < N: {exc}")
    return k, n


def _verify(cfg: ComposerConfig) -> None:
    """Check paths and classifier settings without walking/converting/classifying anything.

    Args:
        cfg: Config to validate.

    Raises:
        SystemExit: If one or more problems are found.
    """
    problems = []

    if cfg.is_classify_only:
        if not cfg.paths.nifti_dir.is_dir():
            problems.append(f"paths.nifti_dir does not exist: {cfg.paths.nifti_dir}")
    else:
        if not cfg.paths.source_dir.is_dir():
            problems.append(f"paths.source_dir does not exist: {cfg.paths.source_dir}")
        try:
            find_dcm2niix(cfg.paths.dcm2niix)
        except FileNotFoundError as exc:
            problems.append(str(exc))

    try:
        build_classifier(
            name    = cfg.classifier.backend,
            device  = cfg.classifier.device,
            regions = cfg.classifier.regions,
        )
    except ValueError as exc:
        problems.append(str(exc))

    if problems:
        for p in problems:
            logger.error("verify: %s", p)
        raise SystemExit(f"verify: {len(problems)} problem(s) found")

    mode = "classify-only" if cfg.is_classify_only else "full pipeline"
    logger.info("verify: config OK (%s, backend=%s)", mode, cfg.classifier.backend)


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    if args.merge_shards is not None:
        merge_shard_states(cfg.paths.output_root, args.merge_shards, force=args.force)
        return

    if args.classify_only and not cfg.is_classify_only:
        if cfg.paths.scratch_dir is None:
            raise SystemExit("--classify-only requires paths.scratch_dir (or paths.nifti_dir) in the config.")
        cfg.paths.nifti_dir = cfg.paths.scratch_dir

    if args.verify:
        _verify(cfg)
        return

    shard = None
    if args.shard:
        k, n = _parse_shard(args.shard)
        shard = (k, n)
        # Differentiate the wandb run per shard so charts don't collide.
        # 1-indexed for human readability (shard 1 of 4, not shard 0 of 4).
        if cfg.wandb.run_name:
            cfg.wandb.run_name = f"{cfg.wandb.run_name}-shard{k + 1}of{n}"

    logger.info("Effective config:\n%s", json.dumps(config_to_dict(cfg), indent=2, default=str))

    classifier = build_classifier(
        name    = cfg.classifier.backend,
        device  = cfg.classifier.device,
        regions = cfg.classifier.regions,
    )
    id_extractor, tp_extractor = cfg.build_extractors()
    reporter = init_wandb(
        enabled  = cfg.wandb.enabled,
        project  = cfg.wandb.project,
        run_name = cfg.wandb.run_name,
        config   = config_to_dict(cfg),
        tags     = cfg.wandb.tags,
    )

    if cfg.is_classify_only:
        compose_from_niftis(
            nifti_dir           = cfg.paths.nifti_dir,
            output_root         = cfg.paths.output_root,
            classifier          = classifier,
            keep_only_relevant  = not cfg.keep_all,
            keep_phases         = cfg.keep_phases,
            id_extractor        = id_extractor,
            tp_extractor        = tp_extractor,
            reporter            = reporter,
            shard               = shard,
        )
    else:
        compose(
            source_dir          = cfg.paths.source_dir,
            output_root         = cfg.paths.output_root,
            raw_nifti_dir       = cfg.paths.scratch_dir,
            classifier          = classifier,
            dcm2niix_path       = cfg.paths.dcm2niix,
            keep_only_relevant  = not cfg.keep_all,
            keep_phases         = cfg.keep_phases,
            id_extractor        = id_extractor,
            tp_extractor        = tp_extractor,
            unzip_archives      = cfg.unzip_archives,
            delete_zips_after   = cfg.delete_zips_after,
            unzip_workers       = cfg.unzip_workers,
            reporter            = reporter,
            convert_only        = args.convert_only,
            reformat_patterns   = cfg.scanner.reformat_patterns,
            topogram_patterns   = cfg.scanner.topogram_patterns,
        )


if __name__ == "__main__":
    main()
