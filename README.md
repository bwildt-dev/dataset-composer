# dataset-composer

DICOM → NIfTI dataset assembly.

Three-phase, fully resumable pipeline:

1. **Discover** — walk a DICOM tree, drop reformats/scouts/wrong-modality
   series via cheap header checks.
2. **Convert** — run surviving series through `dcm2niix`.
3. **Classify** — label each NIfTI's body region and contrast phase using
   BOA (TotalSegmentator + `boa_contrast`), then copy keepers into
   `<output>/<phase>/<patient_id>/<t0|t1>/`.

Each phase checkpoints to disk (`conversion_state.json`,
`classification_state.json`).

## Install

```bash
uv add "dataset-composer @ git+https://github.com/bwildt-dev/dataset-composer"
```

## Usage

See [`example_config.yaml`](example_config.yaml) for the full config schema.

```python
from dataset_composer import build_classifier, compose, load_config

cfg = load_config("composer.yml")
classifier = build_classifier(cfg.classifier.backend, device=cfg.classifier.device,
                               regions=cfg.classifier.regions)
compose(cfg.paths.source_dir, cfg.paths.output_root, cfg.paths.scratch_dir, classifier)
```

Classify-only mode, for a directory that's already NIfTI:

```python
from dataset_composer import compose_from_niftis, build_classifier

classifier = build_classifier("boa")
compose_from_niftis(nifti_dir, output_root, classifier)
```

## CLI

The `dataset-composer` console script is config-driven; only per-invocation
knobs live on the command line:

```bash
dataset-composer --config composer.yml                          # run full pipeline
dataset-composer --config composer.yml --convert-only           # CPU: convert only
dataset-composer --config composer.yml --classify-only          # GPU: then classify
dataset-composer --config composer.yml --shard 0/4              # one of N parallel tasks
dataset-composer --config composer.yml --merge-shards 4         # merge after all shards finish
dataset-composer --config composer.yml --verify                 # validate and exit
```

`--shard K/N` speeds up classify-only mode by splitting a NIfTI backlog
across N parallel tasks. The split is path-hashed and deterministic, but
that's for throughput, not reproducibility. Each shard writes its own state
file; merge them with `--merge-shards N` once all tasks finish.

## Requirements
`dcm2niix` and `boa_contrast` are **not** package dependencies and are never
installed automatically:

- `dcm2niix` is looked up on `PATH` (or passed explicitly) at conversion
  time; install it yourself and `--verify` will confirm it's found.
- `boa_contrast` is only imported the first time `BOAClassifier.classify()`
  actually runs; install it (plus a GPU for TotalSegmentator inference)
  yourself if you use the `boa` backend.
