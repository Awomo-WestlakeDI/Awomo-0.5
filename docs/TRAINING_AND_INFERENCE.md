# Awomo-0.5 Training and Inference

This document contains the operational instructions for installing Awomo-0.5, training AMO-Image or AMO-Video, evaluating the five benchmark families, and running inference.

## Installation and quickstart

Install the pinned environment and package from the repository root:

```bash
<environment-manager> install -f <environment-file>
<environment-manager> run -n <environment-name> pip install -e .
```

Run one inference request or one benchmark evaluation with a released checkpoint:

```bash
<inference-entrypoint> \
  --model <checkpoint-path-or-identifier> \
  --input <observation-or-episode-path> \
  --instruction "<task instruction>"

<evaluation-entrypoint> \
  --model <checkpoint-path-or-identifier> \
  --benchmark <libero-plus|molmospaces|robotwin2.0|robodojo|robocasa> \
  --config <evaluation-config>
```

The environment file, entrypoint names, example checkpoints, and supported accelerator versions will replace these placeholders with the first runnable release.

## Training

Training selects the architecture through its configuration:

```bash
<train-entrypoint> --config <path-to-amo-image-or-video-config> --output-dir <run-directory>
```

Each run records the dataset revision, configuration, code commit, random seed, checkpoint selection, and hardware information.

## Evaluation

Evaluation runs one benchmark adapter at a time. The supported benchmarks are LIBERO Plus, MolmoSpaces, RoboTwin 2.0, RoboDojo, and RoboCasa.

RoboTwin 2.0 evaluations must state whether the run uses the `full data` or `clean 2 random` checkpoint. An evaluation record includes the model name, checkpoint role, benchmark revision, task split, seed policy, success metric, and aggregate result.

## Inference

Inference returns the action sequence expected by the selected simulator. Runtime configuration controls frame sampling, image preprocessing, action horizon, precision, and device placement. Image and video models use the same action output contract.

## Reproducibility

Every published checkpoint or benchmark result retains:

1. Git commit and configuration file.
2. Base checkpoint and training data revision.
3. Benchmark revision, task split, and random seed policy.
4. Inference precision, device type, runtime version, and aggregate metrics.

Results without this metadata are development observations and are not release records.
