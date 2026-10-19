# SFT file inventory

The training package provides Awomo-0.5I SFT on RoboDojo.
Model definitions live in `awomo/awomo_05i`; dataset and preprocessing code live in `awomo/data/robodojo`.

## User-facing training directory

| File | Role |
|---|---|
| `training/README.md` | Training index |
| `training/FILES.md` | This inventory |
| `training/robodojo/README.md` | RoboDojo SFT instructions |
| `training/robodojo/train.py` | Fresh SFT and full-state resume entry |
| `training/robodojo/prepare_data.py` | Build an index from local HDF5 episodes |
| `training/robodojo/episodes.txt` | Ordered relative paths for the training episodes |
| `training/robodojo/configs/sft.yaml` | RoboDojo SFT configuration |
| `training/robodojo/dataset_stats.json` | Supplied normalization parameters |

## Training library

The training library contains 52 Python files, including 11 package initializers, with checkpoint, distributed training and resume support.

| Directory under `awomo/training` | Files | Purpose |
|---|---|---|
| `.` | `__init__.py`, `builders.py`, `ema.py`, `frozen_state_ema.py`, `garbage_collection.py`, `gradient_utils.py`, `gripper_loss.py`, `losses.py`, `lr.py`, `rng.py`, `robodojo.py`, `sampler.py`, `signal_handler.py`, `step_scheduler.py`, `trainer.py` | SFT recipe, model construction, forward/loss, optimizer-step coordination, EMA, clipping, LR/RNG/sampler state. |
| `checkpoint` | `__init__.py`, `addons.py`, `checkpointing.py`, `conversion_mapping.py`, `stateful_wrappers.py`, `utils.py` | Full-state checkpoint/save/resume and supported dense-model metadata. |
| `checkpoint/_backports` | `__init__.py`, `_fsspec_filesystem.py`, `consolidate_hf_safetensors.py`, `filesystem.py`, `hf_storage.py`, `hf_utils.py` | Storage adapters used by the pinned PyTorch checkpoint implementation. |
| `config` | `__init__.py`, `_arg_parser.py`, `loader.py` | Configuration loading and command-line parsing. |
| `distributed` | `__init__.py`, `config.py`, `fsdp.py`, `init_utils.py`, `mesh.py`, `mesh_utils.py` | Distributed initialization and FSDP device meshes. |
| `loggers` | `__init__.py`, `log_utils.py`, `wandb_utils.py` | Training logging and optional experiment tracking. |
| `model_utils` | `__init__.py`, `hf_checkpointing_mixin.py`, `utils.py` | Shared model/precision flags and checkpoint export integration. |
| `optim` | `__init__.py`, `scheduler.py` | Optimizer parameter scheduler. |
| `shared` | `__init__.py`, `import_utils.py`, `torch_patches.py`, `utils.py` | Dtype/import helpers and compatibility patches. |
| `transformer_utils` | `__init__.py`, `mfu.py` | Device throughput/compute metadata required by the current trainer imports. |
| `utilities` | `__init__.py`, `flops_utils.py` | Shared FLOP helpers imported by the current training infrastructure. |

## Verification boundary

CPU tests cover the sampler, weighted loss, action representation, direct checkpoint loading, shared-model input contracts and existing RoboCasa interfaces. Dataset tests additionally check preprocessing tensors. These checks do not constitute a new multi-node training or closed-loop benchmark result.

DataLoader resume tests cover worker RNG, prefetch progress, augmentation, history sampling and parent-process random draws with unchanged loader settings. Older checkpoints without these snapshots restore window positions only.
