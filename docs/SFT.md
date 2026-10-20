# Awomo-0.5I RoboDojo SFT

See [RoboDojo SFT](../training/robodojo/README.md). The benchmark entry is `training/robodojo/train.py`; its configuration is `training/robodojo/configs/sft.yaml`.

Model implementations live in `awomo/awomo_05i`. `awomo/training/trainer.py` owns the shared training lifecycle and native checkpoint recovery; `awomo/training/robodojo.py` defines `RoboDojoTrainer(Trainer)` for conditioning, loss and freeze policy. FLUX block sharding lives in `awomo/training/distributed/fsdp.py` without an additional trainer superclass. See [file inventory](../training/FILES.md).

The model accepts either separate history/current-frame conditions or packed condition tokens with explicit positions and mask. It has no benchmark-name switch. Historical RoboCasa inference calls remain supported. RoboCasa is evaluation-only; no RoboCasa SFT is added. Benchmark-specific RoboDojo evaluation is submitted separately to XPolicy.
