# RoboDojo SFT

Run the commands below from the repository root.

## Data Processing

Download the RoboDojo HDF5 dataset, then build the training index with
[`prepare_data.py`](prepare_data.py):

```bash
export AWOMO_DATA_ROOT=/datasets/RoboDojo
python training/robodojo/prepare_data.py \
  --data-root "$AWOMO_DATA_ROOT" --output "$AWOMO_DATA_ROOT/index.pt"
```

The supplied [`episodes.txt`](episodes.txt) excludes `swap_T`, `swap_blocks` and
`sweep_blocks` (100 episodes each), leaving **32 tasks and 3,200 episodes**.
Use the supplied [`dataset_stats.json`](dataset_stats.json) for normalization.

## Training

Download the [base checkpoint](https://huggingface.co/Auwomo/Awomo-0.5I-base) to `/models/awomo-0.5i-base`.
Training script: [`scripts/train_sft.sh`](../../scripts/train_sft.sh);
configuration: [`configs/sft.yaml`](configs/sft.yaml).

```bash
uv pip install -r requirements-sft.txt

export AWOMO_BASE_CHECKPOINT=/models/awomo-0.5i-base
export AWOMO_DATA_INDEX="$AWOMO_DATA_ROOT/index.pt"
export AWOMO_DATA_STATS="$PWD/training/robodojo/dataset_stats.json"
export AWOMO_QWEN3_VL=/models/Qwen3-VL-4B-Instruct
export AWOMO_FLUX2_TRANSFORMER=/models/FLUX.2-klein-base-4B/transformer
export AWOMO_FLUX2_VAE=/models/FLUX.2-klein-base-4B/vae
export AWOMO_RUN_NAME=awomo-0.5i-robodojo-sft
export AWOMO_RUN_DIR="$PWD/outputs/$AWOMO_RUN_NAME"

export NNODES=4 NODE_RANK=0 MASTER_ADDR=training-node-0 MASTER_PORT=29500
python training/robodojo/train.py --check-config
bash scripts/train_sft.sh
```

Run on four 8-GPU nodes with `NODE_RANK=0,1,2,3` respectively and shared inputs/output.

Resume with `bash scripts/train_sft.sh --resume /path/to/checkpoint`.
Keep the data, rank layout and DataLoader worker settings unchanged when resuming.
New checkpoints include worker RNG and prefetch progress for window, augmentation
and history-sampling replay. Older checkpoints without DataLoader snapshots restore
window positions only and issue a warning; their random preprocessing cannot be replayed exactly.

### Training configuration

| Setting | Value |
|---|---|
| GPUs / batch | 32 GPUs; local batch 8, global batch 256; no accumulation |
| Updates / LR | 6,755 updates; AdamW peak LR `5e-6`; warmup 270; cosine 6,755; minimum LR 0 |
| Sampling | Uniform stride 1; seed 42; fixed-budget sampler |
| Dataset / draws | 3,200 episodes; 1,724,338 windows; 1,729,280 draws including 4,942 repeated tail windows |
| Training scope | DiT: 169 tensors; state encoder, VLM/504 LoRA tensors and VAE frozen |
| EMA | 0.999; all 171 model tensors, including the frozen state encoder |
| Action | Absolute joints/grippers in Canonical80; 14 supervised slots |
| Gripper loss | Slots 16/45 numerator ×2; valid-14D denominator unchanged |
| Noise | Discrete 1,000-point FP32 timestep; shift 5; 80D noise followed by 48 zeros |
| Loss | BSMNTW on; Self-Flow off; video weight 0.5, action weight 1 |
| Images | BGR hero-top mosaic; VAE H320/W288; VLM H448/W448; color augmentation on |
| Conditioning | `prefix_v2` instruction conditioning and 20 history slots |
| Precision | FP32 master parameters/checkpoints; BF16 compute; FP32 gradient reduction |
| Validation | No holdout; training does not start evaluation |

## Evaluation

Download the [RoboDojo SFT checkpoint](https://huggingface.co/Auwomo/Awomo-0.5I-Robodojo).
Use [XPolicyLab `policy/Awomo05`](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Awomo05)
for evaluation. The adapter update for this SFT checkpoint is pending publication.
