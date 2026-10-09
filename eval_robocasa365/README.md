# RoboCasa365 Evaluation

50 target tasks × 30 trials (1,500 episodes), `split=pretrain`, official task horizons.
The policy server and the RoboCasa365 simulator run in two separate environments.

## Prerequisites

### 1. Server environment (`awomo`)

See [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md).

### 2. RoboCasa365 environment (`robocasa365`)

Install RoboCasa and its assets following the [official guide](https://robocasa.ai/docs/introduction/installation.html),
using robocasa commit `4f8a2980def75a55dff96b990745b83540425f09` (1.0.1) and robosuite commit
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. Then:

```bash
pip install -r eval_robocasa365/requirements.txt
```

### 3. Checkpoint

| Model | Checkpoint | Policy module |
|---|---|---|
| Awomo-0.5I | [`Auwomo/Awomo-0.5I-RoboCasa365`](https://huggingface.co/Auwomo/Awomo-0.5I-RoboCasa365) | `eval_robocasa365.awomo_05i.policy` |
| Awomo-0.5V | [`Auwomo/Awomo-0.5V-RoboCasa365`](https://huggingface.co/Auwomo/Awomo-0.5V-RoboCasa365) | `eval_robocasa365.awomo_05v.policy` |

```bash
hf download Auwomo/Awomo-0.5I-RoboCasa365 --local-dir checkpoints/Awomo-0.5I-RoboCasa365
cd checkpoints/Awomo-0.5I-RoboCasa365 && sha256sum -c SHA256SUMS && cd -
# Awomo-0.5V:
hf download Auwomo/Awomo-0.5V-RoboCasa365 --local-dir checkpoints/Awomo-0.5V-RoboCasa365
cd checkpoints/Awomo-0.5V-RoboCasa365 && sha256sum -c SHA256SUMS && cd -
```

Each checkpoint contains `model.pt`, `awomo_config.json` and `SHA256SUMS`. Base models (Awomo-0.5I: Qwen3-VL-4B-Instruct
and the FLUX.2-klein-base-4B VAE; Awomo-0.5V: Wan2.2-TI2V-5B) are downloaded at the revisions pinned in
`awomo_config.json`.

## Evaluation

### Step 1 — Start the policy servers

```bash
conda activate awomo
bash scripts/deploy.sh eval_robocasa365.awomo_05i.policy checkpoints/Awomo-0.5I-RoboCasa365 8
# Awomo-0.5V: bash scripts/deploy.sh eval_robocasa365.awomo_05v.policy checkpoints/Awomo-0.5V-RoboCasa365 8
```

### Step 2 — Run the evaluation

```bash
conda activate robocasa365
bash scripts/launch_robocasa365.sh 8 out/robocasa365_05i
# Awomo-0.5V: bash scripts/launch_robocasa365.sh 8 out/robocasa365_05v
```

Results are written to `<output dir>/summary.json`, together with the model/checkpoint identity and the evaluation
config. Re-running the same command resumes unfinished episodes;
an episode that fails 3 times is listed under `errors` in `summary.json` and retried on the next run.
The launch script exits nonzero while any episode is unfinished or failed.
Use one output directory per model, checkpoint and evaluation setting (task set, trials, task list): the directory
records the policy server's identity (`identity.json`, re-checked after every reconnect) and the setting
(`config.json`), and the launch script stops with an error if either changes.

Smoke test (one episode):

```bash
TASK_SET=atomic_seen TASKS=CloseBlenderLid NUM_TRIALS=1 bash scripts/launch_robocasa365.sh 1 out/smoke_05i
```

## Results

| Model | Atomic-Seen | Composite-Seen | Composite-Unseen | Overall |
|---|---|---|---|---|
| Awomo-0.5I | 86.48 | 58.75 | 45.21 | 64.40 |
| Awomo-0.5V | 85.00 | 60.42 | 39.38 | 62.53 |

1,500 episodes per model, robocasa 1.0.1. Awomo-0.5I: RTX 5090. Awomo-0.5V: RTX 5090, except 9 of the 50 tasks on
RTX PRO 6000. Per-task results: [awomo_05i/summary.json](awomo_05i/summary.json),
[awomo_05v/summary.json](awomo_05v/summary.json).
