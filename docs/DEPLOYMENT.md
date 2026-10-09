# Deployment

The policy server runs in its own environment; each benchmark's simulator runs in another and talks to the server
over TCP (see `deploy/client.py`).

## Server environment

```bash
conda create -n awomo python=3.11 -y && conda activate awomo
pip install -r requirements-05i.txt   # Awomo-0.5I
# or, in a separate environment: pip install -r requirements-05v.txt   # Awomo-0.5V
```

## Start servers

```bash
bash scripts/deploy.sh <policy module> <checkpoint dir> <num_servers> [num_gpus]
# e.g. bash scripts/deploy.sh eval_robocasa365.awomo_05i.policy checkpoints/Awomo-0.5I-RoboCasa365 8
```

Server `i` listens on port `18000 + i`. Base models are downloaded from Hugging Face on first start at the revisions
pinned in the checkpoint's `awomo_config.json`; pass local copies with `SERVER_ARGS`:

- Awomo-0.5I: `SERVER_ARGS="--qwen-path <Qwen3-VL-4B-Instruct dir> --vae-path <FLUX.2 VAE dir>"`
- Awomo-0.5V: `SERVER_ARGS="--wan-path <Wan2.2-TI2V-5B dir>"`

One server uses about 25 GB of GPU memory.

## Benchmarks

- [RoboCasa365](../eval_robocasa365/README.md)
