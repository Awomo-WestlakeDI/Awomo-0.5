<div align="center">

<p>
  <img src="assets/logo.svg" width="120" alt="西湖数智 Logo">
</p>

<h1>Awomo-0.5: World Action Model Pre-trained at Scale</h1>

<p><b>🖼️ Awomo-0.5I</b> | <b>🎥 Awomo-0.5V</b> | <a href="https://robodojo-benchmark.com/LeaderBoard"><b>🥈 #2 on RoboDojo</b></a></p>

<p>🤗 Hugging Face: <a href="https://huggingface.co/Auwomo"><b>Auwomo</b></a> | 🤖 ModelScope: <b>TBD</b> | 📄 Paper: <b>TBD</b> | 🖥️ Demo: <b>TBD</b></p>

<p><a href="#introduction">Introduction</a> | <a href="#checkpoints">Checkpoints</a> | <a href="#documentation">Documentation</a> | <a href="#citation">Citation</a></p>

</div>

## 📖 Introduction

Awomo-0.5 is a codebase for training, evaluating, and running embodied policies in simulation. It provides two architecture variants with a shared action interface:

- **Awomo-0.5I** (Awomo-0.5-Image) processes image observations.
- **Awomo-0.5V** (Awomo-0.5-Video) processes video observations and temporal context.

The release targets five embodied simulation benchmarks: LIBERO Plus, MolmoSpaces, RoboTwin 2.0, RoboDojo, and RoboCasa.
This release contains the inference code and the RoboCasa365 evaluation; the training code and the other benchmarks will be released progressively.

## ✨ Highlights

- 2 architecture variants under one training and evaluation interface (training code coming).
- 5 benchmark adapters for embodied simulation evaluation (RoboCasa365 released, others coming).
- 14 planned model releases.

## 🔔 News

- 🔥 **2026.10.03**: Release RoboDojo checkpoint and evaluation code.

## 📦 Checkpoints

Overall success rates are shown in parentheses.

| **Awomo-0.5I** | **Awomo-0.5V** |
| :-------- | :---------------- |
| Base | Base |
| LIBERO Plus (93.5) | LIBERO Plus (91.1) |
| MolmoSpaces (65.3) | MolmoSpaces (64.5) |
| RoboTwin 2.0 Full (96.1) | RoboTwin 2.0 Full |
| RoboTwin 2.0 Clean-2-Random (79.2) | RoboTwin 2.0 Clean-2-Random |
| RoboDojo (35.3, [🥈 #2](https://robodojo-benchmark.com/LeaderBoard)) | RoboDojo |
| [RoboCasa365](eval_robocasa365/README.md) (64.4, [🤗](https://huggingface.co/Auwomo/Awomo-0.5I-RoboCasa365)) | [RoboCasa365](eval_robocasa365/README.md) (62.5, [🤗](https://huggingface.co/Auwomo/Awomo-0.5V-RoboCasa365)) |

Released checkpoints are on [Hugging Face](https://huggingface.co/Auwomo); each includes `awomo_config.json` and `SHA256SUMS`.

## 📚 Documentation

- [Deployment](docs/DEPLOYMENT.md): policy server environment and startup.
- [RoboCasa365 evaluation](eval_robocasa365/README.md)

## 📜 License

The code in this repository is released under the [Apache License 2.0](LICENSE). The model weights on
[Hugging Face](https://huggingface.co/Auwomo) are released under CC BY-NC 4.0 (see the `LICENSE.md` of each checkpoint).
Base models downloaded by the code (Qwen3-VL-4B-Instruct, FLUX.2-klein-base-4B, Wan2.2-TI2V-5B) remain under their
own licenses.

## 🤝 Acknowledgements

Parts of this repository are adapted from [PatchWAM](https://github.com/TianhengW/PatchWAM).

## 📝 Citation

```bibtex
@software{awomo05,
  author  = {{Awomo-WAM Team}},
  title   = {Awomo-0.5},
  version = {0.5},
  year    = {2026},
}
```
