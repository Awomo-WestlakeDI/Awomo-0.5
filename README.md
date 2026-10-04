<div align="center">

<p>
  <img src="assets/logo.svg" width="120" alt="西湖数智 Logo">
</p>

<h1>Awomo-0.5: World Action Model Pre-trained at Scale</h1>

<p><b>🖼️ AMO-Image</b> | <b>🎥 AMO-Video</b> | <a href="https://robodojo-benchmark.com/LeaderBoard"><b>🥈 #2 on RoboDojo</b></a></p>

<p>🤗 Hugging Face: <b>TBD</b> | 🤖 ModelScope: <b>TBD</b> | 📄 Paper: <b>TBD</b> | 🖥️ Demo: <b>TBD</b></p>

<p><a href="#introduction">Introduction</a> | <a href="#checkpoints">Checkpoints</a> | <a href="#documentation">Documentation</a> | <a href="#citation">Citation</a></p>

</div>

## 📖 Introduction

Awomo-0.5 is a codebase for training, evaluating, and running embodied policies in simulation. It provides two architecture variants with a shared action interface:

- **AMO-Image** processes image observations.
- **AMO-Video** processes video observations and temporal context.

The release targets five embodied simulation benchmarks: LIBERO Plus, MolmoSpaces, RoboTwin 2.0, RoboDojo, and RoboCasa.

## ✨ Highlights

- 2 architecture variants under one training and evaluation interface.
- 5 benchmark adapters for embodied simulation evaluation.
- 14 planned model releases.

## 🔔 News

- 🔥 **2026.10.03**: Release RoboDojo checkpoint and evaluation code.

## 📦 Checkpoints

Overall success rates are shown in parentheses.

| **AMO-Image** | **AMO-Video** |
| :-------- | :---------------- |
| Base | Base |
| LIBERO Plus (93.5) | LIBERO Plus (91.1) |
| MolmoSpaces (65.3) | MolmoSpaces (64.5) |
| RoboTwin 2.0 Full (96.1) | RoboTwin 2.0 Full |
| RoboTwin 2.0 Clean-2-Random (79.2) | RoboTwin 2.0 Clean-2-Random |
| RoboDojo (35.3, [🥈 #2](https://robodojo-benchmark.com/LeaderBoard)) | RoboDojo |
| RoboCasa | RoboCasa |

Model artifact links, configuration files, and checksums will be published with the corresponding checkpoint release.

## 📚 Documentation

Operational instructions for installation, training, evaluation, inference, and reproducibility are available in [docs/TRAINING_AND_INFERENCE.md](docs/TRAINING_AND_INFERENCE.md).

## 📜 License

The Awomo-0.5 license and third-party component licenses will be listed with the public model release.

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
