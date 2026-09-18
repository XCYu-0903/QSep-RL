<div align="center">
  <h1>QSep-RL</h1>

  <p>
    <strong>Xincheng Yu, Shizhong Zhou, Jiankun Li, Ting Lei, Jianwei Zhang, and Yi Lin*</strong>
  </p>

  <p>
    <a href="#"><img src="https://img.shields.io/badge/Paper-coming_soon-blue?style=for-the-badge" alt="Paper"></a>
    <a href="https://XCYu-0903.github.io/QSep-RL-demo/"><img src="https://img.shields.io/badge/Demo-online-green?style=for-the-badge" alt="Demo"></a>
    <a href="https://github.com/XCYu-0903/QSep-RL"><img src="https://img.shields.io/badge/Code-QSep--RL-orange?style=for-the-badge" alt="Code"></a>
  </p>
</div>


## Architecture

<div align="center">
  <img src="assets/QSepRL.png" width="900" alt="QSep-RL architecture">
</div>

## Structure

- `train_qseprl.py`: training entry. It supports training from scratch and resuming from split checkpoints.
- `model/QSepRL.py`: QSep-RL model.
- `model/CLAPSep_decoder.py`: decoder / mask network.
- `data_utils/`: AudioCaps mixture dataset utilities.
- `scripts/prepare_audiocaps_json.py`: converts AudioCaps metadata and local audio paths into JSON files used by training.
- `experiments/QSep-RL/config.json`: default config template.

## Prepare AudioCaps JSON

The training code expects JSON metadata with this schema:

```json
[
  {
    "file_name": "example.wav",
    "path": "/absolute/path/to/example.wav",
    "caption": "a dog barking",
    "mids": "",
    "class_names": ""
  }
]
```

Generate train and validation JSON files from AudioCaps CSV files:

```bash
python scripts/prepare_audiocaps_json.py \
  --audiocaps_csv /path/to/audiocaps_train.csv \
  --audio_root /path/to/AudioSet \
  --output_json metadata/audiocaps_train.json \
  --missing_json metadata/audiocaps_train_missing.json

python scripts/prepare_audiocaps_json.py \
  --audiocaps_csv /path/to/audiocaps_val.csv \
  --audio_root /path/to/AudioSet \
  --output_json metadata/audiocaps_val.json \
  --missing_json metadata/audiocaps_val_missing.json
```

If you use ontology filtering, place the AudioSet ontology JSON at the path configured by `ontology_path` in `experiments/QSep-RL/config.json`.

## Training

Defaults are defined in `train_qseprl.py`. Values in `experiments/QSep-RL/config.json` override those defaults, and command-line arguments override both. By default, RL and GRPO are enabled and the copied encoder uses trainable LoRA.

For multi GPUs run:

```bash
torchrun --nproc_per_node=2 train_qseprl.py \
  --exp_dir ./experiments/QSep-RL \
  --run_name qseprl
```

For a single GPU quick run:

```bash
python train_qseprl.py \
  --exp_dir ./experiments/QSep-RL \
  --run_name qseprl_debug \
  --batch_size 32
```

## Resume

Resume from the latest split checkpoint under `cp/<run_name>`:

```bash
python train_qseprl.py \
  --exp_dir ./experiments/QSep-RL \
  --run_name qseprl \
  --auto_resume true
```

Edit `experiments/QSep-RL/config.json` to set `clap_path`, `input_dir`, metadata JSON paths, and ontology path for your machine.

## Acknowledgements

We referred to [CLAPSep](https://github.com/Aisaka0v0/CLAPSep) and [MARS-Sep](https://github.com/mars-sep/MARS-Sep) to implement this.

## Contact Us

If you are interested in leaving a message to our research team, feel free to email xinchengyu@alu.scu.edu.cn.

<p align="center">
  <img src="assets/scu_logo.png" height="130" align="middle" alt="SCU logo">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/wiseatc_logo.png" height="120" align="middle" alt="WiseATC logo">
</p>
