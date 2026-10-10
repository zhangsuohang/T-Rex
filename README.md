<div align="center">

# 🦖T-Rex: Tactile-Reactive Dexterous Manipulation

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[🌐 **Project Page**](https://tactile-reactive-dexterous.github.io/) | [✍️ **Paper (arXiv)**](https://arxiv.org/abs/2606.17055) | [🤗 **Model**](#-model-zoo) | [🤗 **Dataset (Hugging Face)**](https://huggingface.co/datasets/zekaiwang/trex_dataset)

Dantong Niu<sup>1,2*</sup>, Zhuoyang Liu<sup>1*</sup>, Zekai Wang<sup>1*</sup>, Boning Shao<sup>1</sup>, Zhao-Heng Yin<sup>1</sup>, Anirudh Pai<sup>1</sup>, Yuvan Sharma<sup>1</sup>, Stefano Saravalle<sup>5</sup>, Ruijie Zheng<sup>2</sup>, Jing Wang<sup>2</sup>, Ryan Punamiya<sup>2</sup>, Mengda Xu<sup>2</sup>, Yuqi Xie<sup>2</sup>, Yunfan Jiang<sup>2,3</sup>, Letian Fu<sup>1</sup>, Konstantinos Kallidromitis<sup>4</sup>, Matteo Gioia<sup>5,6</sup>, Junyi Zhang<sup>1</sup>, Jiaxin Ge<sup>1</sup>, Haiwen Feng<sup>1</sup>, Fabio Galasso<sup>5,6</sup>, Wei Zhan<sup>1</sup>, David M. Chan<sup>1</sup>, Yutong Bai<sup>1</sup>, Roei Herzig<sup>1</sup>, Jiahui Lei<sup>1</sup>, Li Fei-Fei<sup>3</sup>, Ken Goldberg<sup>1</sup>, Jitendra Malik<sup>1</sup>, Pieter Abbeel<sup>1</sup>, Yuke Zhu<sup>2</sup>, Danfei Xu<sup>2</sup>, Jim (Linxi) Fan<sup>2</sup>, Trevor Darrell<sup>1</sup>

<sup>1</sup>UC Berkeley &nbsp;&nbsp; <sup>2</sup>NVIDIA &nbsp;&nbsp; <sup>3</sup>Stanford &nbsp;&nbsp; <sup>4</sup>Panasonic &nbsp;&nbsp; <sup>5</sup>La Sapienza University &nbsp;&nbsp; <sup>6</sup>ItalAI

<sup>*</sup>Equal Contribution

</div>

<p align="center">
  <img src="asset/teaser.png" width="95%">
</p>

**T-Rex pushes the frontier of *tactile-reactive* dexterous manipulation** —
reacting dynamically to high-frequency touch, which contemporary VLAs typically
overlook or capture only with static tactile encoders.

> **Abstract.** The ability to react dynamically to tactile signals has long been
> considered crucial to agile human-level dexterity. Yet contemporary
> learning-based VLAs for robotic manipulation generally either overlook the
> tactile modality or are limited to encoders with static cues — in part due to
> the scarcity of diverse training data and standardized evaluation, architectural
> constraints in current Vision-Language-Action (VLA) models, and limitations of
> static tactile encoders. In this paper, we push the frontier of tactile-reactive
> manipulation, addressing all of these limitations. We collect a large-scale,
> 100-hour tactile-reactive dataset via a novel, data-efficient recipe that prioritizes
> elementary motor primitives, and open-source a ~50-hour subset. To effectively exploit naturally
> high-frequency touch signals without sacrificing the existing capabilities of
> existing VLAs, we introduce a variable-rate Mixture-of-Transformer (MoT)
> architecture equipped with a novel temporal tactile VQ-VAE encoder. We
> demonstrate the effectiveness of tactile-reactive policies on 12 manipulation
> tasks requiring delicate force control and deformable object manipulation,
> achieving over 30% higher average success rate than the strongest baseline.

### Highlights

- **100-hour tactile-reactive dataset**, collected with a data-efficient recipe that
  prioritizes elementary motor primitives (22 primitives, 200+ objects, 7700+
  trajectories); **~50 hours open-sourced** in [LeRobot v3.0](#lerobot-v30-data-path-opt-in) format.
- **Asynchronous Mixture-of-Transformers (MoT)** on a Qwen3-VL-2B backbone:
  *latent* (reason), *action*, and *tactile* experts running at different rates —
  slow action denoising (~5 Hz) and fast tactile refinement (~20 Hz) — coupled by
  **cascaded flow matching** so the policy reacts to contact *within* an action
  chunk without re-running the vision stack.
- **Temporal tactile VQ-VAE** that tokenizes high-frequency force/deformation over
  time; embedded in the model and encoded on the fly (no offline code baking).
- **> 30% higher average success** than the strongest baseline across 12
  contact-rich tasks (delicate force control, deformable-object manipulation).

The full method trains in three stages — large-scale tactile-free **pretrain** →
tactile-reactive **midtrain** → task-specific **post-train**.

> **This (`main`) branch ships the post-training + inference code only.** We
> release the **pretrained and midtrained checkpoints** (below), so you start
> directly at post-training and fine-tune on your own task. The pretraining /
> midtraining code lives in the [`full-pipeline`](../../tree/full-pipeline) branch;
> the pretrain/midtrain corpora are not part of this release.

## 🤗 Model Zoo

Checkpoints released on the Hugging Face Hub:

| Checkpoint | Stage | Notes |
|---|---|---|
| [`miniFranka/T-Rex_pretrain_mecka22k_epoch1`](https://huggingface.co/miniFranka/T-Rex_pretrain_mecka22k_epoch1) | Pretrain | VLM-action alignment on ~22k tactile-free episodes (1 epoch); action + latent experts. |
| [`miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6`](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6) | Midtrain | Tactile-reactive (cascaded flow + embedded VQ-VAE), 6 epochs. **Start here to fine-tune on your own task** (set as `RESUME_CHECKPOINT` for `scripts/train.sh`). |

The midtrain checkpoint embeds the tactile VQ-VAE, so post-train auto-detects it
(no separate `VQVAE_CKPT` needed) and encodes tactile codes on the fly.

## Dataset Quickstart

The **T-Rex Dataset** public release — ~50 hours, 5,400+ trajectories (22 motor primitives, 200+
objects) on a bimanual Dexmate Vega-1 with two Sharpa Wave dexterous hands — is released as a
[LeRobotDataset v3.0](https://github.com/huggingface/lerobot) on the
[🤗 Hub](https://huggingface.co/datasets/zekaiwang/trex_dataset). The dataset contains head, left wrist, and right wrist RGB videos; state and action stored as current and target joint positions; 10 per-fingertip image-based tactile sensor raw grayscale images, estimated deform maps, and estimated 6-dimensional wrenches. 

<p align="center">
  <img src="asset/dataset_overview.png" width="95%">
  <br>
  <em>One episode from each of 20 motor primitives (head-camera view, cropped to the workspace), each with a different object.</em>
</p>

[**`dataset_quickstart/`**](dataset_quickstart/README.md) is a standalone companion to **browse, inspect, and replay** the dataset *without*
downloading the whole thing: a Colab-friendly notebook, per-episode selective download, and 3D
replay on the real URDFs. See [`dataset_quickstart/README.md`](dataset_quickstart/README.md) for the
full per-feature schema and installation (including the third-party URDF setup).

Try the quickstart notebook in your browser:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZhuoyangLiu2005/T-Rex/blob/main/dataset_quickstart/quickstart.ipynb)

## Hardware & teleoperation stack

[**`hardware_code/`**](hardware_code/README.md) is the complete data-collection stack that
recorded T-Rex: Manus glove + VIVE tracker teleoperation of the bimanual Vega-1 with whole-arm
IK and collision avoidance, camera/tactile streaming, and synchronized episode recording
(HDF5 + MP4 + losslessly compressed tactile videos), plus the robot-side
inference client for the slow/fast protocol server
([`hardware_code/eval/`](hardware_code/eval/README.md)). See
[`hardware_code/README.md`](hardware_code/README.md) for the system diagram, hardware
requirements, installation (uv/conda), and the step-by-step launch guide.

## Repository layout

```
T-Rex/
├── qwen_vla/                       three-expert MoT model + VLA wrapper
│   ├── modeling_qwen3vl_mot.py     Qwen3VLAttentionMoT, decoder layer, MoT model
│   ├── modeling_vla.py             Qwen3VLVLAModel: ViT + MoT + embedders +
│   │                               forward_flow_action_{full,partial},
│   │                               tactile_flow_continue, tactile_flow_train_step
│   ├── diffusion.py                ActionEmbedder, TimestepEmbedder, FinalLayer
│   ├── DeformAE.py                 DeformEncoder for tactile-deformation images
│   └── lerobot_dataset.py          LeRobot v3.0 dataloader (TRexLeRobotDataset)
├── tactile_vqvae/                  tactile VQ-VAE model (used by the embedded tokenizer)
├── scripts/                        post-train + ZMQ inference server
│   ├── train.sh      + train.py       post-train SFT (fine-tune from a midtrain ckpt)
│   └── test.sh       + test.py        ZMQ inference server
├── utils/                          data prep + checkpoint tooling
│   ├── gen_json_tac_deltabase_eef_bimanual_parallel.py + gen_json_bimanual.sh
│   │                                  raw task data → training JSON (eef-62)
│   ├── convert_inlab_to_lerobot.py (+ .sh)   raw task data → LeRobot v3.0 (eef-62)
│   ├── lerobot_common.py             shared schema + pose math + norm stats
│   ├── encode_vqvae_codes_to_json.py (+ .sh)   optional code pre-baker
│   ├── merge_vqvae_into_ckpt.py      (+ .sh)   bake VQ-VAE into a checkpoint
│   └── analyze_episode.py            per-episode visualization
├── config/sft_qwen.yaml            accelerate + DeepSpeed config
├── dataset_quickstart/             standalone companion: browse / inspect / replay the dataset
├── hardware_code/                  teleoperation + data-collection stack (robot hardware code)
└── pyproject.toml                  pinned dependencies
```

> Pretraining/midtraining scripts (`pretrain.*`, `midtrain.*`,
> `prepare_midtrain_merged.py`, `convert_egodex_to_lerobot.*`) live in the
> [`full-pipeline`](../../tree/full-pipeline) branch.

## Install

```bash
conda create -n trex python=3.10 -y
conda activate trex
# torch first, from the CUDA-12.4 index:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# everything else (pinned in pyproject.toml; transformers>=4.57 for Qwen3-VL):
pip install -e .
# optional — only if you train/convert with the LeRobot v3.0 data path:
pip install -e /path/to/lerobot
```

Each `.sh` has an **editable header** at the top — set `PROJECT_ROOT`, the conda
env path, and the data/checkpoint paths there for your machine (the scripts add
`PROJECT_ROOT` to `PYTHONPATH` themselves). There is no need to export anything
globally.

## Post-training & inference

Fine-tune the released **midtrain** checkpoint on your own task, then serve it.
Edit the path variables at the top of each `.sh`, then run it.

| Step | Script | Key vars to set (top of script) | What it does |
|---|---|---|---|
| **Post-train** | `scripts/train.sh` | `DATA_JSON` (or `LEROBOT_ROOT`), `ORIGIN_MODEL_PATH`, `DEFORM_ENCODER_PATH`, `RESUME_CHECKPOINT` | Task-specific fine-tune on a small JSON or LeRobot dataset, resuming from the released midtrain checkpoint. Tactile codes are encoded on the fly; the embedded VQ-VAE is auto-detected from the checkpoint (no `VQVAE_CKPT` needed). `RESUME_SOURCE=midtrain` keeps the tactile expert as-is. |
| **Inference** | `scripts/test.sh` | `MODEL_PATH` | ZMQ REP server speaking the slow/fast cascaded protocol. Auto-detects architecture + embedded VQ-VAE from the checkpoint's `training_args.json`. |

Each `.sh` is a plain script: paths are direct variable assignments at the top,
the conda env + exports are in the header, and the launch command follows. Only
the multi-node knobs read the environment (`MASTER_ADDR`, `MASTER_PORT`,
`NUM_MACHINES`, `MACHINE_RANK` — see [Multi-node](#multi-node-distributed-launch)).

### Slow / fast protocol

The inference server (`scripts/test.py`) is a single
ZMQ REP socket with three request modes:

- `mode="slow"` — `_run_slow` calls `forward_flow_action_partial(num_steps_total, split_step)`, caches the `[latent | action]` KV at τ_split plus the partially-denoised `x_split`. Returns no actions.
- `mode="fast"` — `_run_fast` clones the cached KV, takes fresh tactile (F6 + deform; the embedded VQ-VAE tokenizes the raw F6 history from a server-side rolling 16-frame buffer — or, for a legacy external-VQ-VAE checkpoint, encodes codes with the separate `VQVAE_CKPT`), runs the remaining `total - split` Euler steps via `tactile_flow_continue`, and returns the denormalised action chunk.
- `mode="slow_and_fast"` — both back-to-back; typical at chunk start.

The ablation `--disable_tactile 1` swaps the slow tick for
`forward_flow_action_full` (full τ ∈ [0, 1] on the action expert alone)
and is the cleanest "without tactile expert" baseline.

The robot-side client that drives this server on the real Vega-1 (REQ socket,
slow every chunk start, tactile-only fast ticks in between) is
[`hardware_code/eval/eval_trex_async.py`](hardware_code/eval/README.md).

## Data preparation (your own task data)

Post-training runs on **your own task episodes**; T-Rex's pretrain/midtrain
corpora are not part of this release. Bring raw episodes laid out as
`<root>/success/episode_*/` (each: a `.h5` + 3 `.mp4` — head + left/right wrist)
and convert them to one of two formats, selected by `--data_format`:
- **`json`** (default) — a per-task training JSON. See [JSON data path](#json-data-path).
- **`lerobot`** (opt-in) — a LeRobot v3.0 dataset directory. See below.

Either way, tactile codes are encoded on the fly (embedded VQ-VAE), so no code
pre-baking is required — see the **VQ-VAE tactile codes** section below.

### JSON data path

`utils/gen_json_tac_deltabase_eef_bimanual_parallel.py` builds the per-task
training JSON (eef-62 delta-base) + a sibling `_statistics.json` from raw
episode dirs. Edit the paths in `utils/gen_json_bimanual.sh` and run it, or call
it directly:

```bash
python utils/gen_json_tac_deltabase_eef_bimanual_parallel.py \
    --data_roots /path/to/raw/task_a /path/to/raw/task_b \
    --img_save_root /path/to/training_data/images \
    --json_save_root /path/to/training_data/json \
    --task_name place_card_lr_bimanual_stride1 \
    --json_name_base place_card_deltabase_axis_eef_lr_bimanual_stride1_train \
    --instruction "Pick up the card ..." \
    --num_workers 16
```

`--data_roots` takes one or more roots (merged). No tactile-code baking is
needed — the model encodes codes on the fly (see below).

### LeRobot v3.0 data path (opt-in)

Alternatively, convert the same raw task episodes to a **LeRobot v3.0** dataset
and train with `--data_format lerobot`. Edit the paths at the top of
`utils/convert_inlab_to_lerobot.sh` (`DATA_ROOTS`, `OUTPUT_ROOT`, `REPO_ID`,
`LEROBOT_SRC`), then run it:

```bash
bash utils/convert_inlab_to_lerobot.sh   # multiple DATA_ROOTS are merged into one dataset
```

The conversion writes a standard LeRobot v3.0 tree plus a
`meta/trex_norm_stats.json` sidecar (q01/q99 + tracking_error), keeping
normalization byte-identical to the JSON pipeline. Schema (`build_trex_features`
in `utils/lerobot_common.py`): `observation.images.{head,wrist_right,wrist_left}`,
`observation.state[62]`, `action[16,62]` (baked delta-base chunk), `action_abs[62]`,
`observation.tactile_f6[10,6]`, and 10 per-finger deform videos
`observation.tactile_deform.{l,r}{0..4}`.

To train on it, set `DATA_FORMAT="lerobot"` and `LEROBOT_ROOT=/data/lerobot/...`
at the top of `scripts/train.sh`, then `bash scripts/train.sh`. The model,
cascaded-flow loss, and training loop are unchanged — the loader
(`qwen_vla/lerobot_dataset.py`) emits the same batch dict as the JSON dataset and
the embedded VQ-VAE tokenizes the raw F6 history that LeRobot `delta_timestamps`
supplies. Requires the `lerobot` package importable (`pip install -e /path/to/lerobot`).

### VQ-VAE tactile codes (optional — on-the-fly is the default)

By default the model **encodes tactile codes on the fly** from the raw F6
window via its embedded VQ-VAE (the trainers run with `--use_tactile_vqvae 1`),
so **no code pre-baking is required** — `gen_json` / the LeRobot converters
already emit code-free data with raw `tactile_f6`.

Pre-baking codes into the JSON is now **optional / legacy** (e.g. to skip
encoding at train time). If you want it, edit `INPUT_JSON` / `VQVAE_CKPT` at the
top of `utils/encode_vqvae_codes_to_json.sh` and run it, or call directly:

```bash
python -m utils.encode_vqvae_codes_to_json \
    --input_json /path/<task>_train.json \
    --output_json /path/<task>_train_vqvae_k64.json \
    --vqvae_ckpt /path/vqvae_f6_w16_k64_finger/latest.pt
```

The output adds a `tactile_codes` field (per-finger ckpt → 10 codes/sample;
per-hand → 2). When such codes are present the loader uses them; otherwise it
encodes on the fly.

## Tactile VQ-VAE

A separate 1-D conv VQ-VAE over rolling F6 windows. See
`tactile_vqvae/README.md` for training / eval / extract.

There are two ways the VLA consumes it. **B (embedded, on-the-fly) is the
default** the training scripts use.

**A. Pre-computed codes (legacy / offline).** Bake a `tactile_codes` field into
the post-train JSON with `utils/encode_vqvae_codes_to_json.py`; at inference, the
server can instead load a separate `VQVAE_CKPT` and encode a rolling F6 buffer
each fast tick.

**B. Embedded VQ-VAE (on-the-fly, default).** The VQ-VAE encoder + quantizer + F6
normalization stats live *inside* the model (`Qwen3VLVLAModel.tactile_vqvae`,
frozen). Training and inference pass a **raw** F6 history window
(`[B, window, 10, 6]`) and the model tokenizes it internally via
`encode_tactile_f6_history` — no `tactile_codes.h5`, no JSON baking, no
separate VQ-VAE at deploy time. The released midtrain checkpoint already embeds
it; the codes are bit-identical to path A for the same window.

**Auto-detect on resume.** When you resume a checkpoint that was merged with an
embedded VQ-VAE (its `training_args.json` has `use_tactile_vqvae=1` +
`vqvae_config`), `train.py` enables path B automatically and takes the VQ-VAE
weights from `model.pt` — so no `--vqvae_ckpt` is needed. The collate is a true
fallback: if the data still carries pre-baked `tactile_codes` they are used,
otherwise codes are encoded on the fly.

To convert an existing path-A checkpoint (trained with `--use_tactile_code 1`)
into a self-contained path-B checkpoint, merge the VQ-VAE weights in:

```bash
python utils/merge_vqvae_into_ckpt.py \
    --vla_ckpt   /path/checkpoint-99-12345 \
    --vqvae_ckpt /path/vqvae_f6_w16_k64_finger_XXXX/latest.pt \
    --output     /path/checkpoint-99-12345-vqvae
```

This writes `tactile_vqvae.*` weights + `tacf6_vqvae_{min,max,mask}` buffers
into `model.pt` and sets `use_tactile_vqvae=1` + `vqvae_config` in
`training_args.json`, so the inference server auto-detects the embedded
tokenizer and `VQVAE_CKPT` is no longer required.

## Checkpoint compatibility

Checkpoints follow this layout:

```
checkpoint-{epoch}-{step}/
├── model.pt              accelerator.get_state_dict(model)
├── processor/            HF processor.save_pretrained(...)
├── config.json           Qwen3-VL config
├── training_args.json    flag values needed to re-instantiate the model
├── stats_data.json       per-dataset action / state / tactile normalisation
└── state/                accelerator.save_state() — optimizer, scheduler, RNG
   training_state.json    epoch, global_step, LR, warmup_rates, min_lr_ratio
```

At inference, `test.py` reads `training_args.json` and auto-restores
`tactile_intermediate_size`, `n_flare_tokens_per_frame`, `n_flare_steps`,
`use_tactile_code`, `vqvae_codebook_size`, `use_tactile_vqvae`, `vqvae_config`,
`cascaded_total_steps`, and `cascaded_split_step` — so flags don't need to be
repeated on the CLI, and an embedded VQ-VAE is rebuilt automatically. `train.py`
likewise auto-detects an embedded VQ-VAE from the resume checkpoint.

`Qwen3VLVLAModel` loads checkpoints with `strict=False` and a
shape-mismatch filter, so checkpoints produced by earlier builds with
slightly different tactile dimensions still load (the mismatched layers
fall back to init values and you'd typically resume training to refit
them).

## Multi-node distributed launch

Every training script honours these env vars:

```bash
export MASTER_ADDR=<rank-0 IP>      # `ifconfig` → eth0 inet on the master
export MASTER_PORT=29500
export NUM_MACHINES=4               # total nodes
export MACHINE_RANK=0               # 0 on master, 1, 2, ... on the others
```

`NUM_PROCESSES` is computed as `NUM_MACHINES * 8` (assumes 8 GPUs/node). For a
different GPU count, edit the `CUDA_VISIBLE_DEVICES` and `NUM_PROCESSES` lines at
the top of the script.

Effective batch size = `train_bsz_per_gpu × NUM_PROCESSES × gradient_accumulation_steps`.

## Logging

W&B is optional. The scripts default to `export WANDB_MODE=offline`; to enable
online logging, set `WANDB_MODE=online` and `WANDB_API_KEY=<key>` in the script
header (or your shell).

## License

Released under the **MIT License** — see the [`LICENSE`](LICENSE) file at the repo root.

## Citation

If you find T-Rex useful, please cite:

```bibtex
@misc{trex2026,
  title={T-Rex: Tactile-Reactive Dexterous Manipulation}, 
  author={Dantong Niu and Zhuoyang Liu and Zekai Wang and Boning Shao and Zhao-Heng Yin and Anirudh Pai and Yuvan Sharma and Stefano Saravalle and Ruijie Zheng and Jing Wang and Ryan Punamiya and Mengda Xu and Yuqi Xie and Yunfan Jiang and Letian Fu and Konstantinos Kallidromitis and Matteo Gioia and Junyi Zhang and Jiaxin Ge and Haiwen Feng and Fabio Galasso and Wei Zhan and David M. Chan and Yutong Bai and Roei Herzig and Jiahui Lei and Fei-Fei Li and Ken Goldberg and Jitendra Malik and Pieter Abbeel and Yuke Zhu and Danfei Xu and Jim Fan and Trevor Darrell},
  year={2026},
  eprint={2606.17055},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2606.17055}, 
}
```
<!-- TODO: replace the title / author / arXiv id above with the real values. -->


## Tianji + Sharpa bimanual adaptation

The `feat/tianji-sharpa-bimanual` branch consumes processed LeRobot **v2.1** mother data. Each sample uses
`[left xyz3, left column-pair Rot6D6, left hand22, right xyz3, right column-pair Rot6D6, right hand22]`.
`observation.state` is measured absolute TCP/hand state; `action` is the recorded absolute commanded target.
Positions use metres and joints use radians. Rot6D is `[R[:,0], R[:,1]]`. There is no atomic-skill channel.

The capture protocol is now **released v1.0.0**, defined by
[`configs/tianji_sharpa/capture_contract.json`](configs/tianji_sharpa/capture_contract.json).
[`features.json`](configs/tianji_sharpa/features.json) is the v2.1 writer input;
[`example_frame.json`](configs/tianji_sharpa/example_frame.json) is a synthetic numeric example;
[`hand_reference.json`](configs/tianji_sharpa/hand_reference.json) defines canonical hand angle origins/axes.
These artifacts are identical across all three policy repositories. Hardware models, SDKs, URDF and calibration
references are optional provenance. The producer maps its hardware into the protocol; the consumer never requires
the producer's SDK or URDF. This supersedes the previous unpublished draft.

Use `build_features()` for tuple shapes, or convert JSON `shape` lists to tuples before calling the pinned writer.
Copy the contract to `DATASET_DIR/meta/capture_contract.json`. Record one entry per episode in
`meta/capture_episodes.jsonl`: episode_index, session_id, task_index and outcome(success/failure/unknown).
The standard info/tasks/episodes/episodes_stats metadata and indices are generated by the v2.1 writer.
`observation.joints` is measured `[left arm7, left hand22, right arm7, right hand22]`.
`action.joints` is optional, contains actual joint commands, and its feature must be added explicitly when available.
Hand blocks must agree with state/action respectively within 1e-4 rad. Raw 58D joint vectors never substitute for EE62.

The base frame is fixed per episode: origin is the midpoint between arm mounting bases at episode start,
+X robot forward horizontally, +Z opposite gravity, +Y=Z cross X (robot left).
The rigid TCP is at the hand wrist mounting plane center: +Z toward middle MCP in the palm plane,
+X outward dorsal normal, +Y=Z cross X. Both measured and commanded poses use this TCP/frame.
R maps TCP vectors into base coordinates; Rot6D concatenates the FIRST TWO COLUMNS, not rows.
Hand q0..21 follow the named joints and zero/positive-axis definitions in the supplied reference;
left/right are robot sides. Do not assume SDK order, signs or degree units match.

RGB keys are observation.images.head/left_hand/right_hand. CHW metadata accompanies HWC uint8 RGB arrays.
Deform keys are left_tactile/right_tactile: five horizontal square grayscale-RGB tiles, thumb to pinky,
tile top toward fingertip and tile right along +pad X. Fixed-calibration intensity d in [0,1] becomes round(255*d),
with zero no-contact baseline; no per-frame min/max. Raw sensor RGB is not a deformation map.
F6 is left30 then right30, Fx/Fy/Fz/Mx/My/Mz per finger, N and N*m, environmental wrench ON the finger.
Each distal pad frame has +Z outward, +Y toward fingertip, +X=Y cross Z; compressive Fz is negative.
Raw bit depth/ranges can be archived separately. RGB defaults to360x640 and deform to240x1200;
other even resolutions are allowed only when consistently declared in the contract and info.json.
All videos are H.264 MP4 at30fps, one decoded frame per Parquet row.

Rows use timestamp=n/30 and contiguous frame_index from0. New required vectors are
observation.source_timestamp11, observation.available_timestamp11, observation.valid11,
action.command_timestamp4 and action.valid4. See features.json for exact stream order.
Observation source <= availability <= row timestamp; state/joints/deform/F6 age <=50ms, RGB <=100ms.
Action is the accepted/clipped/filtered absolute target assigned to that cycle, not the next measured state;
its command timestamp cannot exceed the next cycle boundary. A recorded held target retains its original timestamp.
Missing/stale/invalid data ends the episode; resume in a new episode. No interior-frame deletion or zero fillers.
Tasks are nonempty <=512-character instructions and remain constant within an episode.
Store session_id for split grouping and compute model-specific normalization on training episodes only.
Physical calibration, sensor identity, force units and latest-sample correctness remain the producer's responsibility;
the validator checks machine-observable conformance and cannot establish physical truth from recorded numbers.

Write capture files, conversion outputs and training checkpoints on a local filesystem, then upload
finished artifacts to OSS. Virtual environments and caches also belong outside the OSS worktree.


The new converter preserves the existing T-Rex 62D model and horizon16. It emits T-Rex JSON and
statistics from v2.1 mother data without installing LeRobot v3 in this environment. Choose episode splits
before conversion; only training episodes contribute statistics, and validation reuses those statistics.
Chunks stop at the last complete horizon, rather than supervising targets beyond the episode boundary.
Fast RGB order is right wrist then left wrist; state/actions and tactile order remain left then right.
Deform tiles are resized to 240x240 for the inherited encoder.

```bash
uv run --no-sync python -m utils.convert_tianji_sharpa \
  --dataset-dir "${DATASET_DIR}" --output-dir "${LOCAL_EXPORT_DIR}" \
  --train-episodes 0 1 2 --val-episodes 3
# Activate this project's environment and supply your existing Accelerate configuration.
# This entrypoint sets no GPU count.
accelerate launch --config_file "${ACCELERATE_CONFIG}" scripts/train_tianji_sharpa.py \
  --checkpoint-dir "${MODEL_ROOT}/t-rex/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6" \
  --model-path "${MODEL_ROOT}/qwen/Qwen3-VL-2B-Instruct" \
  --data-path "${LOCAL_EXPORT_DIR}/train.json" \
  --validation-data-path "${LOCAL_EXPORT_DIR}/validation.json" \
  --output-dir "${LOCAL_RUN_ROOT}/trex" --learning_rate 1e-4
```

`train_tianji_sharpa.py` inherits the action/tactile/FLARE/VQ-VAE architecture flags from the checkpoint,
including use_robot_state0 for the downloaded release, and enables `--strict_resume`. Missing, unexpected
or shape-incompatible inherited tensors fail instead of being silently skipped. Existing `scripts/train.sh`
and its distributed settings remain unchanged. This initializes a new post-training run from midtrain
weights; it does not resume optimizer state. `utils.tianji_sharpa.absolute_chunk()` restores denormalized
model outputs into the 62D absolute TCP/hand boundary before the Tianji controller performs IK.

Regression: `python -m unittest utils.test_tianji_sharpa -v`.
Synthetic Parquet/video conversion, episode isolation, training-only statistics and strict resume failures
were tested. A real Qwen processor and SftDataset collate also loaded 62D/H16 targets, three RGB views,
ten deform tiles, VQ-VAE F6 history and eight FLARE targets. The actual downloaded model.pt was inspected against the full model on the meta device:
1,412 keys, no missing/unexpected keys or shape conflicts. The checkpoint values were not run through a
full-model optimizer step, and no Tianji hardware motion was executed.


### Capture protocol commands

```bash
# Activate this project environment; create producer templates in a new directory.
python -m utils.tianji_sharpa --output-dir "${PROTOCOL_OUTPUT_DIR}"
# Before training/conversion, validate all numeric rows, tasks and decoded video frames.
python -m utils.tianji_sharpa --validate-dataset "${DATASET_DIR}"
```

Use H.264 explicitly when the pinned writer would otherwise default to AV1. Do not write generated files
over the release templates; for alternate resolutions call build_features/build_contract with matching shapes.


### Native Tianji/Sharpa MCAP

Copy `configs/tianji_sharpa/native_mcap.json`, set portable MCAP paths and task
instructions, and declare complete episodes as train/validation. Paths are relative
to the manifest. The T-Rex dataset constructs its own 30Hz/H16 windows directly;
no LeRobot/JSON image export is required.

```bash
source env.sh
python -m qwen_vla.mcap_dataset --manifest "${NATIVE_MANIFEST}" --output "${NATIVE_STATS}"
python scripts/train_tianji_sharpa.py \
  --checkpoint-dir "${MIDTRAIN_CHECKPOINT}" --model-path "${QWEN_MODEL}" \
  --data-format mcap --data-path "${NATIVE_MANIFEST}" --mcap-stats-path "${NATIVE_STATS}" \
  --output-dir "${LOCAL_RUN_DIR}"
```

With validation episodes declared, add `--val_ratio 0.05 --val_freq 500` to enable
validation; the value does not randomly split frames. Normalization is computed
from training episodes only. Architecture flags still come from the checkpoint.
There is one slow head RGB view and no wrist views. Ten DEFORM images and the
checkpoint's 16-frame F6 history use left/right and thumb/index/middle/ring/pinky
order. F6 history retains the checkpoint's model-grid time spacing, not 16 native
180Hz samples. FLARE targets use future head frames within the same episode.

Poses use the recorded Pinocchio model world and `EE_0/EE_1`, not a physically
calibrated palm TCP. Source-time observation sampling preserves receipt times
for latency analysis; online delivery-delay augmentation is a separate task.
`utils.tianji_mcap.McapEpisode.tactile_history` also exposes full-rate RAW/DEFORM/F6.
The current model consumes DEFORM/F6; RAW is available to future encoder changes.
