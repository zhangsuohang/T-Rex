"""T-Rex builds its own action, tactile-history and FLARE windows from MCAP."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from qwen_vla.lerobot_dataset import TRexLeRobotDataset
from utils.lerobot_common import DEFORM_KEYS
from utils.lerobot_common import KEY_ACTION
from utils.lerobot_common import KEY_HEAD
from utils.lerobot_common import KEY_STATE
from utils.lerobot_common import KEY_TACF6
from utils.lerobot_common import NormStatsAccumulator
from utils.tianji_mcap import ARM_COMMAND
from utils.tianji_mcap import COMMAND_TOPICS
from utils.tianji_mcap import HEAD
from utils.tianji_mcap import STATE_TOPICS
from utils.tianji_mcap import TACTILE_TOPICS
from utils.tianji_mcap import McapEpisode
from utils.tianji_mcap import load_manifest
from utils.tianji_sharpa import local_chunk


class McapWindows:
    def __init__(
        self,
        manifest,
        *,
        split="train",
        fps=30,
        horizon=16,
        flare_steps=0,
        flare_stride=2,
    ):
        if fps <= 0 or horizon <= 0 or flare_steps < 0 or flare_stride < 1:
            raise ValueError("Invalid T-Rex sampling/window configuration.")
        self.fps, self.horizon = fps, horizon
        self.flare_steps, self.flare_stride = flare_steps, flare_stride
        self.episodes, self.samples = [], []
        for entry in load_manifest(manifest, split):
            reader = McapEpisode(entry["path"])
            start, end = reader.bounds(
                (HEAD, ARM_COMMAND, *STATE_TOPICS, *COMMAND_TOPICS, *TACTILE_TOPICS)
            )
            times = start + np.rint(
                np.arange(int((end - start) * fps / 1e9) + 1) * 1e9 / fps
            ).astype(np.int64)
            states, targets = np.zeros((2, len(times), 62), dtype=np.float32)
            force = np.zeros((len(times), 10, 6), dtype=np.float32)
            valid, target_valid = np.ones((2, len(times)), dtype=bool)
            for i, time_ns in enumerate(times):
                try:
                    states[i] = reader.vector(time_ns)
                    force[i] = reader.force(time_ns)
                    reader.at(HEAD, time_ns, max_age_ns=100_000_000)
                except ValueError:
                    valid[i] = False
                try:
                    targets[i] = reader.vector(time_ns, command=True)
                except ValueError:
                    target_valid[i] = False
            episode_index = len(self.episodes)
            self.episodes.append(
                (
                    reader,
                    entry["task"],
                    times,
                    states,
                    targets,
                    force,
                    valid,
                    target_valid,
                )
            )
            span = max(horizon - 1, flare_steps * flare_stride)
            for i in range(len(times) - span):
                future = [i + k * flare_stride for k in range(flare_steps + 1)]
                if valid[future].all() and target_valid[i : i + horizon].all():
                    self.samples.append((episode_index, i))

    def __len__(self):
        return len(self.samples)

    def numeric(self, index):
        ep, i = self.samples[index]
        _, _, _, states, targets, force, _, _ = self.episodes[ep]
        return (
            local_chunk(states[i], targets[i : i + self.horizon]),
            states[i],
            force[i],
        )


class TRexMcapDataset(TRexLeRobotDataset):
    def __init__(self, config, processor, accelerator, *, split="train"):
        self.config, self.processor, self.accelerator = config, processor, accelerator
        self.image_size = tuple(config.image_size) if config.image_size else None
        self.use_flare = bool(config.use_flare)
        self.n_flare_steps = config.n_flare_steps if self.use_flare else 0
        self.flare_stride = config.flare_frame_stride
        self.use_tactile_vec = bool(config.use_tactile_vec)
        self.use_tactile_deform = bool(config.use_tactile_deform)
        self.use_tactile_vqvae = bool(config.use_tactile_vqvae)
        self.use_robot_state = bool(config.use_robot_state)
        self.vqvae_window = int(getattr(config, "vqvae_window", 16))
        self.action_dim = config.action_dim
        if self.action_dim != 62:
            raise ValueError("Native Tianji MCAP uses the 62D model-EE/hand interface.")
        self.has_wrist, self.has_tactile, self.has_deform = False, True, True
        self.ds = McapWindows(
            config.data_path,
            split=split,
            fps=config.mcap_fps,
            horizon=config.action_chunk,
            flare_steps=self.n_flare_steps,
            flare_stride=self.flare_stride,
        )
        if not len(self.ds):
            raise ValueError(f"No complete T-Rex windows in the {split} split.")
        with open(config.mcap_stats_path) as stream:
            self.stats_data = json.load(stream)
        block = self.stats_data[next(iter(self.stats_data))]
        for name, key in (
            ("action", "action"),
            ("state", "state"),
            ("tacf6", "tactile_f6"),
        ):
            for attr, stat in (("mask", "mask"), ("min", "q01"), ("max", "q99")):
                setattr(self, f"{name}_{attr}", np.asarray(block[key][stat]))
        te = block.get("tracking_error", {})
        self.te_mean = np.asarray(te.get("mean", np.zeros(56)), dtype=np.float32)
        self.te_std = np.asarray(te.get("std", np.zeros(56)), dtype=np.float32)
        accelerator.print(
            f"[MCAP] {split}: {len(self.ds)} native windows, head RGB only"
        )

    def _head_tensor(self, reader, time_ns):
        image = Image.fromarray(reader.head(time_ns))
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255

    def __getitem__(self, index):
        ep, i = self.ds.samples[index]
        reader, task, times, states, targets, force, valid, _ = self.ds.episodes[ep]
        # Retain the checkpoint's 30Hz history meaning; full-rate histories remain available on reader.
        first = i
        while first > 0 and valid[first - 1] and i - first + 1 < self.vqvae_window:
            first -= 1
        indices = np.maximum(np.arange(i - self.vqvae_window + 1, i + 1), first)
        item = {
            KEY_ACTION: local_chunk(states[i], targets[i : i + self.ds.horizon]),
            KEY_STATE: states[i].copy(),
            KEY_TACF6: torch.from_numpy(force[indices].copy()),
            KEY_HEAD: torch.stack(
                [
                    self._head_tensor(reader, times[i + k * self.flare_stride])
                    for k in range(self.n_flare_steps + 1)
                ]
            ),
            "task": task,
        }
        if self.use_tactile_deform:
            for key, image in zip(DEFORM_KEYS, reader.deform(times[i]), strict=True):
                item[key] = torch.from_numpy(image.copy()).unsqueeze(0).float() / 255
        return item

    def create_val_split(self, val_ratio=0.05, seed=42):
        # The manifest is shared by projects; no random frame-level split.
        return TRexMcapDataset(
            self.config, self.processor, self.accelerator, split="validation"
        )


def compute_stats(manifest, output, *, fps=30, horizon=16):
    dataset = McapWindows(manifest, fps=fps, horizon=horizon)
    if not len(dataset):
        raise ValueError("No complete training windows for normalization.")
    stats = NormStatsAccumulator()
    for i in range(len(dataset)):
        stats.add_frame(*dataset.numeric(i))
    for _, _, _, states, targets, _, valid, target_valid in dataset.episodes:
        # Tracking errors are pointwise; exclude invalid rows, never estimate from validation.
        mask = valid & target_valid
        stats.add_episode_tracking(states[mask], targets[mask])
    result = stats.assemble()
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()
    compute_stats(args.manifest, args.output, fps=args.fps, horizon=args.horizon)


if __name__ == "__main__":
    main()
