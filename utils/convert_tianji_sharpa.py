"""Export processed LeRobot v2.1 EE62 mother data to T-Rex training JSON.

Train/validation episode lists are explicit; only training episodes determine
normalization. RGB ordering for T-Rex fast views is right, left; numerical
state/action and ten tactile fingers remain left, right.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from utils.lerobot_common import ACTION_CHUNK
from utils.lerobot_common import NormStatsAccumulator
from utils.tianji_sharpa import SIDES
from utils.tianji_sharpa import episode_paths
from utils.tianji_sharpa import load_metadata
from utils.tianji_sharpa import local_chunk
from utils.tianji_sharpa import validate_numeric_episode
from utils.tianji_sharpa import validate_vectors


def build_chunks(states, targets, horizon=ACTION_CHUNK):
    states = validate_vectors(states)
    targets = validate_vectors(targets, "targets")
    if states.ndim != 2 or states.shape != targets.shape or horizon <= 0:
        raise ValueError("Expected matching (N,62) state/target streams and a positive horizon.")
    # Avoid supervising repeated targets beyond the actual episode boundary.
    return [local_chunk(states[i], targets[i : i + horizon]) for i in range(len(states) - horizon + 1)]


def _write_image(path, values):
    if not cv2.imwrite(str(path), values):
        raise OSError(f"Could not write {path}")


def _export_video_images(videos, output, frames):
    for key, path in videos.items():
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"Could not open video {path}")
        side = "left" if "left" in key else "right"
        view = "head" if key.endswith("head") else f"{side}_wrist"
        try:
            for i in range(frames):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Video has fewer than {frames} frames: {path}")
                if "tactile" in key:
                    height, width = frame.shape[:2]
                    if width != height * 5:
                        raise ValueError("Tactile video must contain five square thumb-first tiles.")
                    for finger, tile in enumerate(np.split(frame[..., 0], 5, axis=1)):
                        # The inherited DeformEncoder projects 128*15*15,
                        # requiring a 240x240 input regardless of capture size.
                        tile = cv2.resize(tile, (240, 240), interpolation=cv2.INTER_AREA)
                        _write_image(output / f"image{i}_tactile_{side}_deform_{finger}.png", tile)
                else:
                    _write_image(output / f"image{i}_{view}.png", frame)
            if capture.read()[0]:
                raise ValueError(f"Video has more than {frames} frames: {path}")
        finally:
            capture.release()


def convert_dataset(root, output, *, train_episodes, val_episodes=()):
    root, output = Path(root), Path(output)
    info, _ = load_metadata(root)
    train_episodes, val_episodes = list(train_episodes), list(val_episodes)
    selected = train_episodes + val_episodes
    if not train_episodes or len(selected) != len(set(selected)):
        raise ValueError("Training must be nonempty; train/validation episode lists must be distinct and disjoint.")
    if any(i < 0 or i >= info["total_episodes"] for i in selected):
        raise ValueError("Episode index outside dataset range.")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tasks = {
        item["task_index"]: item["task"]
        for item in (json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip())
    }
    records = {"train": [], "validation": []}
    stats = NormStatsAccumulator()
    for index in selected:
        parquet, videos = episode_paths(root, info, index)
        rows = pq.read_table(parquet).to_pydict()
        validate_numeric_episode(rows, episode_index=index)
        states = validate_vectors(rows["observation.state"])
        targets = validate_vectors(rows["action"], "action")
        force = np.asarray(rows["observation.tactile_force"], dtype=np.float32)
        if force.shape != (len(states), 60) or not np.all(np.isfinite(force)):
            raise ValueError("Expected a finite (N,60) left-then-right tactile force stream.")
        timestamps = np.asarray(rows["timestamp"], dtype=np.float64)
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Episode timestamps must be strictly increasing and finite.")
        chunks = build_chunks(states, targets)
        if not chunks:
            raise ValueError(f"Episode {index} needs at least {ACTION_CHUNK} frames.")
        directory = output / "images" / f"episode_{index:06d}"
        directory.mkdir(parents=True)
        _export_video_images(videos, directory, len(states))
        split = "train" if index in train_episodes else "validation"
        for i, chunk in enumerate(chunks):
            relative = directory.relative_to(output)

            def name(suffix):
                return str(relative / f"image{i}_{suffix}.png")

            records[split].append(
                {
                    "episode_index": index,
                    "frame_index": i,
                    "input_prompt": tasks[int(rows["task_index"][i])],
                    "input_image_slow": [name("head")],
                    "input_image_fast": [name("right_wrist"), name("left_wrist")],
                    "action": chunk.tolist(),
                    "state_slow": states[i].tolist(),
                    "state_fast": states[i].tolist(),
                    "tactile_f6": force[i].reshape(10, 6).tolist(),
                    "tactile_image_deform": [
                        name(f"tactile_{side}_deform_{finger}") for side in SIDES for finger in range(5)
                    ],
                }
            )
            if split == "train":
                stats.add_frame(chunk, states[i], force[i])
        if split == "train":
            stats.add_episode_tracking(states, targets)
    normalized = stats.assemble()
    for split, samples in records.items():
        if samples:
            (output / f"{split}.json").write_text(json.dumps(samples, ensure_ascii=False) + "\n")
            (output / f"{split}_statistics.json").write_text(json.dumps(normalized, ensure_ascii=False) + "\n")
    (output / "episode_split.json").write_text(
        json.dumps(
            {
                "train": train_episodes,
                "validation": val_episodes,
                "stats_source": "train_only",
                "action_chunk": ACTION_CHUNK,
            },
            indent=2,
        )
        + "\n"
    )
    return records, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--val-episodes", type=int, nargs="*", default=[])
    args = parser.parse_args()
    convert_dataset(
        args.dataset_dir, args.output_dir, train_episodes=args.train_episodes, val_episodes=args.val_episodes
    )


if __name__ == "__main__":
    main()
