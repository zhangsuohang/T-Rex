import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.train_tianji_sharpa import training_arguments
from utils.checkpoint_loading import load_resume_weights
from utils.convert_tianji_sharpa import build_chunks
from utils.convert_tianji_sharpa import convert_dataset
from utils.lerobot_common import compute_chunk_delta_pose
from utils.tianji_sharpa import absolute_chunk
from utils.tianji_sharpa import build_contract
from utils.tianji_sharpa import build_features
from utils.tianji_sharpa import episode_paths


def write_mother_dataset(root):
    (root / "meta").mkdir()
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 2,
        "features": build_features((3, 8, 8), (3, 8, 40)),
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/capture_contract.json").write_text(json.dumps(build_contract()))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test bimanual"}) + "\n")
    for episode in range(2):
        frames = 20
        state = np.zeros((frames, 62), dtype=np.float32)
        state[:, 3:9] = state[:, 34:40] = [1, 0, 0, 0, 1, 0]
        state[:, 0] = np.arange(frames) / 100 + episode * 100
        state[:, 31] = np.arange(frames) / 200 + episode * 100
        targets = state.copy()
        targets[:, 0] += 0.1
        targets[:, 31] += 0.2
        targets[:, 9:31] = 0.3 + episode
        targets[:, 40:62] = 0.4 + episode
        force = np.tile(np.arange(60, dtype=np.float32) + episode * 1000, (frames, 1))
        rows = {
            "observation.state": state.tolist(),
            "action": targets.tolist(),
            "observation.joints": np.zeros((frames, 58), dtype=np.float32).tolist(),
            "observation.tactile_force": force.tolist(),
            "timestamp": (np.arange(frames) / 30).tolist(),
            "frame_index": list(range(frames)),
            "episode_index": [episode] * frames,
            "task_index": [0] * frames,
        }
        data, videos = episode_paths(root, info, episode)
        data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(rows), data)
        for key, video in videos.items():
            video.parent.mkdir(parents=True, exist_ok=True)
            size = (40, 8) if "tactile" in key else (8, 8)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
            if not writer.isOpened():
                raise OSError(video)
            for i in range(frames):
                image = np.full((8, size[0], 3), 60, dtype=np.uint8)
                if "tactile" in key:
                    for finger in range(5):
                        image[:, finger * 8 : (finger + 1) * 8] = 30 + finger * 40
                writer.write(image)
            writer.release()
    return state, targets


class TianjiSharpaTests(unittest.TestCase):
    def test_chunks_match_official_delta_math_and_do_not_cross_episode(self):
        states = np.zeros((18, 62), dtype=np.float32)
        states[:, 3:9] = [0, 1, 0, -1, 0, 0]
        states[:, 34:40] = [1, 0, 0, 0, 0, 1]
        targets = states.copy()
        targets[:, 0] = np.arange(18)
        targets[:, 32] = np.arange(18)
        targets[:, 9:31] = 0.3
        targets[:, 40:62] = 0.4
        chunks = build_chunks(states, targets)
        self.assertEqual(len(chunks), 3)
        np.testing.assert_allclose(absolute_chunk(states[0], chunks[0]), targets[:16], atol=1e-6)
        predicted = chunks[0].copy()
        predicted[:, 3:9] *= 2
        predicted[:, 34:40] *= 3
        np.testing.assert_allclose(absolute_chunk(states[0], predicted), targets[:16], atol=1e-6)
        np.testing.assert_array_equal(chunks[-1][:, 9:31], targets[2:, 9:31])
        for index, offset in ((0, 0), (1, 31)):
            base = np.eye(4)
            base[:3, :3] = np.column_stack(
                (
                    states[0, offset + 3 : offset + 6],
                    states[0, offset + 6 : offset + 9],
                    np.cross(states[0, offset + 3 : offset + 6], states[0, offset + 6 : offset + 9]),
                )
            )
            target = base.copy()
            target[:3, 3] = targets[3, offset : offset + 3]
            np.testing.assert_allclose(
                chunks[0][3, offset : offset + 9], compute_chunk_delta_pose(base, target), atol=1e-6
            )

    def test_full_conversion_uses_only_train_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mother"
            root.mkdir()
            write_mother_dataset(root)
            output = Path(tmp) / "trex"
            records, stats = convert_dataset(root, output, train_episodes=[0], val_episodes=[1])
            self.assertEqual(len(records["train"]), 5)
            self.assertEqual(len(records["validation"]), 5)
            self.assertEqual(np.shape(records["train"][0]["action"]), (16, 62))
            self.assertLess(stats["rlbench"]["state"]["max"][0], 1)
            self.assertLess(max(stats["rlbench"]["tactile_f6"]["max"]), 100)
            self.assertEqual(
                json.loads((output / "train_statistics.json").read_text()),
                json.loads((output / "validation_statistics.json").read_text()),
            )
            sample = records["train"][0]
            self.assertIn("right_wrist", sample["input_image_fast"][0])
            self.assertIn("left_wrist", sample["input_image_fast"][1])
            self.assertIn("left_deform_0", sample["tactile_image_deform"][0])
            self.assertIn("right_deform_0", sample["tactile_image_deform"][5])
            intensity = [
                cv2.imread(str(output / path), cv2.IMREAD_GRAYSCALE).mean()
                for path in sample["tactile_image_deform"][:5]
            ]
            self.assertTrue(all(a < b for a, b in zip(intensity, intensity[1:])))
            with self.assertRaises(ValueError):
                convert_dataset(root, Path(tmp) / "bad", train_episodes=[0], val_episodes=[0])

    def test_strict_resume_rejects_dropped_action_or_tactile_weights(self):
        model = torch.nn.Linear(62, 4)
        original = {key: value.clone() for key, value in model.state_dict().items()}
        load_resume_weights(model, original, strict=True, report=lambda *_: None)
        with self.assertRaisesRegex(ValueError, "shape"):
            load_resume_weights(model, {"weight": torch.ones(4, 32), "bias": torch.zeros(4)}, strict=True)
        with self.assertRaisesRegex(ValueError, "missing"):
            load_resume_weights(model, {"weight": original["weight"]}, strict=True)

    def test_posttrain_profile_inherits_architecture_and_strict_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "training_args.json").write_text(
                json.dumps(
                    {
                        "action_dim": 62,
                        "action_chunk": 16,
                        "use_robot_state": 0,
                        "use_tactile_vqvae": 1,
                        "use_tactile_vec": 1,
                        "use_tactile_deform": 1,
                    }
                )
            )
            (checkpoint / "model.pt").touch()
            args = training_arguments(checkpoint, "qwen", "train.json", "outputs", "validation.json")
            self.assertIn("--strict_resume", args)
            self.assertEqual(args[args.index("--resume_source") + 1], "midtrain")
            self.assertEqual(args[args.index("--action_dim") + 1], "62")
            self.assertEqual(args[args.index("--use_robot_state") + 1], "0")


if __name__ == "__main__":
    unittest.main()
