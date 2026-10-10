from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from qwen_vla.mcap_dataset import McapWindows
from qwen_vla.mcap_dataset import TRexMcapDataset
from qwen_vla.mcap_dataset import compute_stats
from utils.lerobot_common import KEY_ACTION
from utils.lerobot_common import KEY_HEAD
from utils.lerobot_common import KEY_TACF6
from utils.test_tianji_mcap import write_episode
from utils.test_tianji_mcap import write_manifest


class Processor:
    """Exercise collate structure without downloading a language model."""

    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, messages, **kwargs):
        return "native head-only test"

    def __call__(self, *, images, **kwargs):
        result = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.zeros(len(images), 3),
            "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long),
        }
        return Batch(result)

    def image_processor(self, images, **kwargs):
        return SimpleNamespace(
            pixel_values=torch.zeros(len(images), 3),
            image_grid_thw=torch.ones(len(images), 3, dtype=torch.long),
        )


class Batch(dict):
    def __getattr__(self, name):
        return self[name]


class NativeTRexTest(unittest.TestCase):
    def test_windows_stats_and_head_only_tactile_collate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root / "train.mcap", failed_frame=4)
            write_episode(root / "validation.mcap", offset=10_000_000_000)
            manifest = write_manifest(
                root,
                [
                    {"path": "train.mcap", "task": "train", "split": "train"},
                    {
                        "path": "validation.mcap",
                        "task": "validation",
                        "split": "validation",
                    },
                ],
            )
            windows = McapWindows(manifest, horizon=3, flare_steps=1)
            self.assertEqual(windows.samples, [(0, 0), (0, 1), (0, 5), (0, 6)])
            stats_path = root / "stats.json"
            compute_stats(manifest, stats_path, horizon=3)
            config = SimpleNamespace(
                data_path=str(manifest),
                mcap_stats_path=str(stats_path),
                mcap_fps=30,
                action_dim=62,
                action_chunk=3,
                image_size=(8, 6),
                use_flare=1,
                n_flare_steps=1,
                flare_frame_stride=2,
                use_tactile_vec=1,
                use_tactile_deform=1,
                use_tactile_vqvae=1,
                use_robot_state=0,
                vqvae_window=3,
            )
            dataset = TRexMcapDataset(
                config, Processor(), SimpleNamespace(print=lambda *args: None)
            )
            item = dataset[2]
            self.assertEqual(item[KEY_HEAD].shape, (2, 3, 6, 8))
            self.assertEqual(item[KEY_TACF6].shape, (3, 10, 6))
            np.testing.assert_allclose(item[KEY_ACTION][0, :3], [0.1, 0, 0], atol=1e-7)
            batch = dataset.collate_fn([item])
            self.assertEqual(batch["norm_actions"].shape, (1, 3, 62))
            self.assertEqual(batch["tactile_f6_history"].shape, (1, 3, 10, 6))
            self.assertEqual(batch["tactile_deforms"].shape, (1, 10, 1, 2, 2))
            self.assertEqual(batch["n_slow_images"], 1)
            self.assertTrue(torch.isfinite(batch["norm_actions"]).all())
