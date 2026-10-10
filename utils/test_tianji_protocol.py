import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils import tianji_sharpa as protocol


def rows_for_episode(count=20):
    rows = {key: [] for key in protocol.example_frame()}
    for index in range(count):
        frame = protocol.example_frame()
        for key in ("observation.source_timestamp", "observation.available_timestamp", "action.command_timestamp"):
            frame[key] = [value + index / 30 for value in frame[key]]
        frame["timestamp"] = index / 30
        frame["frame_index"] = index
        for key, value in frame.items():
            rows[key].append(value)
    return rows


def write_dataset(root):
    (root / "meta").mkdir()
    features = protocol.build_features((3, 8, 8), (3, 8, 40))
    info = {"codebase_version": "v2.1", "fps": 30, "total_episodes": 1, "total_frames": 20, "features": features}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/capture_contract.json").write_text(json.dumps(protocol.build_contract((3, 8, 8), (3, 8, 40))))
    (root / "meta/capture_episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "session_id": "session_test", "task_index": 0, "outcome": "unknown"}) + "\n"
    )
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Move both hands forward"}) + "\n")
    rows = rows_for_episode()
    fields = {}
    for key, value in rows.items():
        dtype = pa.list_(pa.from_numpy_dtype(np.dtype(features[key]["dtype"]))) if key in features else None
        fields[key] = pa.array(value, type=dtype)
    parquet, videos = protocol.episode_paths(root, info, 0)
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.table(fields), parquet)
    for key, path in videos.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = features[key]["shape"][1:]
        pixels = np.full((20, height, width, 3), 80, dtype=np.uint8)
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                "30",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            input=pixels.tobytes(),
            check=True,
        )
    return parquet, videos


class ProtocolTests(unittest.TestCase):
    def test_valid_rows_and_optional_real_joint_targets(self):
        rows = rows_for_episode()
        self.assertEqual(protocol.validate_numeric_episode(rows, episode_index=0, task_index=0), 20)
        rows["action.joints"] = [[0.0] * 58 for _ in range(20)]
        self.assertEqual(protocol.validate_numeric_episode(rows), 20)

    def test_held_command_requires_an_unchanged_target(self):
        rows = rows_for_episode()
        rows["action.command_timestamp"] = [[0.005] * 4 for _ in range(20)]
        assert protocol.validate_numeric_episode(rows) == 20
        rows["action"][1][0] += 0.1
        with self.assertRaisesRegex(ValueError, "held command"):
            protocol.validate_numeric_episode(rows)

    def test_rejects_future_stale_and_invalid_observations(self):
        for field, value, pattern in (
            ("observation.available_timestamp", 1.0, "Future"),
            ("observation.source_timestamp", -1.0, "Stale"),
            ("observation.valid", False, "Invalid"),
            ("action.valid", False, "Invalid"),
            ("action.command_timestamp", 1.0, "future control"),
        ):
            with self.subTest(field=field):
                rows = rows_for_episode()
                rows[field][0][0] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    protocol.validate_numeric_episode(rows)

    def test_rejects_gap_bad_rotation_and_disagreeing_hand_blocks(self):
        for change, pattern in (
            (lambda rows: rows["timestamp"].__setitem__(1, 0.2), "30 Hz"),
            (lambda rows: rows["frame_index"].__setitem__(1, 2), "contiguous"),
            (lambda rows: rows["observation.state"][0].__setitem__(3, 0), "Rot6D"),
            (lambda rows: rows["observation.joints"][0].__setitem__(7, 0.5), "hand joint"),
            (lambda rows: rows["episode_index"].__setitem__(1, 1), "episode"),
            (lambda rows: rows["task_index"].__setitem__(1, 1), "Task"),
        ):
            with self.subTest(pattern=pattern):
                rows = rows_for_episode()
                change(rows)
                with self.assertRaisesRegex(ValueError, pattern):
                    protocol.validate_numeric_episode(rows)
        rows = rows_for_episode()
        rows["action.joints"] = [[0.0] * 58 for _ in range(20)]
        rows["action.joints"][0][36] = 0.5
        with self.assertRaisesRegex(ValueError, "Commanded hand"):
            protocol.validate_numeric_episode(rows)

    def test_complete_parquet_video_validation_needs_no_hardware_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            path = root / "meta/capture_contract.json"
            contract = json.loads(path.read_text())
            del contract["optional_provenance"]
            path.write_text(json.dumps(contract))
            result = protocol.validate_dataset(root)
            self.assertEqual(result["frames"], 20)
            self.assertEqual(result["protocol_version"], "1.0.0")
            contract["coordinate_frame"] = "right_arm_base"
            path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "coordinate_frame"):
                protocol.load_metadata(root)

    def test_bad_video_and_wrong_numeric_dtype_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet, videos = write_dataset(root)
            table = pq.read_table(parquet)
            column = table.schema.get_field_index("observation.state")
            table = table.set_column(
                column, "observation.state", table["observation.state"].cast(pa.list_(pa.float64()))
            )
            pq.write_table(table, parquet)
            with self.assertRaisesRegex(ValueError, "Parquet dtype"):
                protocol.validate_dataset(root)
            table = table.set_column(
                column, "observation.state", table["observation.state"].cast(pa.list_(pa.float32()))
            )
            pq.write_table(table, parquet)
            path = videos["observation.images.left_tactile"]
            pixels = np.zeros((20, 8, 40, 3), dtype=np.uint8)
            pixels[..., 0] = 255
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "40x8",
                    "-framerate",
                    "30",
                    "-i",
                    "pipe:0",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(path),
                ],
                input=pixels.tobytes(),
                check=True,
            )
            with self.assertRaisesRegex(ValueError, "grayscale"):
                protocol.validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
