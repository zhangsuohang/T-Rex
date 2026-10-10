"""Small self-contained portable MCAP fixtures, including delayed ROS delivery."""

import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from mcap_ros2.writer import Writer
import numpy as np
from PIL import Image
import pytest

from utils import tianji_mcap as native

BASE = 1_700_000_000_000_000_000
HEADER = """
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
PORTABLE = """string metadata_json
rx_recording_msgs/TypedArray[] arrays
================================================================================
MSG: rx_recording_msgs/TypedArray
string path
string dtype
uint64[] shape
uint8[] data
"""
POSE = (
    """std_msgs/Header header
geometry_msgs/Pose pose
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"""
    + HEADER
)
JOINT = (
    """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
"""
    + HEADER
)
RGB = (
    """std_msgs/Header header
string format
uint8[] data
"""
    + HEADER
)


def write_episode(path, *, frames=9, offset=0, failed_frame=None, quaternion=(0, 0, 0, 1)):
    image_bytes = io.BytesIO()
    Image.fromarray(np.full((6, 8, 3), 127, dtype=np.uint8)).save(image_bytes, format="JPEG")
    with Writer(str(path), chunk_size=1024) as writer:
        portable = writer.register_msgdef("rx_recording_msgs/PortableRecord", PORTABLE)
        pose = writer.register_msgdef("geometry_msgs/PoseStamped", POSE)
        joint = writer.register_msgdef("sensor_msgs/JointState", JOINT)
        rgb = writer.register_msgdef("sensor_msgs/CompressedImage", RGB)
        for i in range(frames):
            time_ns = BASE + offset + round(i * 1e9 / 30)
            header = {
                "stamp": {"sec": time_ns // 1_000_000_000, "nanosec": time_ns % 1_000_000_000},
                "frame_id": "world",
            }

            def write_native(topic, payload, arrays=(), *, time_ns, i):
                metadata = {
                    "data": payload,
                    "host_unix_ns": time_ns,
                    "host_monotonic_ns": i * 33_333_333,
                    "sequence": i,
                }
                writer.write_message(
                    topic,
                    portable,
                    {"metadata_json": json.dumps(metadata), "arrays": list(arrays)},
                    log_time=time_ns + 1_000_000,
                )

            write_native(
                native.ARM_STATE,
                {"snapshot": {"outputs": [{"fb_joint_pos": [90] * 7}, {"fb_joint_pos": [-90] * 7}]}},
                time_ns=time_ns,
                i=i,
            )
            write_native(
                native.ARM_COMMAND,
                {
                    "left_position_deg": [45] * 7,
                    "right_position_deg": [-45] * 7,
                    "send_result": 0 if i == failed_frame else 1,
                },
                time_ns=time_ns,
                i=i,
            )
            for side in native.SIDES:
                write_native(native.hand_topic(side), {"state": {"angles": [i / 100] * 22}}, time_ns=time_ns, i=i)
                writer.write_message(
                    native.hand_topic(side, command=True),
                    joint,
                    {"header": header, "name": [], "position": [0.5 + i / 100] * 22, "velocity": [], "effort": []},
                    log_time=time_ns + 10_000_000,
                )
                for command in (False, True):
                    writer.write_message(
                        native.pose_topic(side, command=command),
                        pose,
                        {
                            "header": header,
                            "pose": {
                                "position": {"x": i / 100 + (0.1 if command else 0), "y": 0, "z": 0},
                                "orientation": dict(zip(("x", "y", "z", "w"), quaternion, strict=True)),
                            },
                        },
                        log_time=time_ns + 300_000_000,
                    )
            writer.write_message(
                native.HEAD,
                rgb,
                {"header": header, "format": "jpeg", "data": image_bytes.getvalue()},
                log_time=time_ns + 300_000_000,
            )
            # Two native tactile frames per model frame. Logical shape differs from flat TypedArray shape.
            for subframe in (0, 1):
                time_ns += subframe * 5_000_000
                for topic in native.TACTILE_TOPICS:
                    frame = {
                        "frame_id": i * 2 + subframe,
                        "ts": time_ns / 1e9,
                        "shape": {"RAW": [1, 2, 3], "DEFORM": [1, 2, 2]},
                        "content": {"RAW": {"__tensor__": 0}, "DEFORM": {"__tensor__": 1}, "F6": [i, 1, 2, 3, 4, 5]},
                    }
                    arrays = [
                        {
                            "path": f"/data/frame/content/{key}",
                            "dtype": "|u1",
                            "shape": [size],
                            "data": bytes(range(size)),
                        }
                        for key, size in (("RAW", 6), ("DEFORM", 4))
                    ]
                    write_native(topic, {"frame": frame}, arrays, time_ns=time_ns, i=i)


def write_manifest(directory, episodes):
    path = Path(directory) / "native.json"
    path.write_text(json.dumps({"schema_id": "tianji_sharpa_native_mcap_v1", "episodes": episodes}))
    return path


class McapReaderTest(unittest.TestCase):
    def test_quaternion_becomes_column_pair_rot6d(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.mcap"
            write_episode(path, quaternion=(0, 0, np.sqrt(0.5), np.sqrt(0.5)))
            reader = native.McapEpisode(path)
            np.testing.assert_allclose(reader.vector(BASE)[3:9], [0, 1, 0, -1, 0, 0], atol=1e-7)

    def test_source_alignment_preserves_delivery_latency_and_native_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.mcap"
            write_episode(path)
            reader = native.McapEpisode(path)
            image = reader.at(native.HEAD, BASE + 1_000_000)
            assert image.source_ns == BASE
            assert image.available_ns - image.source_ns == 300000000
            np.testing.assert_allclose(reader.joints(BASE)[:7], np.pi / 2)
            np.testing.assert_allclose(reader.joints(BASE, command=True)[:7], np.pi / 4)
            assert reader.head(BASE).shape == (6, 8, 3)
            history = list(reader.tactile_history(native.TACTILE_TOPICS[0], BASE, BASE + 10_000_000))
            assert len(history) == 2
            assert history[0]["RAW"].shape == (1, 2, 3)
            np.testing.assert_array_equal(history[0]["RAW"].reshape(-1), np.arange(6))
            restored = pickle.loads(pickle.dumps(reader))
            assert restored.deform(BASE).shape == (10, 2, 2)

    def test_future_and_stale_observations_are_not_selected(self):
        stream = native.Stream(
            [
                native.Record(100, 200, "past", "test", 200),
                native.Record(300, 310, "future", "test", 310),
            ]
        )
        assert stream.at(299).value == "past"
        with pytest.raises(ValueError, match="recent record"):
            stream.at(99)
        with pytest.raises(ValueError, match="recent record"):
            stream.at(299, max_age_ns=50)

    def test_failed_submission_is_not_a_training_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.mcap"
            write_episode(path, failed_frame=2)
            reader = native.McapEpisode(path)
            with pytest.raises(ValueError, match="submission failed"):
                reader.vector(BASE + round(2e9 / 30), command=True)

    def test_duplicate_episode_cannot_leak_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_manifest(
                directory, [{"path": "same.mcap", "task": "test", "split": split} for split in ("train", "validation")]
            )
            with pytest.raises(ValueError, match="more than once"):
                native.load_manifest(manifest, "train")


if __name__ == "__main__":
    unittest.main()
