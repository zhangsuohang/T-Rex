"""Read portable Tianji/Sharpa MCAP streams without resampling the recording.

ROS headers carry source time; native host_unix_ns carries host callback time
(not a calibrated device acquisition time). MCAP log_time is retained separately.
Model datasets choose their own sampling times and windows.
"""

from collections import OrderedDict
from dataclasses import dataclass
import io
import json
from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from PIL import Image

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HEAD = "/camera/head/color/image_raw/compressed"
ARM_STATE = "/tianji_arm/native_state"
ARM_COMMAND = "/tianji_arm/native_command"
HAND_NAMES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_DIP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC_FE",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)


def pose_topic(side, *, command=False):
    return f"/tianji_arm/{side}/ee_pose_{'command' if command else 'state'}"


def hand_topic(side, *, command=False):
    return f"/wave/{side}/{'joint_commands' if command else 'native_state'}"


def tactile_topic(side, finger):
    return f"/wave/{side}/tactile/{finger}/native"


TACTILE_TOPICS = tuple(tactile_topic(side, finger) for side in SIDES for finger in FINGERS)
STATE_TOPICS = tuple(pose_topic(side) for side in SIDES) + tuple(hand_topic(side) for side in SIDES)
COMMAND_TOPICS = tuple(pose_topic(side, command=True) for side in SIDES) + tuple(
    hand_topic(side, command=True) for side in SIDES
)
TOPICS = (HEAD, ARM_STATE, ARM_COMMAND, *STATE_TOPICS, *COMMAND_TOPICS, *TACTILE_TOPICS)


@dataclass
class Record:
    source_ns: int
    available_ns: int
    value: object
    source_clock: str
    log_ns: int


@dataclass
class Stream:
    records: list[Record]

    def __post_init__(self):
        self.records.sort(key=lambda record: (record.source_ns, record.log_ns))
        self.times = np.array([record.source_ns for record in self.records], dtype=np.int64)

    def at(self, time_ns, *, max_age_ns=None):
        index = int(np.searchsorted(self.times, time_ns, side="right")) - 1
        if index < 0 or (max_age_ns is not None and time_ns - self.times[index] > max_age_ns):
            raise ValueError(f"No sufficiently recent record at {time_ns}.")
        return self.records[index]


def load_manifest(path, split=None):
    """Paths are relative to the manifest; an episode belongs to exactly one split."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["schema_id"] != "tianji_sharpa_native_mcap_v1":
        raise ValueError("Expected the native MCAP profile, not the LeRobot EE62 contract.")
    entries = []
    paths = set()
    for item in manifest["episodes"]:
        resolved = (path.parent / item["path"]).resolve()
        if resolved in paths:
            raise ValueError(f"An MCAP episode appears more than once: {resolved}")
        paths.add(resolved)
        if item["split"] not in ("train", "validation") or not item["task"].strip():
            raise ValueError("Each episode needs a train/validation split and a nonempty task.")
        if split is None or item["split"] == split:
            entries.append({**item, "path": str(resolved)})
    return entries


def _tensor(message, reference):
    tensor = message.arrays[reference["__tensor__"]]
    return np.frombuffer(bytes(tensor.data), dtype=np.dtype(tensor.dtype)).reshape(tuple(tensor.shape))


def _numeric(message, value):
    return _tensor(message, value) if isinstance(value, dict) and "__tensor__" in value else value


def _pose9(message):
    if message.header.frame_id != "world":
        raise ValueError(f"Expected the recorded model world, got {message.header.frame_id!r}.")
    p, q = message.pose.position, message.pose.orientation
    quaternion = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid pose quaternion.")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            p.x,
            p.y,
            p.z,
            1 - 2 * (y * y + z * z),
            2 * (x * y + z * w),
            2 * (x * z - y * w),
            2 * (x * y - z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z + x * w),
        ],
        dtype=np.float32,
    )


class McapEpisode:
    """Keep numeric indices/JPEGs in memory; decode RAW/DEFORM lazily by chunk.

    No file handles survive a call, so PyTorch spawn workers can pickle this
    reader. The bounded cache contains original native frames, not aligned rows.
    """

    def __init__(self, path):
        self.path = str(path)
        indexed = {topic: [] for topic in TOPICS}
        self._tactile_cache = OrderedDict()
        with open(self.path, "rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, channel, message, decoded in reader.iter_decoded_messages(topics=TOPICS, log_time_order=False):
                topic = channel.topic
                if hasattr(decoded, "metadata_json"):
                    metadata = json.loads(decoded.metadata_json)
                    source = int(metadata["host_unix_ns"])
                    clock = "host_callback_unix"
                    payload = metadata["data"]
                    if topic in TACTILE_TOPICS:
                        value = metadata  # Tensor references remain in metadata; bytes are read lazily.
                    elif topic == ARM_STATE:
                        value = np.stack(
                            [
                                np.deg2rad(_numeric(decoded, arm["fb_joint_pos"]))
                                for arm in payload["snapshot"]["outputs"]
                            ]
                        ).astype(np.float32)
                    elif topic == ARM_COMMAND:
                        value = {
                            "joints": np.stack([np.deg2rad(payload[f"{side}_position_deg"]) for side in SIDES]).astype(
                                np.float32
                            ),
                            "send_result": payload["send_result"],
                        }
                    else:
                        value = np.asarray(_numeric(decoded, payload["state"]["angles"]), dtype=np.float32)
                else:
                    stamp = decoded.header.stamp
                    source = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                    clock = "ros_header_unix"
                    if topic == HEAD:
                        value = bytes(decoded.data)
                    elif "/ee_pose_" in topic:
                        value = _pose9(decoded)
                    else:
                        if decoded.name and tuple(decoded.name) != HAND_NAMES:
                            raise ValueError(f"Unexpected hand command order on {topic}.")
                        value = np.asarray(decoded.position, dtype=np.float32)
                indexed[topic].append(Record(source, message.log_time, value, clock, message.log_time))
        self.streams = {topic: Stream(records) for topic, records in indexed.items() if records}

    def at(self, topic, time_ns, *, max_age_ns=None):
        return self.streams[topic].at(time_ns, max_age_ns=max_age_ns)

    def bounds(self, topics):
        return (
            max(int(self.streams[topic].times[0]) for topic in topics),
            min(int(self.streams[topic].times[-1]) for topic in topics),
        )

    def vector(self, time_ns, *, command=False, max_age_ns=50_000_000):
        """Absolute model-EE pose/SDK hand angles, left31 followed by right31."""
        values = []
        for side in SIDES:
            values.extend(
                [
                    self.at(pose_topic(side, command=command), time_ns, max_age_ns=max_age_ns).value,
                    self.at(hand_topic(side, command=command), time_ns, max_age_ns=max_age_ns).value,
                ]
            )
        vector = np.concatenate(values).astype(np.float32)
        if vector.shape != (62,) or not np.all(np.isfinite(vector)):
            raise ValueError("Expected a finite model-EE62 vector.")
        if command and self.at(ARM_COMMAND, time_ns, max_age_ns=max_age_ns).value["send_result"] != 1:
            raise ValueError("SDK submission failed at this target time.")
        return vector

    def joints(self, time_ns, *, command=False, max_age_ns=50_000_000):
        record = self.at(ARM_COMMAND if command else ARM_STATE, time_ns, max_age_ns=max_age_ns)
        arms = record.value["joints"] if command else record.value
        return np.concatenate(
            [
                part
                for index, side in enumerate(SIDES)
                for part in (
                    arms[index],
                    self.at(hand_topic(side, command=command), time_ns, max_age_ns=max_age_ns).value,
                )
            ]
        ).astype(np.float32)

    def force(self, time_ns, *, max_age_ns=50_000_000):
        force = np.stack(
            [
                self.at(topic, time_ns, max_age_ns=max_age_ns).value["data"]["frame"]["content"]["F6"]
                for topic in TACTILE_TOPICS
            ]
        ).astype(np.float32)
        if force.shape != (10, 6) or not np.all(np.isfinite(force)):
            raise ValueError("Expected finite ten-finger F6 values.")
        return force

    def head(self, time_ns, *, max_age_ns=100_000_000):
        record = self.at(HEAD, time_ns, max_age_ns=max_age_ns)
        return np.asarray(Image.open(io.BytesIO(record.value)).convert("RGB"))

    def tactile(self, record, topic):
        # 50ms receipt-time buckets keep all native samples and bound memory.
        bucket = record.log_ns // 50_000_000
        if bucket not in self._tactile_cache:
            decoded_frames = {}
            with open(self.path, "rb") as stream:
                reader = make_reader(stream, decoder_factories=[DecoderFactory()])
                for _, channel, message, decoded in reader.iter_decoded_messages(
                    topics=TACTILE_TOPICS,
                    start_time=bucket * 50_000_000,
                    end_time=(bucket + 1) * 50_000_000,
                ):
                    frame = json.loads(decoded.metadata_json)["data"]["frame"]
                    decoded_frames[(channel.topic, message.log_time)] = {
                        key: _tensor(decoded, frame["content"][key]).reshape(tuple(frame["shape"][key]))
                        for key in ("RAW", "DEFORM")
                    }
            self._tactile_cache[bucket] = decoded_frames
            if len(self._tactile_cache) > 4:
                self._tactile_cache.popitem(last=False)
        self._tactile_cache.move_to_end(bucket)
        return self._tactile_cache[bucket][(topic, record.log_ns)]

    def deform(self, time_ns, *, max_age_ns=50_000_000):
        return np.stack(
            [
                self.tactile(self.at(topic, time_ns, max_age_ns=max_age_ns), topic)["DEFORM"][0]
                for topic in TACTILE_TOPICS
            ]
        )

    def tactile_history(self, topic, start_ns, end_ns):
        """Original per-finger frames in [start,end], including RAW/DEFORM/F6 and both clocks."""
        records = self.streams[topic]
        first = int(np.searchsorted(records.times, start_ns, side="left"))
        last = int(np.searchsorted(records.times, end_ns, side="right"))
        for record in records.records[first:last]:
            yield {"record": record, **self.tactile(record, topic)}

    def __getstate__(self):
        return {**self.__dict__, "_tactile_cache": OrderedDict()}
