"""Versioned Tianji/Sharpa LeRobot v2.1 mother-data contract.

This module is copied into each policy repository so conversion does not
require an editable install of another model or its dependency environment.
"""

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np

SCHEMA_ID = "tianji_sharpa_ee62_v1"
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ROT_INDICES = (*range(3, 9), *range(34, 40))
CHANNEL_NAMES = [
    f"{side}_{component}"
    for side in SIDES
    for component in (
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "r00",
        "r10",
        "r20",
        "r01",
        "r11",
        "r21",
        *(f"hand_q{i}" for i in range(22)),
    )
]
JOINT_NAMES = [
    f"{side}_{group}_q{i}" for side in SIDES for group, size in (("arm", 7), ("hand", 22)) for i in range(size)
]
F6_NAMES = [
    f"{side}_{finger}_{channel}"
    for side in SIDES
    for finger in FINGERS
    for channel in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
]
VIDEO_KEYS = (
    "observation.images.head",
    "observation.images.left_hand",
    "observation.images.right_hand",
    "observation.images.left_tactile",
    "observation.images.right_tactile",
)


def rot6d_to_matrix(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1:] != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Rot6D must be finite with final dimension 6.")
    first, second = values[..., :3], values[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Rot6D first column has zero norm.")
    first = first / norm
    second = second - (first * second).sum(axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Rot6D columns are collinear.")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def validate_vectors(values, name="state"):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1:] != (62,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite with final dimension 62, got {values.shape}.")
    for offset in (0, 31):
        rotation = values[..., offset + 3 : offset + 9]
        matrices = rot6d_to_matrix(rotation)
        projected = np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1)
        if not np.allclose(rotation, projected, atol=1e-3):
            raise ValueError(f"{name} has non-orthonormal column-pair Rot6D at offset {offset}.")
    return values


def local_chunk(state, targets):
    state = validate_vectors(state)
    targets = validate_vectors(targets, "targets")
    if state.shape != (62,) or targets.ndim != 2:
        raise ValueError("Expected (62,) anchor and (H,62) targets.")
    output = targets.copy()
    for offset in (0, 31):
        rotation = rot6d_to_matrix(state[offset + 3 : offset + 9])
        output[:, offset : offset + 3] = (targets[:, offset : offset + 3] - state[offset : offset + 3]) @ rotation
        relative = rotation.T @ rot6d_to_matrix(targets[:, offset + 3 : offset + 9])
        output[:, offset + 3 : offset + 9] = np.concatenate((relative[:, :, 0], relative[:, :, 1]), axis=-1)
    return output


def absolute_chunk(state, actions):
    """Restore denormalized local model outputs to absolute TCP/hand targets."""
    state = validate_vectors(state)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1:] != (62,) or not np.all(np.isfinite(actions)):
        raise ValueError("Local actions must be finite with final dimension 62.")
    if state.shape != (62,) or actions.ndim != 2:
        raise ValueError("Expected (62,) anchor and (H,62) local actions.")
    output = actions.copy()
    for offset in (0, 31):
        rotation = rot6d_to_matrix(state[offset + 3 : offset + 9])
        output[:, offset : offset + 3] = state[offset : offset + 3] + actions[:, offset : offset + 3] @ rotation.T
        absolute = rotation @ rot6d_to_matrix(actions[:, offset + 3 : offset + 9])
        output[:, offset + 3 : offset + 9] = np.concatenate((absolute[:, :, 0], absolute[:, :, 1]), axis=-1)
    return output


PROTOCOL_VERSION = "1.0.0"
HAND_JOINT_ORDER = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
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
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
STREAM_NAMES = (
    "left_state",
    "right_state",
    "left_joints",
    "right_joints",
    "head_rgb",
    "left_wrist_rgb",
    "right_wrist_rgb",
    "left_deform",
    "right_deform",
    "left_f6",
    "right_f6",
)
COMMAND_NAMES = ("left_tcp", "left_hand", "right_tcp", "right_hand")
MAX_AGES = np.array([0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
NUMERIC_FEATURES = {
    "observation.state": ("float32", CHANNEL_NAMES),
    "action": ("float32", CHANNEL_NAMES),
    "observation.joints": ("float32", JOINT_NAMES),
    "observation.tactile_force": ("float32", F6_NAMES),
    "observation.source_timestamp": ("float64", list(STREAM_NAMES)),
    "observation.available_timestamp": ("float64", list(STREAM_NAMES)),
    "observation.valid": ("bool", list(STREAM_NAMES)),
    "action.command_timestamp": ("float64", list(COMMAND_NAMES)),
    "action.valid": ("bool", list(COMMAND_NAMES)),
}


def _hand_reference():
    # Both editable policy packages and the T-Rex standalone utils ship the
    # same reference file. It defines angle semantics, not the installed robot.
    here = Path(__file__).resolve()
    root = here.parents[3] if here.parent.name == "shared" else here.parents[1]
    return json.loads((root / "configs/tianji_sharpa/hand_reference.json").read_text())


def build_features(rgb_shape=(3, 360, 640), tactile_shape=(3, 240, 1200)):
    features = {
        key: {"dtype": dtype, "shape": (len(names),), "names": names}
        for key, (dtype, names) in NUMERIC_FEATURES.items()
    }
    for key in VIDEO_KEYS:
        shape = tuple(tactile_shape if "tactile" in key else rgb_shape)
        if len(shape) != 3 or shape[0] != 3 or min(shape) <= 0 or shape[1] % 2 or shape[2] % 2:
            raise ValueError("Video declarations must be positive CHW RGB shapes.")
        if "tactile" in key and shape[2] != 5 * shape[1]:
            raise ValueError("Deform video must be five horizontal square tiles.")
        features[key] = {"dtype": "video", "shape": shape, "names": ["channels", "height", "width"]}
    return features


def build_contract(rgb_shape=(3, 360, 640), tactile_shape=(3, 240, 1200)):
    features = build_features(rgb_shape, tactile_shape)
    return {
        "schema_id": SCHEMA_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "released",
        "codebase_version": "v2.1",
        "fps": 30,
        "state_semantics": "measured_absolute_tcp_pose_hand_joints",
        "action_semantics": "commanded_absolute_tcp_pose_hand_targets",
        "rotation_6d_convention": "column_pair",
        "channel_names": CHANNEL_NAMES,
        "position_unit": "metre",
        "joint_unit": "radian",
        "coordinate_frame": "episode_robot_base",
        "world_frame_definition": {
            "origin": "Midpoint of the left and right arm mounting-base origins at episode start.",
            "z": "Opposite gravity.",
            "x": "Robot forward direction projected perpendicular to z and normalized.",
            "y": "z cross x; right-handed, robot left.",
            "lifetime": "Fixed for the entire episode, including when the robot base moves. Left/right are robot anatomical sides, not viewer sides.",
        },
        "tcp_definition": {
            "origin": "Center of the rigid hand wrist mounting plane; never a moving fingertip.",
            "z": "From wrist center toward middle-finger MCP, projected into the palm plane and normalized.",
            "x": "Outward dorsal (back-of-hand) normal, perpendicular to z.",
            "y": "z cross x; right-handed for BOTH hands.",
            "transform": "p_base = R_base_tcp @ p_tcp + translation_base_tcp. Rot6D=[R[:,0],R[:,1]]; third column=cross(first,second). State and targets use the same rigid TCP.",
            "producer": "Convert the SDK/tool frame to this TCP before writing; no SDK, FK or IK is called by the training consumer.",
        },
        "vector_slices": {
            "left_xyz": [0, 3],
            "left_rot6d": [3, 9],
            "left_hand": [9, 31],
            "right_xyz": [31, 34],
            "right_rot6d": [34, 40],
            "right_hand": [40, 62],
        },
        "arm_dof_per_side": 7,
        "hand_dof_per_side": 22,
        "joint_layout": "left_arm7,left_hand22,right_arm7,right_hand22",
        "arm_joint_convention": "Serial chain order from arm base toward wrist; absolute angle in radians about each joint's positive local axis, referenced to the device's mechanical zero. Do not use an episode-relative zero or modulo-wrap continuous angles. Zero/axis definitions must remain fixed within a dataset; device metadata is optional provenance.",
        "hand_joint_order": list(HAND_JOINT_ORDER),
        "hand_reference": _hand_reference(),
        "hand_finger_sizes": [5, 4, 4, 4, 5],
        "joint_consistency": "The hand blocks of observation.joints equal observation.state; when action.joints exists its hand blocks equal action. Tolerance 1e-4 rad. Arm targets are actual accepted commands, never measured future joints.",
        "tactile_finger_order": list(FINGERS),
        "tactile_force_layout": "left30,right30; per finger Fx,Fy,Fz,Mx,My,Mz",
        "tactile_force_convention": {
            "force_unit": "newton",
            "torque_unit": "newton_metre",
            "origin": "Center of the distal finger tactile pad contact surface.",
            "z": "Outward pad surface normal.",
            "y": "Distal phalanx direction toward fingertip, projected perpendicular to z.",
            "x": "y cross z; right-handed. Frame moves with the distal phalanx.",
            "sign": "External environment wrench acting ON the finger; compression along outward normal has negative Fz.",
            "producer": "Rotate force and torque into this frame and translate torque to this origin. Preserve signed values; no per-frame normalization or clipping in the numeric mother field.",
        },
        "tactile_deform_encoding": "derived_uint8_grayscale_rgb_packed_thumb_first",
        "deform_definition": {
            "tiles": "Five square tiles horizontally, thumb/index/middle/ring/pinky, channel-identical RGB.",
            "value": "Scalar deformation intensity d in [0,1], no-contact baseline=0, fixed calibration full scale=1; quantize round(255*clip(d,0,1)).",
            "scaling": "Fixed per-sensor calibration for the whole dataset; no per-frame min/max or auto-exposure rescaling. SDK/camera raw RGB is not a deformation map.",
            "tile_orientation": "Top=toward fingertip (+pad y); right=+pad x. Producer removes sensor-specific flips/rotations before packing.",
            "raw": "Raw high-bit-depth/deformation arrays may be retained separately; they do not replace these derived tiles.",
        },
        "video_shapes": {key: list(features[key]["shape"]) for key in VIDEO_KEYS},
        "video_encoding": {
            "container": "mp4",
            "codec": "h264",
            "pixel_format": "yuv420p",
            "fps": 30,
            "frame_order": "One decoded frame per Parquet row, in row order, no dropped/reordered frames.",
            "rgb": "Unmirrored, upright RGB on the writer boundary; decode OpenCV BGR to RGB. CHW metadata, HWC uint8 capture arrays. Keep a camera's FOV, mounting and preprocessing fixed within a dataset.",
            "views": "head=fixed overview camera; left_hand/right_hand=corresponding wrist camera. No swapping sides or substituting another view.",
            "resolution": "Default RGB 360x640, deform 240x1200. Other positive EVEN CHW shapes are allowed if declared here and in info.json; deform width=5*height. No undeclared per-episode shape changes.",
        },
        "clock": {
            "unit": "second",
            "domain": "One monotonic acquisition clock per recording; hardware clocks must be converted before export.",
            "origin": "Episode first control-cycle boundary=0; warmup source timestamps may be negative.",
            "grid": "timestamp[n]=n/30; frame_index starts at 0, contiguous. No normalization, action chunking or relative TCP conversion in mother data.",
            "observation_stream_order": list(STREAM_NAMES),
            "observation_source_rule": "source_timestamp <= available_timestamp <= timestamp[n]. Select the latest causally available sample; no future-nearest or future interpolation.",
            "max_observation_age_seconds": MAX_AGES.tolist(),
            "command_stream_order": list(COMMAND_NAMES),
            "action_rule": "action[n] is the final accepted absolute command assigned to control cycle n, after controller clipping/filtering. command_timestamp is its real acceptance/send timestamp, no later than the next boundary (n+1)/30. A held target retains its original timestamp. Log targets after any hardware-specific interpolation into the protocol boundary; never synthesize targets from observation[n+1].",
            "freshness": "Repeated slower sensor samples are allowed only within the age limits. A repeated command is allowed only when it is the recorded controller-held target.",
            "invalid": "All required validity flags must be true for exported training rows. Stop/split the episode on missing/stale/invalid data or command failures; resume with a new episode index and timestamp=0. Never silently remove interior rows, fill images/forces with zeros, or bridge gaps.",
        },
        "required_feature_keys": list(features),
        "optional_features": {"action.joints": {"dtype": "float32", "shape": [58], "names": JOINT_NAMES}},
        "episode_metadata": {
            "file": "meta/capture_episodes.jsonl",
            "required": {
                "episode_index": "integer",
                "session_id": "nonempty string",
                "task_index": "integer",
                "outcome": "success|failure|unknown",
            },
            "task": "One nonempty instruction per episode, referenced by task_index in standard meta/tasks.jsonl; <=512 characters for the FTP exporter.",
            "split": "Record session_id for grouping; determine train/validation split before model normalization. Never compute validation-only statistics.",
        },
        "optional_provenance": {
            "hardware_models": None,
            "sdk_versions": None,
            "urdf": None,
            "joint_maps": None,
            "calibration_references": None,
            "raw_archive": None,
            "rule": "Optional provenance only. Empty or omitted values never block training acceptance. The producer is responsible for conforming numeric meanings and calibration conversions.",
        },
        "acceptance_boundary": "Automated validation checks structure, finite values, rotations, joint consistency, timing, validity, tasks, video decoding/shape/fps/codec and episode isolation. It cannot prove that a producer used the correct physical calibration, force units or camera mounting; the producer must attest semantic conformance.",
        "change_policy": "This is the first released v1.0.0 of the previous unpublished EE62 draft. Semantic or ordering changes require a new protocol_version and matching consumer update, never silent reuse of 1.0.0.",
    }


def load_metadata(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    contract = json.loads((root / "meta/capture_contract.json").read_text())
    if info.get("codebase_version") != "v2.1" or info.get("fps") != 30:
        raise ValueError("Mother data must be LeRobot v2.1 at 30 fps.")
    if contract.get("protocol_version") != PROTOCOL_VERSION or contract.get("status") != "released":
        raise ValueError("Expected released Tianji Sharpa protocol 1.0.0, not the earlier draft.")
    reference = build_contract()
    # Shapes are declared dataset parameters and optional provenance has no
    # training semantics. Every normative convention must match the release.
    for key, expected in reference.items():
        if key in ("video_shapes", "optional_provenance"):
            continue
        if contract.get(key) != expected:
            raise ValueError(f"Wrong protocol convention: {key}.")
    for key, (dtype, names) in NUMERIC_FEATURES.items():
        feature = info["features"].get(key, {})
        if feature.get("dtype") != dtype or feature.get("shape") != [len(names)] or feature.get("names") != names:
            raise ValueError(f"Wrong dtype/shape/channel order for {key}.")
    for key in VIDEO_KEYS:
        feature = info["features"].get(key, {})
        shape = feature.get("shape", [])
        if (
            feature.get("dtype") != "video"
            or feature.get("names") != ["channels", "height", "width"]
            or len(shape) != 3
            or shape[0] != 3
            or min(shape) <= 0
            or shape[1] % 2
            or shape[2] % 2
        ):
            raise ValueError(f"Wrong CHW video declaration for {key}.")
        if "tactile" in key and shape[2] != shape[1] * 5:
            raise ValueError(f"Wrong five-tile deform shape for {key}.")
        if contract.get("video_shapes", {}).get(key) != shape:
            raise ValueError(f"Contract and info.json video shapes disagree for {key}.")
    if (
        "action.joints" in info["features"]
        and info["features"]["action.joints"] != reference["optional_features"]["action.joints"]
    ):
        raise ValueError("Wrong optional action.joints declaration.")
    return info, contract


def episode_paths(root, info, episode_index):
    root = Path(root)
    args = {"episode_index": episode_index, "episode_chunk": episode_index // info.get("chunks_size", 1000)}
    data = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return root / data.format(**args), {key: root / video.format(**args, video_key=key) for key in VIDEO_KEYS}


def validate_numeric_episode(rows, *, episode_index=None, task_index=None):
    arrays = {}
    for key, (dtype, names) in NUMERIC_FEATURES.items():
        values = np.asarray(list(rows[key]))
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError(f"Wrong frame vector shape for {key}.")
        if dtype == "bool" and values.dtype != np.bool_:
            raise ValueError(f"{key} must contain actual booleans.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite values in {key}.")
        arrays[key] = values
    state = validate_vectors(arrays["observation.state"])
    actions = validate_vectors(arrays["action"], "action")
    n = len(state)
    if n == 0 or any(len(value) != n for value in arrays.values()):
        raise ValueError("An episode must have nonempty, equal-length streams.")
    timestamp = np.asarray(rows["timestamp"], dtype=np.float64)
    if (
        timestamp.shape != (n,)
        or not np.all(np.isfinite(timestamp))
        or not np.allclose(timestamp, np.arange(n) / 30, atol=1e-4, rtol=1e-7)
    ):
        raise ValueError("timestamp must be the contiguous 30 Hz grid starting at zero.")
    if not np.array_equal(np.asarray(rows["frame_index"]), np.arange(n)):
        raise ValueError("frame_index must be contiguous from zero; split episodes at gaps.")
    episodes = np.asarray(rows["episode_index"])
    tasks = np.asarray(rows["task_index"])
    if episodes.shape != (n,) or np.any(episodes != (episodes[0] if episode_index is None else episode_index)):
        raise ValueError("Cross-episode or incorrect episode_index values.")
    if tasks.shape != (n,) or np.any(tasks != (tasks[0] if task_index is None else task_index)):
        raise ValueError("Task instruction changes inside an episode.")
    source, available = arrays["observation.source_timestamp"], arrays["observation.available_timestamp"]
    if np.any(source > available + 1e-4) or np.any(available > timestamp[:, None] + 1e-4):
        raise ValueError("Future observation: require source <= available <= row timestamp.")
    if np.any(timestamp[:, None] - source > MAX_AGES + 1e-4):
        raise ValueError("Stale observation exceeds protocol age budget.")
    if not arrays["observation.valid"].all() or not arrays["action.valid"].all():
        raise ValueError("Invalid observation/command rows must be split out before training export.")
    if np.any(arrays["action.command_timestamp"] > timestamp[:, None] + 1 / 30 + 1e-4):
        raise ValueError("Command belongs to a future control cycle.")
    for key in ("observation.source_timestamp", "observation.available_timestamp", "action.command_timestamp"):
        if np.any(np.diff(arrays[key], axis=0) < -1e-4):
            raise ValueError(f"Clock moves backward in {key}.")
    command_times = arrays["action.command_timestamp"]
    for column, start, stop in ((0, 0, 9), (1, 9, 31), (2, 31, 40), (3, 40, 62)):
        held = np.abs(np.diff(command_times[:, column])) < 1e-9
        changed = np.any(np.abs(np.diff(actions[:, start:stop], axis=0)) > 1e-6, axis=1)
        if np.any(held & changed):
            raise ValueError("A held command timestamp cannot label a changed target.")
    joints = arrays["observation.joints"]
    for joint_offset, state_offset in ((7, 9), (36, 40)):
        if not np.allclose(
            joints[:, joint_offset : joint_offset + 22], state[:, state_offset : state_offset + 22], atol=1e-4, rtol=0
        ):
            raise ValueError("Measured hand joint blocks disagree between state and observation.joints.")
    if "action.joints" in rows:
        targets = np.asarray(list(rows["action.joints"]), dtype=np.float32)
        if targets.shape != (n, 58) or not np.isfinite(targets).all():
            raise ValueError("Optional action.joints must be finite (N,58) commands.")
        for joint_offset, action_offset in ((7, 9), (36, 40)):
            if not np.allclose(
                targets[:, joint_offset : joint_offset + 22],
                actions[:, action_offset : action_offset + 22],
                atol=1e-4,
                rtol=0,
            ):
                raise ValueError("Commanded hand blocks disagree between action and action.joints.")
    return n


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_dataset(root):
    """Streaming validation of exported mother episodes; never alters data."""
    # Only dataset validation needs these; generation and transforms use NumPy alone.
    cv2 = import_module("cv2")
    pa = import_module("pyarrow")
    pq = import_module("pyarrow.parquet")

    root = Path(root)
    info, _ = load_metadata(root)
    count = info.get("total_episodes", 0)
    if count <= 0:
        raise ValueError("Dataset contains no episodes.")
    captures = _jsonl(root / "meta/capture_episodes.jsonl")
    if len(captures) != count or {x["episode_index"] for x in captures} != set(range(count)):
        raise ValueError("capture_episodes must describe every episode exactly once.")
    captures = {x["episode_index"]: x for x in captures}
    task_rows = _jsonl(root / "meta/tasks.jsonl")
    if len({x["task_index"] for x in task_rows}) != len(task_rows):
        raise ValueError("Duplicate task_index in tasks.jsonl.")
    tasks = {x["task_index"]: x["task"] for x in task_rows}
    total_frames = 0
    for index in range(count):
        capture_info = captures[index]
        if not isinstance(capture_info.get("session_id"), str) or not capture_info["session_id"].strip():
            raise ValueError(f"Episode {index} has no session_id.")
        if capture_info.get("outcome") not in ("success", "failure", "unknown"):
            raise ValueError(f"Episode {index} has invalid outcome.")
        task = tasks.get(capture_info.get("task_index"))
        if not isinstance(task, str) or not task.strip() or len(task) > 512:
            raise ValueError(f"Episode {index} needs a nonempty task instruction <=512 characters.")
        parquet, videos = episode_paths(root, info, index)
        table = pq.read_table(parquet)
        for key, (dtype, _) in NUMERIC_FEATURES.items():
            column_type = table.schema.field(key).type
            if not (
                pa.types.is_list(column_type) or pa.types.is_fixed_size_list(column_type)
            ) or column_type.value_type != pa.from_numpy_dtype(np.dtype(dtype)):
                raise ValueError(f"Parquet dtype disagrees with features for {key}.")
        n = validate_numeric_episode(table.to_pydict(), episode_index=index, task_index=capture_info["task_index"])
        total_frames += n
        for key, path in videos.items():
            video = cv2.VideoCapture(str(path))
            if not video.isOpened():
                raise ValueError(f"Cannot decode video {path}.")
            try:
                if abs(video.get(cv2.CAP_PROP_FPS) - 30) > 0.01:
                    raise ValueError(f"Video fps differs from 30: {path}.")
                codec = int(video.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii")
                if codec not in ("avc1", "H264", "h264"):
                    raise ValueError(f"Video must use H.264, got {codec}: {path}.")
                shape = tuple(info["features"][key]["shape"])
                for _ in range(n):
                    ok, frame = video.read()
                    if not ok or frame.shape != (shape[1], shape[2], 3):
                        raise ValueError(f"Video frame count/shape differs from Parquet/declaration: {path}.")
                    if "tactile" in key and np.max(frame.max(axis=-1).astype(int) - frame.min(axis=-1)) > 3:
                        raise ValueError(f"Deform must be grayscale RGB, not raw color: {path}.")
                if video.read()[0]:
                    raise ValueError(f"Video has extra frames: {path}.")
            finally:
                video.release()
    if info.get("total_frames") != total_frames:
        raise ValueError("info.json total_frames differs from actual Parquet rows.")
    return {"schema_id": SCHEMA_ID, "protocol_version": PROTOCOL_VERSION, "episodes": count, "frames": total_frames}


def example_frame():
    state = np.zeros(62, dtype=np.float32)
    state[:3], state[31:34] = [0.4, 0.25, 0.6], [0.4, -0.25, 0.6]
    state[3:9] = state[34:40] = [1, 0, 0, 0, 1, 0]
    target = state.copy()
    target[[0, 31]] += 0.001
    return {
        "observation.state": state.tolist(),
        "action": target.tolist(),
        "observation.joints": [0.0] * 58,
        "observation.tactile_force": [0.0] * 60,
        "observation.source_timestamp": [-0.02] * 11,
        "observation.available_timestamp": [-0.01] * 11,
        "observation.valid": [True] * 11,
        "action.command_timestamp": [0.005] * 4,
        "action.valid": [True] * 4,
        "timestamp": 0.0,
        "frame_index": 0,
        "episode_index": 0,
        "task_index": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--output-dir", type=Path, help="Generate released writer features, contract and examples.")
    choice.add_argument("--validate-dataset", type=Path, help="Validate all mother Parquet rows and video frames.")
    args = parser.parse_args()
    if args.validate_dataset:
        print(json.dumps(validate_dataset(args.validate_dataset), ensure_ascii=False))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = (
        ("features.json", build_features()),
        ("capture_contract.json", build_contract()),
        (
            "example_frame.json",
            {
                "example_only": True,
                "frame": example_frame(),
                "videos": "Supply five actual HWC uint8 images for capture; this numeric example is not a real demonstration.",
            },
        ),
        (
            "example_capture_episode.json",
            {"episode_index": 0, "session_id": "session_example", "task_index": 0, "outcome": "unknown"},
        ),
    )
    if any((args.output_dir / name).exists() for name, _ in files):
        raise FileExistsError("Protocol outputs already exist; choose a new output directory.")
    for name, data in files:
        (args.output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
