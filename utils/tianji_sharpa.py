"""Versioned Tianji/Sharpa LeRobot v2.1 mother-data contract.

This module is copied into each policy repository so conversion does not
require an editable install of another model or its dependency environment.
"""

import argparse
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


def build_features(rgb_shape=(3, 360, 640), tactile_shape=(3, 240, 1200)):
    features = {
        "observation.state": {"dtype": "float32", "shape": (62,), "names": CHANNEL_NAMES},
        "action": {"dtype": "float32", "shape": (62,), "names": CHANNEL_NAMES},
        "observation.joints": {"dtype": "float32", "shape": (58,), "names": JOINT_NAMES},
        "observation.tactile_force": {"dtype": "float32", "shape": (60,), "names": F6_NAMES},
    }
    for key in VIDEO_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": tactile_shape if "tactile" in key else rgb_shape,
            "names": ["channels", "height", "width"],
        }
    return features


def build_contract():
    return {
        "schema_id": SCHEMA_ID,
        "status": "draft",
        "codebase_version": "v2.1",
        "fps": 30,
        "state_semantics": "measured_absolute_tcp_pose_hand_joints",
        "action_semantics": "commanded_absolute_tcp_pose_hand_targets",
        "rotation_6d_convention": "column_pair",
        "channel_names": CHANNEL_NAMES,
        "position_unit": "metre",
        "joint_unit": "radian",
        "coordinate_frame": None,
        "arm_dof_per_side": 7,
        "hand_dof_per_side": 22,
        "joint_layout": "left_arm7,left_hand22,right_arm7,right_hand22",
        "hand_finger_sizes": [5, 4, 4, 4, 5],
        "tactile_finger_order": list(FINGERS),
        "tactile_force_layout": "left30,right30; per finger Fx,Fy,Fz,Mx,My,Mz",
        "tactile_deform_encoding": "derived_uint8_grayscale_rgb_packed_thumb_first",
        "tactile_raw_policy": "Retain source bit depth, range and source timestamps outside the derived videos.",
        "optional_features": {"action.joints": {"dtype": "float32", "shape": [58], "names": JOINT_NAMES}},
        "timing": "observation at t; action is the command assigned to t; future observations are never targets",
        "synchronization": "Resample observations causally. Preserve source timestamps and validity in capture records.",
        "tianji_model": None,
        "sharpa_model": None,
        "urdf": None,
        "left_tcp_calibration": None,
        "right_tcp_calibration": None,
        "left_hand_sdk_to_q": None,
        "right_hand_sdk_to_q": None,
        "camera_calibrations": None,
        "tactile_deform_scaling": None,
        "freeze_before_capture": [
            "hardware models",
            "SDK joint maps",
            "base/TCP/camera calibrations",
            "sensor timestamps and command alignment",
            "RGB shapes",
            "deform scaling/F6 units",
        ],
    }


def load_metadata(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    contract = json.loads((root / "meta/capture_contract.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("Mother data must be LeRobot v2.1 (per-episode Parquet/video).")
    if contract.get("schema_id") != SCHEMA_ID or contract.get("rotation_6d_convention") != "column_pair":
        raise ValueError("Expected Tianji Sharpa EE62 v1 column-pair contract.")
    if contract.get("action_semantics") != "commanded_absolute_tcp_pose_hand_targets":
        raise ValueError("Action must be recorded absolute command targets.")
    if contract.get("position_unit") != "metre" or contract.get("joint_unit") != "radian":
        raise ValueError("Expected metres and radians.")
    if contract.get("tactile_finger_order") != list(FINGERS):
        raise ValueError("Tactile must be thumb/index/middle/ring/pinky, left then right.")
    if contract.get("state_semantics") != "measured_absolute_tcp_pose_hand_joints":
        raise ValueError("State must be measured absolute TCP/hand state.")
    for key, names in (
        ("observation.state", CHANNEL_NAMES),
        ("action", CHANNEL_NAMES),
        ("observation.joints", JOINT_NAMES),
        ("observation.tactile_force", F6_NAMES),
    ):
        feature = info["features"][key]
        if feature["shape"] != [len(names)] or feature.get("names") != names:
            raise ValueError(f"Wrong shape or channel order for {key}.")
    return info, contract


def episode_paths(root, info, episode_index):
    root = Path(root)
    args = {"episode_index": episode_index, "episode_chunk": episode_index // info.get("chunks_size", 1000)}
    data = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return root / data.format(**args), {key: root / video.format(**args, video_key=key) for key in VIDEO_KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("features.json", build_features()), ("capture_contract.json", build_contract())):
        destination = args.output_dir / name
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
