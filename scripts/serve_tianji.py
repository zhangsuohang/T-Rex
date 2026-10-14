"""Native MCAP head-only T-Rex service with absolute model-world EE62 outputs.

Use the shared tianji-client package from ftp1-sharpa-policy/deployment/tianji-client.
No wrist RGB or external VQ-VAE checkpoint is used for this trained profile.
"""

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from test import CascadedServer, model_load
from tianji_client.protocol import absolute_chunk
from tianji_client.server import serve_policy


class TianjiPolicy:
    def __init__(self, checkpoint_dir, cuda="0"):
        saved = json.loads((Path(checkpoint_dir) / "training_args.json").read_text())
        if saved["action_dim"] != 62 or saved["action_chunk"] != 16:
            raise ValueError("This deployment profile requires native Tianji EE62/H16")
        args = SimpleNamespace(**saved)
        args.checkpoint_path = checkpoint_dir
        args.base_model_path = ""
        args.stats_path = ""
        args.dataset_name = ""
        args.cuda = cuda
        args.strict_load = True
        args.disable_tactile = 0
        args.vqvae_ckpt = ""  # Embedded tokenizer is reconstructed from training_args.
        args.image_size = (224, 224)  # train_tianji_sharpa.py's native profile
        model, processor, stats = model_load(args)
        self.server = CascadedServer(args, model, processor, stats)
        self.metadata = {
            "embodiment": "tianji_sharpa_bimanual",
            "state_dim": 62,
            "action_dim": 62,
            "action_horizon": 16,
            "control_hz": 30,
            "head_only": True,
            "action_semantics": "absolute_model_ee_pose_absolute_sdk_hand_joints",
            "pose_frame": "pinocchio_model_world",
            "end_effectors": ["EE_0", "EE_1"],
            "data_schema_id": "tianji_sharpa_native_mcap_v1",
            "rotation_6d_convention": "column_pair",
            "skill_embedded_in_actions": False,
            "f6_history_frames": 16,
            "f6_history_hz": 30,
        }

    def infer(self, obs):
        from tianji_client.protocol import Observation

        history = np.asarray(obs["observation/tactile_f6_history"], dtype=np.float32)
        deform = np.concatenate(
            (
                obs["observation/left_tactile_deform"],
                obs["observation/right_tactile_deform"],
            )
        )
        snapshot = Observation(
            int(obs["observation/time_ns"]),
            obs["observation/state"],
            obs["observation/image"],
            deform=deform,
            f6_history=history,
        )
        validated = snapshot.request("t-rex", obs["prompt"])
        encoded = io.BytesIO()
        Image.fromarray(validated["observation/image"]).resize(
            (224, 224), Image.Resampling.LANCZOS
        ).save(encoded, format="PNG")
        result = self.server.predict(
            "slow_and_fast",
            {
                "image_head": encoded.getvalue(),
                "task_description": validated["prompt"],
                "tactile_f6": history,
                "tactile_deform": deform.astype(np.float32) / 255.0,
                "state_fast": validated["observation/state"],
            },
        )
        # Anchor stays fixed at request construction, even if the robot moves while inference runs.
        return {
            "actions": absolute_chunk(
                validated["observation/state"], result["actions"]
            ),
            "observation/time_ns": snapshot.time_ns,
            "server_timing": {"infer_ms": result["latency_ms"]},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--cuda", default="0")
    args = parser.parse_args()
    policy = TianjiPolicy(args.checkpoint_dir, args.cuda)
    # Warm-up uses the actual native profile: one RGB, ten fingers and 16 F6 frames.
    from tianji_client.protocol import Observation

    state = np.zeros(62, dtype=np.float32)
    state[3:9] = state[34:40] = [1, 0, 0, 0, 1, 0]
    dummy = Observation(
        0,
        state,
        np.zeros((224, 224, 3), dtype=np.uint8),
        deform=np.zeros((10, 240, 240), dtype=np.uint8),
        f6_history=np.zeros((16, 10, 6), dtype=np.float32),
    )
    output = policy.infer(dummy.request("t-rex", "Press the button."))
    if output["actions"].shape != (16, 62) or not np.isfinite(output["actions"]).all():
        raise ValueError("Native-profile warm-up failed")
    print("Native Tianji warm-up passed: (16,62) absolute actions", flush=True)
    serve_policy(policy, args.host, args.port)


if __name__ == "__main__":
    main()
