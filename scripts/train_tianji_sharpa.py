"""Post-train Tianji/Sharpa JSON exports using the midtrain architecture.

This entrypoint inherits the checkpoint's model flags and does not configure
devices or a distributed launcher. Use the existing Accelerate configuration.
"""

import argparse
import json
import os
from pathlib import Path
import sys


def training_arguments(checkpoint, model_path, data_path, output_dir, validation_data_path=None):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "training_args.json").read_text())
    if config.get("action_dim") != 62 or config.get("action_chunk") != 16:
        raise ValueError("Expected the 62D, horizon-16 bimanual midtrain checkpoint.")
    if not (checkpoint / "model.pt").is_file():
        raise FileNotFoundError(checkpoint / "model.pt")
    args = [
        "--model_path",
        str(model_path),
        "--data_format",
        "json",
        "--data_path",
        str(data_path),
        "--resume_checkpoint",
        str(checkpoint),
        "--resume_source",
        "midtrain",
        "--strict_resume",
        "--output_dir",
        str(output_dir),
        "--log_dir",
        str(Path(output_dir) / "logs"),
        "--experiment_name",
        "tianji_sharpa_bimanual",
        "--val_ratio",
        "0",
    ]
    keys = (
        "action_dim",
        "action_chunk",
        "use_robot_state",
        "use_tactile_deform",
        "use_tactile_vec",
        "tactile_intermediate_size",
        "training_stage",
        "use_flare",
        "n_flare_tokens_per_frame",
        "n_flare_steps",
        "flare_layer_index",
        "use_tactile_code",
        "vqvae_codebook_size",
        "use_tactile_vqvae",
        "flare_frame_stride",
        "cascaded_total_steps",
        "cascaded_split_step",
    )
    for key in keys:
        if key in config:
            args.extend((f"--{key}", str(config[key])))
    if validation_data_path:
        args.extend(("--validation_data_path", str(validation_data_path), "--val_freq", "500"))
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-data-path", type=Path)
    args, overrides = parser.parse_known_args()
    architecture = {
        "--action_dim",
        "--action_chunk",
        "--use_robot_state",
        "--use_tactile_deform",
        "--use_tactile_vec",
        "--use_tactile_code",
        "--use_tactile_vqvae",
        "--vqvae_codebook_size",
        "--tactile_intermediate_size",
        "--use_flare",
        "--n_flare_tokens_per_frame",
        "--n_flare_steps",
        "--flare_layer_index",
    }
    if any(token.split("=", 1)[0] in architecture for token in overrides):
        parser.error(
            "Architecture flags are inherited from training_args.json; use train.py for architecture experiments."
        )
    arguments = training_arguments(
        args.checkpoint_dir, args.model_path, args.data_path, args.output_dir, args.validation_data_path
    )
    script = Path(__file__).with_name("train.py")
    os.execv(sys.executable, [sys.executable, str(script), *arguments, *overrides])


if __name__ == "__main__":
    main()
