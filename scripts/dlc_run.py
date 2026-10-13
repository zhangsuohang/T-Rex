"""Run a single-model Tianji/Sharpa job with OSS assets and local checkpoints."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

DATA_ROOT = Path("/mnt/data/zsh/data/TJ_sharpa/test/real_20261012")
RUN_ROOT = Path("/mnt/data/zsh/runs/TJ_sharpa/tianji_sharpa_v001")
WEIGHT_ROOT = Path("/mnt/data/zsh/code/model")


def copy_tree(source, target):
    """Copy closed files to an OSS mount using sequential writes."""
    source, target = Path(source), Path(target)
    files = []
    for src in sorted(source.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(source)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as reader, dst.open("wb") as writer:
            shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
        size = src.stat().st_size
        if dst.stat().st_size != size:
            raise OSError(f"Incomplete OSS copy: {dst}")
        files.append({"path": str(rel), "bytes": size})
    return files


def archive_checkpoint(checkpoint):
    """Called by trainers after a checkpoint has finished saving."""
    archive = os.environ.get("DLC_ARCHIVE_ROOT")
    if not archive:
        return
    local = Path(os.environ["DLC_LOCAL_RUN"])
    checkpoint = Path(checkpoint)
    relative = checkpoint.relative_to(local / "checkpoints")
    destination = Path(archive) / "checkpoints" / relative
    files = copy_tree(checkpoint, destination)
    if not files:
        raise ValueError(f"Empty checkpoint: {checkpoint}")
    (destination / "archive_complete.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    (Path(archive) / "latest_checkpoint.json").write_text(
        json.dumps({"path": str(Path("checkpoints") / relative)}, indent=2) + "\n"
    )
    print(f"Checkpoint archived: {destination}", flush=True)


def warmup_steps(steps):
    """Keep the decay phase valid even for one- and two-step smoke runs."""
    return min(100, steps - 1)


def train_command(args, local, project):
    assets = local / "assets"
    checkpoints = local / "checkpoints"
    if args.model == "openpi":
        return [sys.executable, str(project / "scripts/dlc_run.py"), "--openpi-train"]
    if args.model == "ftp1":
        return [
            sys.executable,
            "-u",
            "scripts/zarr_train_ftp1_pytorch.py",
            "ftp1_tianji_sharpa_mcap",
            f"--exp_name={args.run_id}",
            f"--dataset_config_path={DATA_ROOT / 'mcap_datasets.json'}",
            f"--assets_base_dir={assets}",
            f"--checkpoint_base_dir={checkpoints}",
            "--batch_size=1",
            "--num_workers=0",
            "--norm_num_workers=0",
            "--val_num_workers=0",
            "--use_val_dataset",
            "--val_ratio=0.1",
            "--no-wandb_enabled",
            f"--pytorch_weight_path={WEIGHT_ROOT / 'ftp1/ftp1_pretrain_v0426_50kstep'}",
            f"--num_train_steps={args.steps}",
            "--model.tactile_tokenizer_config.no-load_t3_pretrained_checkpoint",
            "--no-use_torch_compile",
            "--log_interval=10",
            "--save_interval=500",
            f"--val_interval={0 if args.smoke else 500}",
            f"--lr_schedule.warmup_steps={warmup_steps(args.steps)}",
            "--lr_schedule.peak_lr=5e-6",
        ]
    return [
        str(Path(sys.executable).parent / "accelerate"),
        "launch",
        "--config_file",
        "config/sft.yaml",
        "--num_processes",
        "1",
        "--main_process_port",
        "29612",
        "scripts/train_tianji_sharpa.py",
        "--checkpoint-dir",
        str(WEIGHT_ROOT / "t-rex/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6"),
        "--model-path",
        str(WEIGHT_ROOT / "qwen/Qwen3-VL-2B-Instruct"),
        "--data-path",
        str(DATA_ROOT / "native_mcap.json"),
        "--data-format",
        "mcap",
        "--mcap-stats-path",
        str(assets / "norm_stats.json"),
        "--output-dir",
        str(checkpoints),
        "--max_train_steps",
        str(args.steps),
        "--num_workers",
        "0",
        "--n_epochs",
        "1000000",
        "--train_bsz_per_gpu",
        "1",
        "--learning_rate",
        "5e-6",
        "--image_size",
        "224",
        "224",
        "--save_freq",
        "500",
        "--save_step_interval",
        "500",
        "--max_ckpts",
        "2",
        "--val_ratio",
        "0.1",
        "--val_freq",
        str(0 if args.smoke else 500),
        "--max_val_batches",
        "50",
    ]


def openpi_train():
    # Keep the launcher usable without model dependencies for job previews.
    import openpi.training.config as config_lib  # noqa: PLC0415
    from scripts.train import init_logging  # noqa: PLC0415
    from scripts.train import main as train  # noqa: PLC0415

    local = Path(os.environ["DLC_LOCAL_RUN"])
    base = config_lib.get_config("pi05_tianji_sharpa_mcap")
    steps = int(os.environ["DLC_TRAIN_STEPS"])
    config = dataclasses.replace(
        base,
        exp_name=os.environ["DLC_RUN_ID"],
        assets_base_dir=str(local / "assets"),
        checkpoint_base_dir=str(local / "checkpoints"),
        batch_size=1,
        num_workers=0,
        num_train_steps=steps,
        log_interval=10,
        save_interval=500,
        keep_period=None,
        wandb_enabled=False,
        ema_decay=None,
        lr_schedule=dataclasses.replace(
            base.lr_schedule, warmup_steps=warmup_steps(steps), peak_lr=5e-6, decay_steps=steps
        ),
    )
    init_logging()
    train(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("openpi", "ftp1", "t-rex"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-root", type=Path, default=Path("/mnt/workspace/zsh/dlc"))
    args = parser.parse_args()
    if args.smoke:
        args.steps = 2
    if args.steps <= 0:
        parser.error("--steps must be positive")
    project = Path(__file__).resolve().parents[1]
    local = args.local_root / args.model / args.run_id
    archive = RUN_ROOT / args.model / args.run_id
    command = train_command(args, local, project)
    if args.dry_run:
        print(json.dumps({"command": command, "local_run": str(local), "oss_run": str(archive)}, indent=2))
        return
    local.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "manifests"):
        (local / name).mkdir()
    shutil.copytree(RUN_ROOT / args.model / "v001/assets", local / "assets")
    shutil.copyfile(DATA_ROOT / "native_mcap.json", local / "manifests/native_mcap.json")
    if args.model in ("openpi", "ftp1"):
        cache = local / "cache/big_vision"
        cache.mkdir(parents=True)
        shutil.copyfile(
            RUN_ROOT / "preparation/runtime_assets/paligemma_tokenizer.model",
            cache / "paligemma_tokenizer.model",
        )
    if args.prepare_only:
        print(f"Assets prepared: {local}")
        return
    archive.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        {
            "DLC_LOCAL_RUN": str(local),
            "DLC_ARCHIVE_ROOT": str(archive),
            "DLC_RUN_ID": args.run_id,
            "DLC_TRAIN_STEPS": str(args.steps),
            "WANDB_MODE": "disabled",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONPATH": str(project) + os.pathsep + env.get("PYTHONPATH", ""),
            "TIANJI_MCAP_MANIFEST": str(DATA_ROOT / "native_mcap.json"),
            "OPENPI_PRETRAIN_PARAMS": str(WEIGHT_ROOT / "openpi/pi05_base/params"),
            "OPENPI_DATA_HOME": str(local / "cache"),
            "TORCH_EXTENSIONS_DIR": str(local / "torch_extensions"),
        }
    )
    if args.model == "ftp1":
        env["JAX_PLATFORMS"] = "cpu"
    metadata = {
        "model": args.model,
        "run_id": args.run_id,
        "steps": args.steps,
        "command": command,
        "job_id": env.get("DLC_JOB_ID"),
        "code_revision": env.get("DLC_CODE_REVISION"),
        "environment_bundle": env.get("DLC_ENVIRONMENT_BUNDLE"),
    }
    (local / "manifests/launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    status = {**metadata, "status": "running"}
    child = None

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        with (local / "logs/train.log").open("w", buffering=1) as log:
            child = subprocess.Popen(
                command,
                cwd=project,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            for line in child.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = child.wait()
        status.update(status="succeeded" if code == 0 else "failed", exit_code=code)
        if code:
            raise subprocess.CalledProcessError(code, command)
        if not (archive / "latest_checkpoint.json").is_file():
            status["status"] = "failed"
            raise RuntimeError("Training exited without an archived checkpoint")
    except BaseException:
        status["status"] = "failed"
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        for name in ("logs", "manifests", "assets"):
            copy_tree(local / name, archive / name)
        (archive / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--openpi-train"]:
        openpi_train()
    else:
        main()
