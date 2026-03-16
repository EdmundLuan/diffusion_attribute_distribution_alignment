"""
Non-Ray Batch Job Scheduler for Multi-GPU Execution.

This script manages batch execution of jobs across multiple GPUs without Ray.
Each job executes a Python script with specific arguments (e.g., hyperparameter search).

Features:
- Simple subprocess-based execution (no Ray dependency)
- Script caching at startup (prevents mid-run edit issues)
- Automatic job queueing (round-robin across GPUs)
- Same YAML config format as ray_exec.py
- Process-secure: cleans up subprocesses on interrupt (Ctrl+C)

Usage:
    python scripts/batch_exec.py --devices 0,1 --dryrun        # Preview jobs
    python scripts/batch_exec.py --devices 0,1                 # Run jobs
    python scripts/batch_exec.py --config my_config.yaml       # Custom config
    python scripts/batch_exec.py --devices 0,1-3,6             # Range support
"""

import argparse
import atexit
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from guided_diffusion.launcher_util import (
    ensure_dir, load_yaml_config, write_yaml_config,
    build_run_dirs_log, append_run_dir_log, backup_code,
    now_tag, build_outdir_name,
)


# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Base directory for relative paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Directories to exclude from code backup
BACKUP_EXCLUDES = [
    "checkpoints", "datasets", "eval", "training-runs",
    "docs", "figs", "__pycache__", ".git", "tests", "evaluations", 
    "outputs", "models", "pbsjobs", "data",
]


# -------------------------------
# Code Snapshot (Immutability)
# -------------------------------
def create_code_snapshot(base_dir: Path, outroot: Path, excludes: List[str]) -> Path:
    """
    Create a frozen snapshot of the entire codebase for job execution.

    This ensures code immutability: all jobs run from this snapshot,
    so edits to the main codebase won't affect running/queued jobs.
    Similar to Ray's runtime_env working_dir feature.

    Args:
        base_dir: Source directory containing the codebase
        outroot: Output root directory to store the snapshot
        excludes: Directory names to exclude from backup

    Returns:
        Path to the snapshot directory
    """
    snapshot_path = backup_code(outroot, base_dir, excludes=excludes)
    logger.info(f"Created code snapshot: {snapshot_path}")
    logger.info("All jobs will run from this frozen snapshot for consistency.")
    return Path(snapshot_path)


# -------------------------------
# Job Configuration
# -------------------------------
def get_job_configs(config_path: Path) -> Dict[str, Any]:
    """
    Load job configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configuration dictionary containing:
        - script: Path to the Python script to execute
        - fixed_args: Arguments that stay constant across all jobs
        - sweep_args: Lists of values to sweep over (Cartesian product)
        - outroot: Output root directory
        - tag: Tag for this sweep

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required keys are missing
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml_config(config_path)

    # Validate required keys
    required_keys = ["script", "fixed_args", "sweep_args", "outroot", "tag"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return config


def parse_devices(devices_str: str) -> List[int]:
    """
    Parse device string with range support.

    Args:
        devices_str: Comma-separated device IDs with optional ranges
            Examples: "0,1,2" or "0,1-3,6" or "0-3"

    Returns:
        Sorted list of unique device IDs
    """
    devices = []
    for part in devices_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            # Range: "1-3" -> [1, 2, 3]
            start, end = part.split("-", 1)
            devices.extend(range(int(start), int(end) + 1))
        else:
            devices.append(int(part))

    # Remove duplicates and sort
    devices = sorted(set(devices))

    if not devices:
        raise ValueError(f"No valid devices parsed from: {devices_str}")
    if any(d < 0 for d in devices):
        raise ValueError(f"Device IDs must be non-negative: {devices}")

    return devices


def generate_jobs(config: Dict[str, Any], outroot: Path, tag: str) -> List[Dict[str, Any]]:
    """
    Generate list of job configurations from sweep parameters.

    Creates Cartesian product of all sweep_args values.

    Args:
        config: Job configuration dictionary
        outroot: Root directory for outputs
        tag: Tag for this sweep

    Returns:
        List of job dictionaries, each containing script path and arguments
    """
    script = config["script"]
    fixed_args = config["fixed_args"]
    sweep_args = config["sweep_args"]

    # Get sweep parameter names and values
    sweep_keys = list(sweep_args.keys())
    sweep_values = list(sweep_args.values())

    jobs = []
    for combo in itertools.product(*sweep_values):
        # Build argument dictionary for this job
        job_args = fixed_args.copy()
        for key, value in zip(sweep_keys, combo):
            job_args[key] = value

        # Output directory will be generated at job execution time
        jobs.append({
            "script": script,
            "args": job_args,
            "tag": tag,
            "sweep_keys": sweep_keys,
            "outroot": str(outroot),
        })

    return jobs


def build_command(script: str, args: Dict[str, Any]) -> List[str]:
    """
    Build command line arguments from script path and argument dictionary.

    Handles different argument types:
    - Boolean: passed as --flag True/False
    - List: expanded as space-separated values
    - Dict: serialized as JSON string for CLI passthrough
    - Other: passed as --key value

    Args:
        script: Path to the Python script
        args: Dictionary of arguments

    Returns:
        List of command line arguments
    """
    cmd = ["python", script]
    for key, value in args.items():
        if isinstance(value, bool):
            cmd.append(f"--{key}")
            cmd.append("True" if value else "False")
        elif isinstance(value, list):
            if value:  # Only add if non-empty
                cmd.append(f"--{key}")
                cmd.extend(str(v) for v in value)
        elif isinstance(value, dict):
            # Serialize dicts as JSON strings for CLI passthrough
            cmd.append(f"--{key}")
            cmd.append(json.dumps(value))
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    return cmd


# -------------------------------
# Process Management
# -------------------------------
# Global list to track active subprocesses for cleanup
_active_processes: List[Dict[str, Any]] = []


def cleanup_processes():
    """
    Terminate all active subprocesses gracefully.
    Called on program exit or signal interrupt.
    """
    global _active_processes

    if not _active_processes:
        return

    logger.info(f"Cleaning up {len(_active_processes)} active process(es)...")

    for slot in _active_processes:
        proc = slot.get("process")
        if proc is not None and proc.poll() is None:
            try:
                # Try graceful termination first
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if SIGTERM didn't work
                proc.kill()
            except Exception as e:
                logger.warning(f"Error cleaning up process: {e}")

        # Close log file if open
        log_f = slot.get("log_file")
        if log_f and not log_f.closed:
            log_f.close()

    _active_processes.clear()
    logger.info("Process cleanup complete")


def setup_signal_handlers():
    """
    Register signal handlers for graceful shutdown.
    Ensures child processes are cleaned up on SIGINT/SIGTERM.
    """
    def signal_handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
        cleanup_processes()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_processes)


# -------------------------------
# Job Launcher (Round-Robin Scheduler)
# -------------------------------
def launch_jobs(
    jobs: List[Dict[str, Any]],
    devices: List[int],
    base_dir: Path,
    outroot: Path,
    snapshot_dir: Path,
    dry_run: bool = False,
) -> None:
    """
    Launch all jobs with round-robin GPU scheduling.

    Maintains at most len(devices) concurrent jobs.
    When a job completes, its GPU is assigned to the next queued job.
    All jobs run from the frozen code snapshot for immutability.

    Args:
        jobs: List of job configuration dictionaries
        devices: List of GPU device IDs to use
        base_dir: Base directory (for output path calculation)
        outroot: Root output directory
        snapshot_dir: Path to the frozen code snapshot
        dry_run: If True, only print commands without executing
    """
    global _active_processes

    # existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    # if existing:
    #     # PBS already constrained GPUs; interpret --devices as indices into that list
    #     visible = [x.strip() for x in existing.split(",") if x.strip()]
    #     devices = [visible[d] for d in devices]  # devices should be 0..len(visible)-1
    #     logger.info(f"Existing CUDA_VISIBLE_DEVICES={existing}; mapped devices to {devices}")

    num_gpus = len(devices)
    total_jobs = len(jobs)

    logger.info(f"Prepared {total_jobs} jobs for execution on {num_gpus} GPUs")
    logger.info(f"Code snapshot: {snapshot_dir}")

    # Dry run: just print what would be executed
    if dry_run:
        logger.info("[DRY RUN] Printing job commands without execution:")
        for i, job in enumerate(jobs):
            snapshot_script = snapshot_dir / job["script"]
            cmd = build_command(str(snapshot_script), job["args"])
            logger.info(f"  [{i+1}/{total_jobs}] {' '.join(cmd[:5])}...")
            logger.info(f"      Output: {job['tag']}-<params>-ts_<runtime>")
        return

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    # Create run log files
    success_log, failed_log = build_run_dirs_log(outroot)

    # Track timing
    start_time = dt.datetime.now()

    # Job queue and counters
    queue = list(jobs)
    completed = 0
    success_count = 0
    failed_count = 0

    # Start initial wave of jobs (one per GPU)
    for i in range(min(num_gpus, len(queue))):
        gpu = devices[i]
        job = queue.pop(0)
        slot = _start_job(job, gpu, total_jobs, len(queue), snapshot_dir, base_dir, outroot)
        _active_processes.append(slot)

    # Main loop: poll for completed jobs and start new ones
    try:
        while _active_processes or queue:
            # Check for completed jobs
            newly_freed = []

            for slot in list(_active_processes):
                ret = slot["process"].poll()
                if ret is not None:
                    # Job completed
                    job_duration = dt.datetime.now() - slot["start_time"]
                    duration_str = str(job_duration).split('.')[0]

                    # Write end marker to log
                    slot["log_file"].write("\n" + "-" * 60 + "\n")
                    slot["log_file"].write(f"[JOB END] Return code: {ret}\n")
                    slot["log_file"].close()

                    # Update counters
                    completed += 1

                    if ret == 0:
                        success_count += 1
                        append_run_dir_log(success_log, slot["outdir"], base_dir)
                        logger.info(
                            f"[DONE] [{completed}/{total_jobs}] [GPU {slot['gpu']}] "
                            f":: {slot['outdir_rel']} (success) [{duration_str}]"
                        )
                    else:
                        failed_count += 1
                        append_run_dir_log(failed_log, slot["outdir"], base_dir)
                        logger.warning(
                            f"[FAIL] [{completed}/{total_jobs}] [GPU {slot['gpu']}] "
                            f":: {slot['outdir_rel']} (code={ret}) [{duration_str}]"
                        )

                    # Track freed GPU
                    newly_freed.append(slot["gpu"])
                    _active_processes.remove(slot)

                    # Log elapsed time
                    elapsed = str(dt.datetime.now() - start_time).split('.')[0]
                    logger.info(f"[TIMER] Elapsed: {elapsed}")

            # Start new jobs on freed GPUs
            for gpu in newly_freed:
                if queue:
                    job = queue.pop(0)
                    time.sleep(1.1)  # Small delay to stagger job starts
                    slot = _start_job(job, gpu, total_jobs, len(queue), snapshot_dir, base_dir, outroot)
                    _active_processes.append(slot)

            # Small sleep to avoid busy-waiting
            if not newly_freed:
                time.sleep(1.0)

        # Final summary
        elapsed = str(dt.datetime.now() - start_time).split('.')[0]
        logger.info("-" * 80)
        logger.info(f"[COMPLETE] {success_count} succeeded, {failed_count} failed")
        logger.info(f"[TIMER] Total wall time: {elapsed}")

    except KeyboardInterrupt:
        logger.warning("Interrupted! Cleaning up...")
        raise


def _start_job(
    job: Dict[str, Any],
    gpu: int,
    total_jobs: int,
    queue_remaining: int,
    snapshot_dir: Path,
    base_dir: Path,
    outroot: Path,
) -> Dict[str, Any]:
    """
    Start a single job on the specified GPU using the frozen code snapshot.

    Args:
        job: Job configuration dictionary
        gpu: GPU device ID to use
        total_jobs: Total number of jobs (for logging)
        queue_remaining: Jobs remaining in queue after this one (for logging)
        snapshot_dir: Path to the frozen code snapshot
        base_dir: Base directory for relative paths (used only for output path calculation)
        outroot: Root output directory

    Returns:
        Slot dictionary with process info for tracking
    """
    # Generate timestamp and output directory
    timestamp = now_tag()
    args = job["args"].copy()
    outdir_name = build_outdir_name(job["tag"], args, timestamp, sweep_keys=job["sweep_keys"])
    outdir = outroot / outdir_name
    outdir_rel = outdir.relative_to(base_dir)

    # Update args with sample_dir
    args["sample_dir"] = str(outdir_rel)

    # Create output directory and write config
    ensure_dir(outdir)
    write_yaml_config(outdir / "config.yaml", args)

    # Build command using the script from the snapshot directory
    # The script path in config is relative to base_dir, so it works in snapshot too
    snapshot_script = snapshot_dir / job["script"]
    cmd = build_command(str(snapshot_script), args)

    # Setup environment:
    # 1. CUDA_VISIBLE_DEVICES for GPU isolation
    # 2. PYTHONPATH points to snapshot so all imports use frozen code
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Prepend snapshot_dir to PYTHONPATH for code immutability
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{snapshot_dir}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(snapshot_dir)

    # Open log file
    log_path = outdir / "run.log"
    log_f = open(log_path, "w", buffering=1)
    log_f.write(f"[JOB START] GPU {gpu}\n")
    log_f.write(f"Snapshot: {snapshot_dir}\n")
    log_f.write(f"PYTHONPATH: {env['PYTHONPATH']}\n")
    log_f.write(f"Command: {' '.join(cmd)}\n")
    log_f.write(f"Output dir: {outdir_rel}\n")
    log_f.write("-" * 60 + "\n")
    log_f.flush()

    # Start subprocess with cwd=base_dir so relative paths to models/datasets work
    # PYTHONPATH points to snapshot, so all imports use frozen code
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(base_dir),
        env=env,
    )

    job_num = total_jobs - queue_remaining
    logger.info(f"[LAUNCH] [{job_num}/{total_jobs}] [GPU {gpu}] :: {outdir_rel}")

    return {
        "process": proc,
        "gpu": gpu,
        "outdir": outdir,
        "outdir_rel": outdir_rel,
        "log_file": log_f,
        "start_time": dt.datetime.now(),
    }


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Non-Ray batch job scheduler for multi-GPU execution"
    )
    parser.add_argument(
        "--config", type=str, default="configs/ray_launch_emsa.yaml",
        help="Path to YAML config file (default: configs/ray_launch_emsa.yaml)"
    )
    parser.add_argument(
        "--devices", type=str, default="0,1",
        help="GPU IDs with range support (e.g., '0,1-3,6' -> [0,1,2,3,6])"
    )
    parser.add_argument(
        "--outroot", type=str, default=None,
        help="Override output root directory"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Override sweep tag"
    )
    parser.add_argument(
        "--dryrun", dest="dry_run", action="store_true",
        help="Print commands without executing"
    )
    # Note: --backup flag removed; code snapshot is always created for immutability

    args = parser.parse_args()

    # Parse device IDs with range support
    args.devices = parse_devices(args.devices)

    return args


def main():
    """Main entry point."""
    try:
        args = parse_args()
        num_gpus = len(args.devices)

        # Load job configuration from YAML file
        config_path = BASE_DIR / args.config
        config = get_job_configs(config_path)
        logger.info(f"Loaded config: {args.config}")

        # Override from CLI if provided
        if args.outroot:
            config["outroot"] = args.outroot
        if args.tag:
            config["tag"] = args.tag

        outroot = BASE_DIR / config["outroot"]
        tag = config["tag"]

        # Print configuration
        logger.info("=" * 80)
        logger.info("Batch Job Scheduler (Non-Ray)")
        logger.info("=" * 80)
        logger.info(f"Config:      {args.config}")
        logger.info(f"Script:      {config['script']}")
        logger.info(f"Output root: {outroot}")
        logger.info(f"Tag:         {tag}")
        logger.info(f"Devices:     {args.devices} ({num_gpus} GPUs)")
        logger.info(f"Dry run:     {args.dry_run}")
        logger.info("=" * 80)

        # Ensure output directory exists
        ensure_dir(outroot)

        # Create code snapshot for immutability (always, even for dry run to show what would happen)
        # This ensures all jobs run from the same frozen code, even if the original is modified
        logger.info("Creating code snapshot for immutability...")
        snapshot_dir = create_code_snapshot(BASE_DIR, outroot, BACKUP_EXCLUDES)

        # Verify the script exists in the snapshot
        snapshot_script = snapshot_dir / config["script"]
        if not snapshot_script.exists():
            raise FileNotFoundError(f"Script not found in snapshot: {snapshot_script}")

        # Generate jobs from configuration
        jobs = generate_jobs(config, outroot, tag)

        # Print sweep summary
        logger.info(f"Generated {len(jobs)} jobs from parameter sweep:")
        for key, values in config["sweep_args"].items():
            logger.info(f"  {key}: {values}")

        # Launch jobs (all will run from the frozen snapshot)
        launch_jobs(
            jobs=jobs,
            devices=args.devices,
            base_dir=BASE_DIR,
            outroot=outroot,
            snapshot_dir=snapshot_dir,
            dry_run=args.dry_run,
        )
        return 0

    except KeyboardInterrupt:
        logger.warning("Program interrupted by user")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
