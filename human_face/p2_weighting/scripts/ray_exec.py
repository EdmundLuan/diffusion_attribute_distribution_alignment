"""
Ray-based Job Scheduler for Multi-GPU Execution.

This script uses Ray to manage batch running of jobs across multiple GPUs.
Each job executes a Python script with specific arguments (e.g., hyperparameter search).

Features:
- Ray handles GPU resource allocation (1 GPU per job)
- Automatic job queueing when more jobs than GPUs
- Output directories tagged with timestamps
- Code snapshot at startup via Ray's runtime_env (prevents mid-run edit issues)
- Flexible YAML configuration for different scripts
- Process-secure: cleans up subprocesses on interrupt (Ctrl+C)

Usage:
    python scripts/ray_exec.py --devices 0,1 --dryrun        # Preview jobs
    python scripts/ray_exec.py --devices 0,1                 # Run jobs
    python scripts/ray_exec.py --config my_config.yaml       # Custom config
    python scripts/ray_exec.py --devices 0,1-3,6             # Range support

Configuration files are in YAML format (see configs/ray_launch_emsa.yaml).
"""

import argparse
import atexit
import itertools
import json
import signal
import sys
import os
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, Any, List

import ray

from guided_diffusion.launcher_util import (
    ensure_dir, load_yaml_config, write_yaml_config,
    build_run_dirs_log, append_run_dir_log, backup_code,
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

# Directories to exclude from code backup and Ray's working_dir
BACKUP_EXCLUDES = [
    "checkpoints", "datasets", "eval", "training-runs",
    "docs", "figs", "__pycache__", ".git", "tests", "evaluations", 
    "outputs", "models", "pbsjobs", "data"
]


# -------------------------------
# Process-Secure Cleanup
# -------------------------------
def cleanup_ray():
    """
    Graceful cleanup of Ray cluster.
    Called on program exit or signal interrupt.
    """
    try:
        if ray.is_initialized():
            logger.info("Shutting down Ray cluster...")
            ray.shutdown()
            logger.info("Ray cluster shutdown complete")
    except Exception as e:
        logger.error(f"Error during Ray cleanup: {e}")


def setup_signal_handlers():
    """
    Register signal handlers for graceful shutdown.
    Ensures Ray and child processes are cleaned up on SIGINT/SIGTERM.
    """
    def signal_handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
        cleanup_ray()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_ray)


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

    Examples:
        "0,1,2" -> [0, 1, 2]
        "0,1-3,6" -> [0, 1, 2, 3, 6]
        "0-3" -> [0, 1, 2, 3]
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

        # NOTE: Timestamp and output directory will be generated at job execution time
        # to ensure unique timestamps (jobs may start at different times)
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
            # Boolean args: pass True as flag, skip False
            cmd.append(f"--{key}")
            cmd.append("True" if value else "False")
        elif isinstance(value, list):
            # List args: expand as space-separated values (for nargs='*')
            # e.g., classifier_heads: ["gender", "age_group"] -> --classifier_heads gender age_group
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
# Ray Remote Job Executor
# -------------------------------
@ray.remote(num_gpus=1)
def run_job(job: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """
    Execute a single job on a GPU. This runs as a Ray remote task.

    Ray automatically sets CUDA_VISIBLE_DEVICES for the allocated GPU.
    Uses process groups for reliable cleanup on interruption.

    Args:
        job: Job configuration dictionary with 'script', 'args', 'tag', 'sweep_keys', 'outroot'
        base_dir: Base directory path for relative paths

    Returns:
        Dictionary with job result: success, return_code, outdir
    """
    import os
    import signal
    import subprocess
    import datetime as dt
    import logging
    from pathlib import Path
    from guided_diffusion.launcher_util import now_tag, build_outdir_name, write_yaml_config

    # Use shared logger
    logger = logging.getLogger(__name__)

    base_dir = Path(base_dir)
    script = job["script"]
    args = job["args"]
    tag = job["tag"]
    sweep_keys = job["sweep_keys"]
    outroot = Path(job["outroot"])

    # Generate timestamp NOW when job actually starts (ensures unique timestamps)
    timestamp = now_tag()
    outdir_name = build_outdir_name(tag, args, timestamp, sweep_keys=sweep_keys)
    outdir = outroot / outdir_name
    outdir_rel = outdir.relative_to(base_dir)

    # Update args with the actual sample_dir
    args["sample_dir"] = str(outdir_rel)

    # Get assigned GPU info (for logging)
    gpu_ids = ray.get_runtime_context().get_assigned_resources().get("GPU", [])
    # Convert GPU IDs to integers for cleaner display
    if isinstance(gpu_ids, list):
        gpu_ids_int = [int(gpu_id) for gpu_id in gpu_ids]
        gpu_info = f"GPU {gpu_ids_int[0] if len(gpu_ids_int) == 1 else gpu_ids_int}"
    elif isinstance(gpu_ids, (int, float)):
        gpu_info = f"GPU {int(gpu_ids)}"
    else:
        gpu_info = "No GPU"

    # Create output directory and write config when job actually starts
    # (moved from launch_jobs to ensure unique directories per job)
    outdir.mkdir(parents=True, exist_ok=True)
    write_yaml_config(outdir / "config.yaml", args)

    # Log launch message with actual directory name
    logger.info(f"[LAUNCH] {gpu_info} :: {outdir_rel}")

    # Build command
    cmd = build_command(script, args)

    # Log file for this job
    log_path = outdir / "run.log"

    # Track job wallclock time
    job_start_time = dt.datetime.now()

    process = None
    try:
        with open(log_path, "w") as log_f:
            log_f.write(f"[JOB START] {gpu_info}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Working dir: .\n")
            log_f.write(f"Output dir: {outdir_rel}\n")
            log_f.write("-" * 60 + "\n")
            log_f.flush()

            # Run the job in a new session (process group) for reliable cleanup
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(base_dir),
                start_new_session=True,  # Create new process group
            )
            return_code = process.wait()

            log_f.write("\n" + "-" * 60 + "\n")
            log_f.write(f"[JOB END] Return code: {return_code}\n")

        # Calculate job duration
        job_duration = dt.datetime.now() - job_start_time
        duration_str = str(job_duration).split('.')[0]  # Format: HH:MM:SS

        return {
            "success": return_code == 0,
            "return_code": return_code,
            "outdir": str(outdir_rel),
            "gpu": gpu_info,
            "duration": duration_str,
        }

    except Exception as e:
        # Terminate subprocess and its children if still running
        if process is not None and process.poll() is None:
            try:
                # Kill entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except Exception:
                # Force kill if SIGTERM didn't work
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:
                    pass

        # Calculate duration even on failure
        job_duration = dt.datetime.now() - job_start_time
        duration_str = str(job_duration).split('.')[0]

        return {
            "success": False,
            "return_code": -1,
            "outdir": str(outdir_rel),
            "error": str(e),
            "gpu": gpu_info,
            "duration": duration_str,
        }


# -------------------------------
# Main Launcher
# -------------------------------
def launch_jobs(
    jobs: List[Dict[str, Any]],
    devices: List[int],
    base_dir: Path,
    outroot: Path,
    dry_run: bool = False,
) -> None:
    """
    Launch all jobs using Ray, with automatic queueing.
    Process-secure: properly cleans up on interruption.

    Args:
        jobs: List of job configuration dictionaries
        devices: List of GPU device IDs to use
        base_dir: Base directory for code
        outroot: Root output directory
        dry_run: If True, only print commands without executing
    """
    num_gpus = len(devices)
    total_jobs = len(jobs)
    logger.info(f"Prepared {total_jobs} jobs for execution on {num_gpus} GPUs")

    if dry_run:
        cuda_devices = ",".join(str(d) for d in devices)
        logger.info(f"[DRY RUN] Would set CUDA_VISIBLE_DEVICES={cuda_devices}")
        logger.info("[DRY RUN] Printing job commands without execution:")
        for i, job in enumerate(jobs):
            cmd = build_command(job["script"], job["args"])
            logger.info(f"  [{i+1}/{total_jobs}] {' '.join(cmd[:5])}...")
            logger.info(f"      Output: {job['tag']}-<params>-ts_<runtime>")
        return

    # Register signal handlers before initializing Ray
    setup_signal_handlers()

    # Restrict Ray to specified GPUs via CUDA_VISIBLE_DEVICES
    # This must be set BEFORE ray.init() so Ray only sees these GPUs
    # cuda_devices = ",".join(str(d) for d in devices)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # logger.info(f"Setting CUDA_VISIBLE_DEVICES={cuda_devices}")

    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing:
        # PBS already constrained GPUs; interpret --devices as indices into that list
        visible = [x.strip() for x in existing.split(",") if x.strip()]
        chosen = [visible[i] for i in devices]  # devices should be 0..len(visible)-1
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(chosen)
        logger.info(f"PBS provided CUDA_VISIBLE_DEVICES={existing}; using subset={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # Create run log files
    success_log, failed_log = build_run_dirs_log(outroot)

    import sys
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    # Initialize Ray with GPU resources
    # runtime_env snapshots the working directory to ensure code consistency
    ray.init(
        num_gpus=num_gpus,
        # runtime_env={
        #     "working_dir": str(base_dir),
        #     "excludes": BACKUP_EXCLUDES,
        # },
        ignore_reinit_error=True,
        # Suppress metrics/telemetry warnings (jobs run fine without them)
        _metrics_export_port=None,
        include_dashboard=False,
        ## key HPC hardening
        # _node_ip_address=os.environ.get("RAY_NODE_IP_ADDRESS", "127.0.0.1"),
        # _temp_dir=os.environ.get("RAY_TMPDIR"),
    )

    start_time = dt.datetime.now()
    logger.info(f"Ray initialized with {num_gpus} GPUs")

    # Submit all jobs to Ray (it handles queueing automatically)
    # Note: Directory and config will be created in run_job() when job actually starts
    futures = []
    for i, job in enumerate(jobs):
        # Submit to Ray
        future = run_job.remote(job, str(base_dir))
        futures.append((i, job, future))

    # Collect results as they complete
    completed = 0
    success_count = 0
    failed_count = 0

    try:
        while futures:
            # Wait for any job to complete
            done_futures = [f for _, _, f in futures]
            ready, _ = ray.wait(done_futures, num_returns=1, timeout=1.0)

            for ready_future in ready:
                # Find the job that completed
                for idx, (i, job, future) in enumerate(futures):
                    if future == ready_future:
                        result = ray.get(future)
                        completed += 1

                        # Use actual directory from result (created at job execution time)
                        outdir = base_dir / result["outdir"]
                        job_duration = result.get("duration", "unknown")

                        if result["success"]:
                            success_count += 1
                            append_run_dir_log(success_log, outdir, base_dir)
                            logger.info(
                                f"[DONE] [{completed}/{total_jobs}] {result['gpu']} "
                                f":: {outdir.name} (success) [{job_duration}]"
                            )
                        else:
                            failed_count += 1
                            append_run_dir_log(failed_log, outdir, base_dir)
                            logger.warning(
                                f"[FAIL] [{completed}/{total_jobs}] {result['gpu']} "
                                f":: {outdir.name} (code={result['return_code']}) [{job_duration}]"
                            )

                        futures.pop(idx)

                        elapsed = str(dt.datetime.now() - start_time).split('.')[0]
                        logger.info(f"[TIMER] Elapsed: {elapsed}")

                        # Add delay after each job completion (1100ms)
                        import time
                        time.sleep(1.1)

                        break

        # Final summary
        elapsed = str(dt.datetime.now() - start_time).split('.')[0]
        logger.info("-" * 80)
        logger.info(f"[COMPLETE] {success_count} succeeded, {failed_count} failed")
        logger.info(f"[TIMER] Total wall time: {elapsed}")

    except KeyboardInterrupt:
        logger.warning("Interrupted! Cancelling remaining jobs...")
        # Cancel all pending futures
        for _, _, future in futures:
            try:
                ray.cancel(future, force=True)
            except Exception:
                pass
        raise

    finally:
        # Always clean up Ray
        cleanup_ray()


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ray-based job scheduler for multi-GPU execution"
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
        "--backup", action="store_true",
        help="Create code backup before running"
    )
    parser.add_argument(
        "--dryrun", dest="dry_run", action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    # Parse device IDs with range support
    args.devices = parse_devices(args.devices)

    return args


def main():
    """Main entry point with proper exception handling."""
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
        logger.info("Ray Job Scheduler (Process-Secure)")
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

        # Optional code backup
        if args.backup and not args.dry_run:
            logger.info("Creating code backup...")
            backup_path = backup_code(outroot, BASE_DIR, excludes=BACKUP_EXCLUDES)
            logger.info(f"Code backed up to: {backup_path}")

        # Generate jobs from configuration
        jobs = generate_jobs(config, outroot, tag)

        # Print sweep summary
        logger.info(f"Generated {len(jobs)} jobs from parameter sweep:")
        for key, values in config["sweep_args"].items():
            logger.info(f"  {key}: {values}")

        # Launch jobs
        launch_jobs(
            jobs=jobs,
            devices=args.devices,
            base_dir=BASE_DIR,
            outroot=outroot,
            dry_run=args.dry_run,
        )
        return 0

    except KeyboardInterrupt:
        logger.warning("Program interrupted by user")
        return 130  # Standard exit code for SIGINT (128 + 2)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
