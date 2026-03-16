"""
Utility functions for the job launcher.

Provides:
- Timestamp generation for run tagging
- Directory and file management
- YAML configuration reading/writing
- Code backup utilities
"""

import datetime as dt
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml


# -------------------------------
# YAML Loading
# -------------------------------
def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# -------------------------------
# Minimal YAML dumper (no deps)
# -------------------------------
def _yaml_dump(obj: Any, indent: int = 0) -> str:
    """Convert Python object to YAML string without external dependencies."""
    sp = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(_yaml_dump(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        lines = []
        for v in obj:
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_yaml_dump(v, indent + 1))
            else:
                lines.append(f"{sp}- {_yaml_scalar(v)}")
        return "\n".join(lines)
    else:
        return f"{sp}{_yaml_scalar(obj)}"


def _yaml_scalar(v: Any) -> str:
    """Convert a scalar value to YAML string representation."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Quote if contains special chars/spaces/colons
    special_chars = [":", "#", "{", "}", "[", "]", ",", "&", "*", "!", "|", ">", "'", '"', "%", "@", "`"]
    if any(c in s for c in special_chars) or s.strip() != s or " " in s:
        return '"' + s.replace('"', '\\"') + '"'
    return s


# -------------------------------
# Timestamp and Directory Utils
# -------------------------------
def now_tag() -> str:
    """Generate a timestamp string for run disambiguation (local time)."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    """Create directory and all parents if they don't exist."""
    p.mkdir(parents=True, exist_ok=True)


def write_yaml_config(path: Path, cfg: Dict[str, Any]) -> None:
    """Write configuration dictionary to a YAML file."""
    ensure_dir(path.parent)
    path.write_text(_yaml_dump(cfg) + "\n", encoding="utf-8")


# -------------------------------
# Directory Naming
# -------------------------------
# Abbreviation mapping for parameter names in directory naming
# 'ts' is reserved for timestamp, so use unique short names for other params
PARAM_ABBREV = {
    "rho_u": "rho",
    "emsa_iters": "iter",
    "decay": "decay",
    "minibatch": "mb",
    "classifier_heads": "heads",
    "target_class": "cls",
    "temp_start": "tstart",
    "temp_end": "tend",
    "temp_schedule": "tsched",
    "cost_scale": "cscale",
    "compensation_strategy": "comp",
    "convergence_mode": "conv",
    "cost_tol_abs": "ctolabs",
    "cost_tol_rel": "ctolrel",
    "reverse_kl": "rkl",
    "seed": "seed",
    "num_samples": "nsamp",
    "batch_size": "bs",
}


def _flatten_value(value: Any) -> str:
    """
    Recursively flatten a value (including nested lists) into a string for directory naming.

    Args:
        value: Any value (str, int, float, list, nested list, etc.)

    Returns:
        Flattened string representation joined by underscores
    """
    if isinstance(value, list):
        # Recursively flatten each element and join with underscore
        return "_".join(_flatten_value(v) for v in value)
    else:
        return str(value)


def _get_param_abbrev(param_name: str) -> str:
    """Get unique abbreviation for a parameter name."""
    if param_name in PARAM_ABBREV:
        return PARAM_ABBREV[param_name]
    # Default: use first 4 chars, but avoid 'ts' prefix (reserved for timestamp)
    abbrev = param_name[:4]
    if abbrev.startswith("ts"):
        abbrev = param_name[:5] if len(param_name) > 4 else param_name
    return abbrev


def build_outdir_name(
    tag: str,
    args: Dict[str, Any],
    timestamp: str,
    sweep_keys: List[str] = None,
) -> str:
    """
    Build a descriptive output directory name from job arguments.

    Includes ALL swept parameters to ensure unique directory names even when
    multiple jobs are generated within the same second.

    Args:
        tag: Prefix tag for the sweep
        args: Dictionary of job arguments
        timestamp: Timestamp string
        sweep_keys: List of parameter names being swept (for uniqueness)

    Returns:
        Directory name string
    """
    parts = [tag]

    # Track which params have been added to avoid duplicates
    added_params = set()

    # Handle classifier mode naming (existing logic, kept intact)
    if args.get('classifier_heads'):
        # Multi-head mode: use heads joined by underscore
        heads = args['classifier_heads']
        heads_str = _flatten_value(heads)
        parts.append(f"heads_{heads_str}")
        added_params.add('classifier_heads')
    elif 'target_class' in args:
        # Single-head mode: use target class
        parts.append(f"cls_{args['target_class']}")
        added_params.add('target_class')

    # Common parameters (existing logic, kept intact)
    common_params = ['rho_u', 'emsa_iters', 'decay', 'minibatch']
    for param in common_params:
        if param in args:
            abbrev = _get_param_abbrev(param)
            parts.append(f"{abbrev}_{_flatten_value(args[param])}")
            added_params.add(param)

    # Add any additional swept parameters not yet included
    if sweep_keys:
        for param in sweep_keys:
            if param not in added_params and param in args:
                abbrev = _get_param_abbrev(param)
                parts.append(f"{abbrev}_{_flatten_value(args[param])}")
                added_params.add(param)

    # Always end with timestamp
    parts.append(f"ts_{timestamp}")

    return "-".join(str(p) for p in parts)


# -------------------------------
# Run Logging
# -------------------------------
def build_run_dirs_log(logdir: Path) -> Tuple[Path, Path]:
    """
    Create log files to track successful and failed runs.

    Returns:
        Tuple of (success_log_path, failed_log_path)
    """
    timestamp_str = now_tag()
    log_path = logdir / f"run_dirs-{timestamp_str}.log"
    failed_log_path = logdir / f"failed_runs-{timestamp_str}.log"
    # Create empty log files
    log_path.write_text("", encoding="utf-8")
    failed_log_path.write_text("", encoding="utf-8")
    return log_path, failed_log_path


def append_run_dir_log(log_path: Path, outdir: Path, base_dir: Path) -> None:
    """Append a run directory path to the log file."""
    with log_path.open("a", encoding="utf-8") as f:
        outdir_log = str(outdir.relative_to(base_dir))
        f.write(outdir_log + "\n")


# -------------------------------
# Code Backup
# -------------------------------
def backup_code(
    outroot: Path,
    base_dir: Path,
    excludes: List[str] = None
) -> str:
    """
    Backup all Python files from base_dir to outroot/code_bak-{timestamp}.

    Args:
        outroot: Root directory for the backup
        base_dir: Source directory containing code to backup
        excludes: List of directory names to exclude (default: common large dirs)

    Returns:
        Path to the backup directory
    """
    if excludes is None:
        excludes = []

    timestamp_str = now_tag()
    backup_dir = outroot / f"code_bak-{timestamp_str}"
    ensure_dir(backup_dir)

    source_root = base_dir.resolve()

    # Convert excludes to absolute paths for comparison
    exclude_paths = set()
    for exc in excludes:
        exc_path = (source_root / exc).resolve()
        exclude_paths.add(exc_path)

    def is_excluded(path: Path) -> bool:
        """Check if path or any of its parents is in excludes."""
        path = path.resolve()
        for exc in exclude_paths:
            if path == exc or exc in path.parents:
                return True
        return False

    # Find all .py files and copy them with structure
    for py_file in source_root.rglob("*.py"):
        if is_excluded(py_file):
            continue

        # Compute relative path from source root
        rel_path = py_file.relative_to(source_root)
        dest_path = backup_dir / rel_path

        # Ensure destination directory exists
        ensure_dir(dest_path.parent)

        # Copy the file
        shutil.copy2(py_file, dest_path)

    return str(backup_dir)
