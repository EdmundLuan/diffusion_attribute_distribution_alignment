"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except (ImportError, RuntimeError):
    MPI_AVAILABLE = False
    MPI = None
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

# Global variable to store configured device
_configured_device = None


def setup_device(device_spec=None, distributed=False):
    """
    Configure device based on specification.

    Args:
        device_spec: Device specification string:
            - "0", "1", etc. for specific GPU
            - "0,1,2" for multiple GPUs
            - "cpu" for CPU
            - None for auto-detection (first available GPU or CPU)
        distributed: If True, sets up torch.distributed for multi-GPU training

    Returns:
        torch.device for the current process

    In single-process mode (distributed=False):
        - Sets CUDA_VISIBLE_DEVICES to restrict visible GPUs
        - Returns device for the first specified GPU

    In distributed mode (distributed=True):
        - Should be called before setup_dist()
        - Sets up environment for distributed training
        - Returns device for the current rank's GPU
    """
    global _configured_device

    if device_spec is None:
        # Auto-detect: use first GPU if available, else CPU
        if th.cuda.is_available():
            _configured_device = th.device("cuda:0")
        else:
            _configured_device = th.device("cpu")
        return _configured_device

    if device_spec.lower() == "cpu":
        _configured_device = th.device("cpu")
        return _configured_device

    # Parse GPU specification
    gpu_ids = [int(g.strip()) for g in device_spec.split(",")]

    if distributed and len(gpu_ids) > 1:
        # For distributed mode, set CUDA_VISIBLE_DEVICES to all specified GPUs
        # The actual device assignment will happen in setup_dist()
        os.environ["CUDA_VISIBLE_DEVICES"] = device_spec
        # Update GPUS_PER_NODE for this run
        global GPUS_PER_NODE
        GPUS_PER_NODE = len(gpu_ids)
        # Will be assigned properly after setup_dist()
        _configured_device = th.device("cuda:0")
    else:
        # Single-process mode: restrict to specified GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = device_spec
        _configured_device = th.device("cuda:0")

    return _configured_device


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if MPI_AVAILABLE:
        # Use MPI for distributed training
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
        comm = MPI.COMM_WORLD
        backend = "gloo" if not th.cuda.is_available() else "nccl"

        if backend == "gloo":
            hostname = "localhost"
        else:
            hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
        os.environ["RANK"] = str(comm.rank)
        os.environ["WORLD_SIZE"] = str(comm.size)

        port = comm.bcast(_find_free_port(), root=0)
        os.environ["MASTER_PORT"] = str(port)
        if th.cuda.is_available():
            device_id = th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
            dist.init_process_group(backend=backend, init_method="env://", device_id=device_id)
        else:
            dist.init_process_group(backend=backend, init_method="env://")
    else:
        # Single process setup without MPI
        backend = "gloo" if not th.cuda.is_available() else "nccl"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        try:
            if th.cuda.is_available():
                device_id = th.device("cuda:0")
                dist.init_process_group(backend=backend, init_method="env://", world_size=1, rank=0, device_id=device_id)
            else:
                dist.init_process_group(backend=backend, init_method="env://", world_size=1, rank=0)
        except Exception:
            # If initialization fails, continue without distributed training
            pass


def cleanup_dist():
    """
    Clean up the distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def dev():
    """
    Get the device to use for torch.distributed.

    If setup_device() was called, returns the configured device.
    Otherwise, returns cuda:0 if available, else cpu.
    """
    global _configured_device
    if _configured_device is not None:
        return _configured_device
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI_AVAILABLE:
        chunk_size = 2 ** 30  # MPI has a relatively small size limit
        if MPI.COMM_WORLD.Get_rank() == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
            num_chunks = len(data) // chunk_size
            if len(data) % chunk_size:
                num_chunks += 1
            MPI.COMM_WORLD.bcast(num_chunks)
            for i in range(0, len(data), chunk_size):
                MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
        else:
            num_chunks = MPI.COMM_WORLD.bcast(None)
            data = bytes()
            for _ in range(num_chunks):
                data += MPI.COMM_WORLD.bcast(None)
        return th.load(io.BytesIO(data), **kwargs)
    else:
        # Simple loading without MPI
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
