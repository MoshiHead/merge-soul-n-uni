"""Multi-GPU coordination for the UniTalk streaming pipeline.

UniTalk runs as ONE process per GPU under ``torchrun``. Rank 0 owns the
FastAPI server, the WebSocket, Moshi, and the adapter. Rank ≠ 0 holds
only the FlashHead pipeline and follows rank 0's commands. xfuser's
sequence-parallel attention requires every rank to call the same
forward() in lockstep over NCCL — that is what this module orchestrates.

Three pieces:

* :class:`DistContext` — singleton snapshot of ``RANK / WORLD_SIZE /
  LOCAL_RANK`` taken once at startup. The init does NOT create the NCCL
  process group itself; SoulX's ``get_pipeline`` (via ``get_device`` →
  ``dist.init_process_group``) does that during FlashHead construction.
  This module just reads env vars + records what was assigned.

* Command codes (``CMD_*``) — the lockstep RPC vocabulary spoken
  between rank 0 and worker ranks.

* :func:`broadcast_cmd` / :func:`broadcast_tensor` — thin wrappers used
  by both rank 0 (issuer) and worker ranks (receivers).

Single-GPU mode (``WORLD_SIZE`` unset or 1) takes the no-op path
everywhere — every call short-circuits and the code behaves exactly as
the original single-process pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


# ──────────────────────────────────────────────────────────────────────────
# Command vocabulary spoken on the rank-0 → workers broadcast channel.
# Encoded as a single int64 scalar so we can broadcast a 1-element tensor.
# ──────────────────────────────────────────────────────────────────────────

CMD_STOP: int = 0           # No payload. Worker exits its loop.
CMD_RUN_PIPELINE: int = 1   # Followed by an audio-embedding broadcast.
CMD_RESET_REF: int = 2      # No payload. Worker calls reset_person_name.


# ──────────────────────────────────────────────────────────────────────────
# Process-wide singleton describing the distributed layout.
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    # True iff WORLD_SIZE > 1 — i.e. we were launched under torchrun /
    # an explicit multi-GPU setup. Used to short-circuit broadcasts when
    # there's nobody to talk to.
    multi_gpu: bool = False

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        """Device string for the rank-local GPU. CPU fallback if no CUDA."""
        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{self.local_rank}"


_CTX: DistContext = DistContext()


def init_dist_context() -> DistContext:
    """Read ``RANK / WORLD_SIZE / LOCAL_RANK`` from the environment and
    record them in the process-wide :class:`DistContext`. Idempotent.

    We do NOT call ``dist.init_process_group`` here. SoulX-FlashHead's
    ``get_pipeline`` does that (via ``usp_device.get_device``) the first
    time it's invoked with ``world_size > 1``. Doing it twice trips a
    "process group already initialized" error.
    """
    global _CTX
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _CTX = DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        multi_gpu=world_size > 1,
    )
    return _CTX


def get_dist_context() -> DistContext:
    """Return the global :class:`DistContext` (call ``init_dist_context``
    once at startup before using this)."""
    return _CTX


# ──────────────────────────────────────────────────────────────────────────
# Lockstep RPC helpers.
# ──────────────────────────────────────────────────────────────────────────

def broadcast_cmd(cmd: int, device: Optional[str] = None) -> int:
    """Rank 0 sends a command code; worker ranks receive it.

    Returns the command on every rank. In single-GPU mode it's a no-op
    that returns the input unchanged (useful so the caller can `if cmd
    == CMD_STOP: break` symmetrically on every rank).
    """
    ctx = _CTX
    if not ctx.multi_gpu:
        return cmd
    if not dist.is_initialized():
        # Process group never came up (e.g. xfuser init failed). Fall
        # back to single-GPU behavior rather than hanging.
        return cmd

    dev = device or ctx.device
    t = torch.tensor([cmd], dtype=torch.long, device=dev)
    dist.broadcast(t, src=0)
    return int(t.item())


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """In-place tensor broadcast from ``src`` (default 0) to all ranks.

    On rank ``src`` the input is the source data. On other ranks the
    tensor must be pre-allocated with the matching shape + dtype + device;
    the contents get overwritten with src's data.

    No-op in single-GPU mode.
    """
    ctx = _CTX
    if not ctx.multi_gpu or not dist.is_initialized():
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    dist.broadcast(tensor, src=src)
    return tensor


def barrier() -> None:
    """Optional sync barrier. No-op outside multi-GPU mode."""
    if _CTX.multi_gpu and dist.is_initialized():
        dist.barrier()


def shutdown_workers(device: Optional[str] = None) -> None:
    """Rank 0: tell every worker rank to exit its loop. Safe to call
    multiple times; safe to call in single-GPU mode (no-op)."""
    if not _CTX.multi_gpu or not _CTX.is_rank0:
        return
    if not dist.is_initialized():
        return
    try:
        broadcast_cmd(CMD_STOP, device=device)
    except Exception as e:
        # Best-effort — workers may already be torn down on hard exits.
        print(f"[dist] shutdown_workers warning: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Worker loop for non-zero ranks.
# ──────────────────────────────────────────────────────────────────────────

def run_worker_loop(flash_engine) -> None:
    """Block on the broadcast channel and execute commands until rank 0
    sends ``CMD_STOP``. Only called on rank ≠ 0.

    The flow per loop iteration mirrors rank 0's pipeline call site
    (``_run_pipeline_lockstep`` in flashhead_engine.py):

      1. Receive the command code from rank 0.
      2. ``CMD_STOP``                — break.
      3. ``CMD_RESET_REF``           — call ``reset_person_name`` so our
                                       latent_motion_frames realign with
                                       rank 0's fresh-session state.
      4. ``CMD_RUN_PIPELINE``        — pre-allocate the audio embedding
                                       buffer, receive rank 0's tensor
                                       into it, then call ``run_pipeline``.
                                       xfuser's sequence-parallel attention
                                       does the cross-rank exchange inside
                                       the model forward.

    Output of the local ``run_pipeline`` is discarded — only rank 0's
    output reaches the WebSocket dispatcher.
    """
    # Import here to avoid a circular dep with flashhead_engine.
    from .utils import ensure_flashhead_imports

    ctx = _CTX
    assert not ctx.is_rank0, "run_worker_loop is for non-zero ranks only"
    assert flash_engine.pipeline is not None, "flash_engine.load() must run first"

    fh = ensure_flashhead_imports(flash_engine.soulx_root)

    print(f"[worker rank={ctx.rank}] Entering broadcast loop.")
    try:
        while True:
            cmd = broadcast_cmd(0)
            if cmd == CMD_STOP:
                print(f"[worker rank={ctx.rank}] CMD_STOP received. Exiting.")
                break

            if cmd == CMD_RESET_REF:
                flash_engine.pipeline.reset_person_name(
                    flash_engine.pipeline.person_name
                )
                continue

            if cmd == CMD_RUN_PIPELINE:
                audio_emb = torch.empty(
                    flash_engine.embedding_shape,
                    dtype=flash_engine.embedding_dtype,
                    device=flash_engine.device,
                )
                broadcast_tensor(audio_emb)
                with torch.inference_mode():
                    _ = fh["run_pipeline"](flash_engine.pipeline, audio_emb)
                continue

            # Unknown command — print + ignore. Don't break, because
            # the rank-0 side might recover; if it doesn't, CMD_STOP
            # will eventually arrive on shutdown.
            print(f"[worker rank={ctx.rank}] Unknown command {cmd}; ignoring.")
    except Exception as e:
        print(f"[worker rank={ctx.rank}] Loop error: {e!r}. Exiting.")
        import traceback
        traceback.print_exc()
