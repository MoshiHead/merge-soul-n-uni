"""Shared helpers: queue ops, audio (re)encoding, lazy upstream imports.

Kept in one module since each piece is small and the three groups are
clearly delimited by section banners below.
"""

from __future__ import annotations

import base64
import os
import queue
import sys
from typing import Any

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Queue helpers
# ──────────────────────────────────────────────────────────────────────────

def put_latest(q: "queue.Queue[Any]", item: Any) -> None:
    """Bounded insert — drops the oldest item when the queue is full."""
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# ──────────────────────────────────────────────────────────────────────────
# Audio I/O — resampling and base64 encoding for WebSocket transport
# ──────────────────────────────────────────────────────────────────────────

def resample_mono_f32(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono float audio with soxr if available, else linear interp."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or src_sr == dst_sr:
        return audio
    try:
        import soxr
        return np.asarray(soxr.resample(audio, src_sr, dst_sr), dtype=np.float32)
    except Exception:
        dst_len = max(1, int(round(audio.size * float(dst_sr) / float(src_sr))))
        x_old = np.linspace(0, 1, num=audio.size, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0, 1, num=dst_len, endpoint=False, dtype=np.float64)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def encode_jpeg_b64(frame_rgb: np.ndarray, quality: int = 95) -> str | None:
    """Encode an RGB frame to JPEG base64 (returns None on failure).

    Default quality 95 is the sweet spot for streaming:
      - Visually indistinguishable from the model's raw RGB output
        (skin texture, eye edges, hair strands all preserved).
      - Encoding cost is ~2ms per 512×512 frame on modern CPU — well
        under the 40ms per-frame budget of the 25fps dispatcher.
      - Bandwidth ~15 Mbps at 25fps, comfortable on any modern link
        (LAN, RunPod, residential broadband).
    Drop to 85 only if you need to fit a constrained network.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def encode_pcm_s16le_b64(audio_f32: np.ndarray) -> str:
    """Encode float32 audio in [-1, 1] to PCM Int16 base64."""
    pcm = np.clip(audio_f32, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    return base64.b64encode(pcm_i16.tobytes()).decode("ascii")


# ──────────────────────────────────────────────────────────────────────────
# Lazy upstream imports (Moshi + SoulX-FlashHead live as sibling source trees)
# ──────────────────────────────────────────────────────────────────────────

_moshi_cache: dict[str, Any] | None = None
_personaplex_cache: dict[str, Any] | None = None
_flashhead_cache: dict[str, Any] | None = None


def ensure_moshi_imports(moshi_pkg: str) -> dict[str, Any]:
    """Import Moshi ``LMGen`` + ``CheckpointInfo``, caching the result.

    Args:
        moshi_pkg: Absolute path to the inner ``moshi/moshi`` directory.
    """
    global _moshi_cache
    if _moshi_cache is not None:
        return _moshi_cache
    if _personaplex_cache is not None:
        raise RuntimeError(
            "PersonaPlex was already imported in this process — cannot also "
            "import vanilla Moshi (both packages are named `moshi`). Relaunch "
            "with --s2s-engine moshi only."
        )

    if moshi_pkg not in sys.path:
        sys.path.insert(0, moshi_pkg)

    from moshi.models import LMGen
    from moshi.models.loaders import CheckpointInfo

    _moshi_cache = {"LMGen": LMGen, "CheckpointInfo": CheckpointInfo}
    print("[Moshi] Imports OK.")
    return _moshi_cache


def ensure_personaplex_imports(pplex_pkg: str) -> dict[str, Any]:
    """Import PersonaPlex's loader API, caching the result.

    PersonaPlex ships its own fork of the ``moshi`` package — same import
    name, different code path. Only one of the two trees can live on
    ``sys.path`` per process; the engine factory guarantees this by
    calling exactly one of ``ensure_moshi_imports`` /
    ``ensure_personaplex_imports`` per process.

    Args:
        pplex_pkg: Absolute path to ``personaplex/moshi`` (one level up
            from the ``moshi/`` python package).
    """
    global _personaplex_cache
    if _personaplex_cache is not None:
        return _personaplex_cache
    if _moshi_cache is not None:
        raise RuntimeError(
            "Vanilla Moshi was already imported in this process — cannot "
            "also import PersonaPlex (both packages are named `moshi`). "
            "Relaunch with --s2s-engine personaplex only."
        )

    if pplex_pkg not in sys.path:
        sys.path.insert(0, pplex_pkg)

    from moshi.models import LMGen
    from moshi.models import loaders as loaders_mod
    from moshi.models.loaders import get_mimi, get_moshi_lm
    from huggingface_hub import hf_hub_download
    from sentencepiece import SentencePieceProcessor

    _personaplex_cache = {
        "LMGen": LMGen,
        "loaders": loaders_mod,
        "get_mimi": get_mimi,
        "get_moshi_lm": get_moshi_lm,
        "hf_hub_download": hf_hub_download,
        "SentencePieceProcessor": SentencePieceProcessor,
    }
    print("[PersonaPlex] Imports OK.")
    return _personaplex_cache


def ensure_flashhead_imports(soulx_root: str) -> dict[str, Any]:
    """Import SoulX-FlashHead pipeline entry points, caching the result.

    The upstream module performs path-relative file discovery during import,
    so we temporarily ``chdir`` into the SoulX root.
    """
    global _flashhead_cache
    if _flashhead_cache is not None:
        return _flashhead_cache

    if soulx_root not in sys.path:
        sys.path.insert(0, soulx_root)

    prev_cwd = os.getcwd()
    try:
        os.chdir(soulx_root)
        from flash_head.inference import (
            get_base_data,
            get_infer_params,
            get_pipeline,
            run_pipeline,
        )
    finally:
        os.chdir(prev_cwd)

    _flashhead_cache = {
        "get_pipeline": get_pipeline,
        "get_base_data": get_base_data,
        "get_infer_params": get_infer_params,
        "run_pipeline": run_pipeline,
    }
    print("[FlashHead] Imports OK.")
    return _flashhead_cache
