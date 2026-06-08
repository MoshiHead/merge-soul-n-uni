#!/usr/bin/env python3
# Copyright 2024-2025 The UniTalk / SoulX-FlashHead Authors.
"""Offline talking-head video generation via the UniTalk adapter.

Replaces the Wav2Vec2 audio encoder with Moshi's Mimi codec + LM and
the trained MoshiToWav2VecAdapter, so you can drive SoulX-FlashHead from
any WAV / MP3 / FLAC file without needing Wav2Vec2 for audio encoding.

Pipeline
--------
1.  ``WavToMoshiTokens.encode(audio_path)``
        WAV → Moshi transformer_out tokens  [N, 4096]  @ 12.5 Hz
2.  Sliding deque (8-second window, 100 tokens)
3.  ``get_audio_embedding_from_tokens(adapter, deque, frame_num, ...)``
        [100, 4096] → interpolate → adapter → windowed context
        → [1, frame_num, 5, 12, 768]  (same shape as Wav2Vec2 path)
4.  ``run_pipeline(pipeline, audio_emb)``
        DiT denoising (4 steps) + VAE decode → [frame_num, H, W, 3]
5.  Concatenate chunks, mux with original audio, save MP4.

Usage
-----
::

    python generate_video_adapter.py \\
        --soulx_root    /workspace/SoulX-FlashHead \\
        --ckpt_dir      /workspace/SoulX-FlashHead/models/SoulX-FlashHead-1_3B \\
        --wav2vec_dir   /workspace/SoulX-FlashHead/models/wav2vec2-base-960h \\
        --model_type    lite \\
        --cond_image    /workspace/SoulX-FlashHead/examples/girl.png \\
        --audio_path    /workspace/SoulX-FlashHead/examples/podcast_sichuan_16k.wav \\
        --adapter_ckpt  /workspace/merge-soul-n-uni/checkpoints/moshi_to_adapter/moshi_to_flashhead_phase1_best.pt \\
        --moshi_pkg     /workspace/merge-soul-n-uni/moshi/moshi

``--soulx_root`` points to the SoulX-FlashHead root directory that contains
``flash_head/``.  If omitted it defaults to the directory containing this
script (for development) or tries ``/workspace/SoulX-FlashHead`` (RunPod
default).
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import imageio
import numpy as np
import torch
from loguru import logger

# ── unitalk/ must be importable ────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))  # .../SoulX-FlashHead-original/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                # .../merge-soul-n-uni/

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate talking-head video using the UniTalk adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── SoulX root ──
    p.add_argument(
        "--soulx_root", default=None,
        help="Path to the SoulX-FlashHead root (the directory that contains "
             "flash_head/).  Defaults to /workspace/SoulX-FlashHead on RunPod "
             "or the directory containing this script for local development.",
    )
    # ── SoulX model paths ──
    p.add_argument("--ckpt_dir",    required=True,
                   help="SoulX-FlashHead-1_3B checkpoint directory")
    p.add_argument("--wav2vec_dir", required=True,
                   help="wav2vec2-base-960h directory (still loaded by the "
                        "pipeline at init; not used for audio encoding in "
                        "adapter mode)")
    p.add_argument("--model_type",  required=True, choices=["lite", "pro"],
                   help="Model variant")
    p.add_argument("--cond_image",  required=True,
                   help="Reference / conditioning portrait image")
    p.add_argument("--audio_path",  required=True,
                   help="Input audio file (WAV / MP3 / FLAC / OGG)")
    # ── Adapter ──
    p.add_argument("--adapter_ckpt", required=True,
                   help="Path to the trained MoshiToWav2VecAdapter .pt file")
    # ── Moshi encoder ──
    p.add_argument("--moshi_pkg",  default=None,
                   help="Path to the Moshi inner package directory "
                        "(parent of the moshi/ Python package). "
                        "Defaults to <project_root>/moshi/moshi")
    p.add_argument("--moshi_repo", default="kyutai/moshiko-pytorch-bf16",
                   help="HuggingFace repo ID for the Moshi checkpoint")
    # ── Output / misc ──
    p.add_argument("--save_file",   default=None,
                   help="Output MP4 path (auto-named under sample_results/ if omitted)")
    p.add_argument("--base_seed",   type=int, default=42)
    p.add_argument("--use_face_crop", action="store_true",
                   help="Run mediapipe face detection on the reference image")
    p.add_argument("--device",      default="cuda",
                   help="PyTorch device")
    return p.parse_args()


def _resolve_soulx_root(args_soulx_root: str | None) -> str:
    """Return the best available SoulX root directory."""
    candidates = []
    if args_soulx_root:
        candidates.append(os.path.abspath(args_soulx_root))
    # RunPod: original notebook clones to /workspace/SoulX-FlashHead
    candidates.append("/workspace/SoulX-FlashHead")
    # Side-by-side clone next to merge-soul-n-uni
    candidates.append(os.path.join(os.path.dirname(_PROJECT_ROOT), "SoulX-FlashHead"))
    # This script's own directory (development mode)
    candidates.append(_SCRIPT_DIR)

    for candidate in candidates:
        fh = os.path.join(candidate, "flash_head")
        if os.path.isdir(fh):
            return candidate

    # Fallback — let ensure_flashhead_imports raise a clear error
    return candidates[0] if candidates else _SCRIPT_DIR


# ══════════════════════════════════════════════════════════════════════════
# Video I/O
# ══════════════════════════════════════════════════════════════════════════

def save_video(
    frames_list: list[torch.Tensor],
    video_path: str,
    audio_path: str,
    fps: int,
) -> None:
    """Write frames to MP4 and mux with the original audio track."""
    tmp = video_path.replace(".mp4", "_tmp.mp4")
    with imageio.get_writer(
        tmp, format="mp4", mode="I",
        fps=fps, codec="h264",
        ffmpeg_params=["-bf", "0"],
    ) as writer:
        for frames in frames_list:
            frames_np = frames.numpy().astype(np.uint8)
            for i in range(frames_np.shape[0]):
                writer.append_data(frames_np[i])

    subprocess.run(
        ["ffmpeg", "-i", tmp, "-i", audio_path,
         "-c:v", "copy", "-c:a", "aac", "-shortest",
         video_path, "-y"],
        check=True,
    )
    os.remove(tmp)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    # ── 0. Resolve all paths ──────────────────────────────────────────────
    soulx_root = _resolve_soulx_root(args.soulx_root)
    logger.info(f"SoulX root: {soulx_root}")

    if args.moshi_pkg is None:
        args.moshi_pkg = os.path.join(_PROJECT_ROOT, "moshi", "moshi")

    args.adapter_ckpt = os.path.abspath(args.adapter_ckpt)
    args.ckpt_dir     = os.path.abspath(args.ckpt_dir)
    args.wav2vec_dir  = os.path.abspath(args.wav2vec_dir)
    args.cond_image   = os.path.abspath(args.cond_image)
    args.audio_path   = os.path.abspath(args.audio_path)

    # ── 1. Load SoulX-FlashHead pipeline via unitalk's import helper ──────
    # ensure_flashhead_imports handles sys.path + chdir so flash_head/ is
    # always found from the correct SoulX root, regardless of where this
    # script lives.
    from unitalk.utils import ensure_flashhead_imports
    fh = ensure_flashhead_imports(soulx_root)

    get_pipeline      = fh["get_pipeline"]
    get_base_data     = fh["get_base_data"]
    get_infer_params  = fh["get_infer_params"]
    run_pipeline      = fh["run_pipeline"]

    logger.info("Loading SoulX-FlashHead pipeline (DiT + VAE + Wav2Vec2)...")
    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir=args.ckpt_dir,
        model_type=args.model_type,
        wav2vec_dir=args.wav2vec_dir,
    )
    get_base_data(
        pipeline,
        cond_image_path_or_dir=args.cond_image,
        base_seed=args.base_seed,
        use_face_crop=args.use_face_crop,
    )
    infer_params      = get_infer_params()
    frame_num         = int(infer_params["frame_num"])           # 33
    motion_frames_num = int(infer_params["motion_frames_num"])   # 9 (lite) / 5 (pro)
    slice_len         = frame_num - motion_frames_num            # 24 / 28
    tgt_fps           = int(infer_params["tgt_fps"])             # 25

    # ── 2. Load UniTalk adapter ───────────────────────────────────────────
    from unitalk.models.adapter import MoshiToWav2VecAdapter
    from unitalk.settings import MOSHI_DIM, WAV2VEC_DIM, WAV2VEC_LAYERS, DEQUE_SIZE

    logger.info("Loading MoshiToWav2VecAdapter...")
    adapter = MoshiToWav2VecAdapter(
        moshi_dim=MOSHI_DIM,
        hidden_dim=WAV2VEC_DIM,
        num_layers=WAV2VEC_LAYERS,
    )
    loaded = adapter.load_checkpoint(args.adapter_ckpt)
    if not loaded:
        logger.warning(
            f"[Adapter] Checkpoint NOT found at {args.adapter_ckpt}\n"
            "  Running with random adapter weights.  "
            "The video will animate but will NOT lip-sync until a trained "
            "checkpoint is provided."
        )
    adapter = adapter.to(args.device).eval()
    adapter_params = sum(p.numel() for p in adapter.parameters())
    logger.info(
        f"[Adapter] {'Loaded ✓' if loaded else 'Random init ⚠'} — "
        f"{adapter_params:,} params ({adapter_params * 4 / 1e6:.1f} MB @ fp32)"
    )

    # ── 3. Encode WAV → Moshi transformer_out tokens ─────────────────────
    from unitalk.offline_encoder import WavToMoshiTokens

    logger.info("Encoding audio with Moshi (Mimi codec + LM)…")
    encoder = WavToMoshiTokens(
        moshi_pkg=args.moshi_pkg,
        hf_repo=args.moshi_repo,
        device=args.device,
    )
    encoder.load()
    all_tokens = encoder.encode(args.audio_path)   # [N_total, 4096] on CPU
    N_total = all_tokens.shape[0]

    # ── 4. Build sliding deque and generate video chunks ─────────────────
    from unitalk.streaming import get_audio_embedding_from_tokens

    # Each video slice corresponds to `slice_len` frames at 25 fps = slice_len×40ms.
    # Moshi runs at 12.5 Hz (one token per 80 ms), so:
    #   tokens_per_chunk = slice_len × 40ms / 80ms = slice_len / 2
    #   Lite: slice_len=24 → tokens_per_chunk=12  (12 × 80ms = 960ms of audio)
    #   Pro:  slice_len=28 → tokens_per_chunk=14  (14 × 80ms = 1120ms of audio)
    tokens_per_chunk = slice_len // 2

    n_chunks = max(1, math.ceil(N_total / tokens_per_chunk))

    # Deque initialised with silence (zeros), FIFO of the last 8 s of tokens.
    token_deque: deque[torch.Tensor] = deque(maxlen=DEQUE_SIZE)
    for _ in range(DEQUE_SIZE):
        token_deque.append(torch.zeros(MOSHI_DIM))

    generated: list[torch.Tensor] = []

    logger.info(
        f"Generating video: {N_total} tokens @ 12.5 Hz "
        f"({N_total / 12.5:.1f}s audio) → "
        f"{n_chunks} chunks × {slice_len} frames "
        f"({n_chunks * slice_len / tgt_fps:.1f}s video)"
    )

    for chunk_idx in range(n_chunks):
        # Slide deque forward by tokens_per_chunk new tokens (or silence if
        # the audio ended before the last chunk).
        start = chunk_idx * tokens_per_chunk
        end   = min(start + tokens_per_chunk, N_total)
        for tok in all_tokens[start:end]:
            token_deque.append(tok)
        for _ in range(tokens_per_chunk - (end - start)):   # silence padding
            token_deque.append(torch.zeros(MOSHI_DIM))

        # Snapshot → [DEQUE_SIZE, 4096] on CPU.
        snap = torch.stack(list(token_deque), dim=0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            audio_emb = get_audio_embedding_from_tokens(
                adapter, snap, frame_num,
                device=torch.device(args.device),
                dtype=pipeline.param_dtype,
            )
            video = run_pipeline(pipeline, audio_emb)

        # Discard motion-carry frames (streaming mode).
        video = video[motion_frames_num:]

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"  chunk {chunk_idx + 1}/{n_chunks}: "
            f"{video.shape[0]} frames  "
            f"{elapsed_ms:.0f}ms  "
            f"({elapsed_ms / max(1, video.shape[0]):.1f}ms/frame)"
        )
        generated.append(video.cpu())

    # ── 5. Save MP4 ───────────────────────────────────────────────────────
    if args.save_file is None:
        out_dir = os.path.join(_SCRIPT_DIR, "sample_results")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.save_file = os.path.join(out_dir, f"res_adapter_{ts}.mp4")

    logger.info(f"Saving → {args.save_file}")
    save_video(generated, args.save_file, args.audio_path, tgt_fps)
    total_frames = sum(v.shape[0] for v in generated)
    logger.info(
        f"Done.  {total_frames} frames @ {tgt_fps} fps "
        f"= {total_frames / tgt_fps:.1f}s  →  {args.save_file}"
    )


if __name__ == "__main__":
    main()
