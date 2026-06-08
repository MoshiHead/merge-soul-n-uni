"""Core function that converts a Moshi token deque into a FlashHead audio embedding.

Imported by :mod:`unitalk.models.flashhead_engine` for both real-time streaming
and offline (batch) inference.  The function is the only bridge between the
adapter's output format and the ``[1, frame_num, 5, 12, 768]`` tensor that
SoulX-FlashHead's ``AudioProjModel`` expects.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .settings import INTERP_TARGET, WAV2VEC_DIM, WAV2VEC_LAYERS
from .models.adapter import MoshiToWav2VecAdapter


@torch.no_grad()
def get_audio_embedding_from_tokens(
    adapter: MoshiToWav2VecAdapter,
    deque_snapshot: torch.Tensor,
    frame_num: int,
    device: torch.device,
    dtype: torch.dtype,
    context_radius: int = 2,
) -> torch.Tensor:
    """Convert a DEQUE_SIZE-token snapshot into a FlashHead audio embedding.

    Mirrors the ``get_audio_embedding`` function in ``flash_head/inference.py``
    but replaces Wav2Vec2 with the adapter:

    Wav2Vec2 path (original)::

        raw audio → CNN extractor → linear interpolation → 12 transformer layers
        → stack hidden states → [T, 12, 768]
        → windowed context → [1, frame_num, 5, 12, 768]

    Adapter path (this function)::

        Moshi tokens [100, 4096]
        → linear interpolation → [200, 4096]
        → adapter (12 transformer layers) → [200, 12, 768]
        → windowed context → [1, frame_num, 5, 12, 768]

    Args:
        adapter:         Trained MoshiToWav2VecAdapter in eval mode on ``device``.
        deque_snapshot:  ``[DEQUE_SIZE, MOSHI_DIM]`` — last 8 s of Moshi tokens.
                         May be on CPU; will be moved to ``device`` internally.
        frame_num:       Video frames per chunk (33 for both Lite and Pro).
        device:          CUDA / CPU device.
        dtype:           Pipeline parameter dtype (bfloat16 / float32).
        context_radius:  Half-width of the context window used by AudioProjModel
                         (default 2, giving 5-frame windows matching SoulX training).

    Returns:
        ``[1, frame_num, 2*context_radius+1, WAV2VEC_LAYERS, WAV2VEC_DIM]``
        float tensor on ``device`` matching the format expected by FlashHead's
        ``AudioProjModel``.
    """
    # ── 1. Move to device; float32 for numerically stable adapter forward ──
    x = deque_snapshot.to(device=device, dtype=torch.float32)  # [DEQUE_SIZE, 4096]

    # ── 2. Linear-interpolate DEQUE_SIZE (100) → INTERP_TARGET (200) ──────
    #   Wav2Vec2's CNN extractor upsamples CNN features to seq_len=200 before
    #   entering its 12 transformer layers.  We do the same here so the adapter
    #   always sees the full 8-second context at 25-fps resolution.
    x = x.t().unsqueeze(0)                                     # [1, 4096, 100]
    x = F.interpolate(x, size=INTERP_TARGET, mode="linear", align_corners=True)
    x = x.squeeze(0).t()                                       # [200, 4096]

    # ── 3. Adapter forward → per-layer hidden states ───────────────────────
    hidden = adapter(x)                                        # [200, 12, 768]

    # ── 4. Extract the last `frame_num` positions (most recent audio) ──────
    audio_start = INTERP_TARGET - frame_num                   # 200 - 33 = 167
    audio_end   = INTERP_TARGET                               # 200

    # ── 5. Build sliding ±context_radius window for each frame ─────────────
    #   Matches the windowing in flash_head/inference.py::get_audio_embedding:
    #     indices = (torch.arange(2*2+1) - 2) * 1   →  [-2,-1,0,1,2]
    #     center_indices = arange(start, end).unsqueeze(1) + indices.unsqueeze(0)
    window  = torch.arange(-context_radius, context_radius + 1)        # [5]
    centers = torch.arange(audio_start, audio_end)                     # [33]
    indices = centers.unsqueeze(1) + window.unsqueeze(0)                # [33, 5]
    indices = indices.clamp(0, INTERP_TARGET - 1)

    # ── 6. Gather → [frame_num, 5, 12, 768], add batch dim, cast ──────────
    embedding = hidden[indices]                                # [33, 5, 12, 768]
    return embedding.unsqueeze(0).to(dtype=dtype).contiguous()  # [1, 33, 5, 12, 768]
