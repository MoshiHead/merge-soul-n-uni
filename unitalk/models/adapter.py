"""MoshiToWav2VecAdapter — 12-layer transformer that replaces Wav2Vec2.

Architectural overview
----------------------
FlashHead's ``AudioProjModel`` was trained on Wav2Vec2 hidden states stacked
from all 12 transformer layers. Each layer captures a different level of
audio abstraction:

  * Layers 1–3:  low-level acoustic features (phonemes, pitch, energy)
  * Layers 4–8:  mid-level features (syllables, prosody, rhythm)
  * Layers 9–12: high-level features (semantic, linguistic, speaker)

A simple linear projection cannot replicate this multi-level structure.
This adapter uses the same 12-layer transformer architecture as Wav2Vec2-base
so each layer learns to produce outputs at the corresponding abstraction
level, exactly matching what ``AudioProjModel`` expects.

What it replaces (vs standard FlashHead)
----------------------------------------
Standard Wav2Vec2 pipeline::

    Raw audio → CNN feature extractor (7 conv layers → 512-dim)
              → Linear interpolation (→ 200 tokens)
              → Feature projection (512 → 768)
              → 12 transformer layers → 12 × (1, 200, 768)
              → Stack → (200, 12, 768)

Our adapter::

    Moshi tokens → Linear interpolation (100 → 200, done BEFORE adapter)
                 → Feature projection (4096 → 768)
                 → 12 transformer layers → 12 × (1, 200, 768)
                 → Stack → (200, 12, 768)  ← SAME OUTPUT FORMAT

Parameter count
---------------
  * Feature projection:   4096 × 768 + norms  ≈   3.2M
  * Conv position:        768 × 48 × 128      ≈   4.7M
  * 12 Transformer layers: 12 × ~7.1M          ≈  85.0M
  * Total:                                     ≈  93.0M (vs Wav2Vec2-base: 95M)

Checkpoint
----------
``load_checkpoint(path)`` takes the full ``.pt`` path. Both absolute and
relative paths work — relative paths are resolved against the project root
in ``settings.resolve_path``.

I/O
---
  Input:  ``[N, 4096]``  or ``[B, N, 4096]``
  Output: ``[N, 12, 768]`` or ``[B, N, 12, 768]``
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..settings import MOSHI_DIM, WAV2VEC_DIM, WAV2VEC_LAYERS


# ═══════════════════════════════════════════════════════════════════════════
#  Canonical adapter filename + top-K retention utilities
# ═══════════════════════════════════════════════════════════════════════════
#
# All adapter checkpoints — both the training working dir and the
# per-engine inference dir — follow ONE filename pattern:
#
#     {engine}_to_flashhead_phase{N}_{tag}.pt
#
# Examples:
#     moshi_to_flashhead_phase1_best.pt          ← Phase 1 final best
#     moshi_to_flashhead_phase1_latest.pt        ← Phase 1 latest epoch
#     moshi_to_flashhead_phase1_ep7.pt           ← Phase 1 top-K history
#     personaplex_to_flashhead_phase2_best.pt    ← Phase 2 final best
#     personaplex_to_flashhead_phase2_step1500.pt← Phase 2 mid-train save
#
# Retention policy: per phase, per engine dir, we keep exactly:
#     * 1 ``_best.pt``   (current best, overwritten by each new best)
#     * 1 ``_latest.pt`` (last epoch's weights, overwritten every epoch)
#     * up to 3 ``_ep{N}.pt`` runner-ups (2nd/3rd/4th best by val_loss)
# = 5 files maximum per (dir × phase). The 4th-best gets evicted when a
# new best arrives. (Phase 2 ``_step{N}.pt`` mid-training checkpoints
# are rotated separately — they're for crash recovery, not ranking.)
#
# Side-car index: a small JSON file per (dir × phase) records each ep's
# val_loss so eviction can pick the worst correctly. Stored at
#     {dir}/{engine}_to_flashhead_phase{N}_index.json

import glob
import json


def adapter_filename(engine: str, phase: int, tag: str) -> str:
    """Build the canonical adapter filename.

    Pattern: ``{engine}_to_flashhead_phase{phase}_{tag}.pt``

    Args:
        engine: "moshi" or "personaplex".
        phase:  1 (feature distillation) or 2 (end-to-end distillation).
        tag:    short identifier — ``"best"``, ``"latest"``, ``"ep5"``,
                ``"step1500"``, ``"converged"``, etc. Goes verbatim
                into the filename. Use ``"best"`` / ``"latest"`` for the
                no-epoch slots; use ``"ep{N}"`` / ``"step{N}"`` for
                history.

    Examples:
        adapter_filename("moshi", 1, "best")
            -> "moshi_to_flashhead_phase1_best.pt"
        adapter_filename("personaplex", 2, "ep7")
            -> "personaplex_to_flashhead_phase2_ep7.pt"
    """
    return f"{engine}_to_flashhead_phase{phase}_{tag}.pt"


def _index_path(dir_path, engine: str, phase: int):
    return f"{dir_path}/{engine}_to_flashhead_phase{phase}_index.json"


def update_top_k_index(dir_path, engine: str, phase: int,
                        epoch: int, val_loss: float) -> None:
    """Record an epoch's val_loss in the side-car JSON index.

    Called whenever we save an ``_ep{N}.pt`` file so the rotation logic
    knows which ones to evict.
    """
    path = _index_path(dir_path, engine, phase)
    index = {}
    if os.path.isfile(path):
        try:
            with open(path) as f:
                index = json.load(f)
        except Exception:
            index = {}
    index.setdefault("epochs", {})[str(epoch)] = float(val_loss)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


def rotate_top_k_epochs(dir_path, engine: str, phase: int,
                         k: int = 3) -> list[str]:
    """Delete ``_ep{N}.pt`` files outside the top-K by val_loss.

    Reads val_loss for each epoch from the side-car JSON index. Files
    without an index entry are treated as ``inf`` (will be evicted
    first). Returns the list of deleted basenames.
    """
    dir_path = str(dir_path)
    prefix = f"{engine}_to_flashhead_phase{phase}"
    ep_files = glob.glob(os.path.join(dir_path, f"{prefix}_ep*.pt"))
    if len(ep_files) <= k:
        return []

    # Load the index.
    idx_path = _index_path(dir_path, engine, phase)
    index = {}
    if os.path.isfile(idx_path):
        try:
            with open(idx_path) as f:
                index = json.load(f)
        except Exception:
            index = {}
    epochs = index.get("epochs", {})

    # Score each file by val_loss; missing entries → inf (evict first).
    scored = []
    for f in ep_files:
        basename = os.path.basename(f)
        ep_str = basename[len(prefix) + len("_ep"):-len(".pt")]
        loss = float(epochs.get(ep_str, float("inf")))
        scored.append((loss, f, ep_str))
    scored.sort(key=lambda x: x[0])

    evicted = []
    for loss, f, ep_str in scored[k:]:
        os.unlink(f)
        evicted.append(os.path.basename(f))
        epochs.pop(ep_str, None)

    # Persist the trimmed index.
    index["epochs"] = epochs
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)

    return evicted


class MoshiToWav2VecAdapter(nn.Module):
    """Transformer-based adapter replicating the Wav2Vec2-base encoder."""

    def __init__(
        self,
        moshi_dim: int = MOSHI_DIM,
        hidden_dim: int = WAV2VEC_DIM,     # 768 (same as Wav2Vec2-base)
        num_layers: int = WAV2VEC_LAYERS,  # 12  (same as Wav2Vec2-base)
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,              # no dropout at inference
        conv_pos_kernel: int = 128,
        conv_pos_groups: int = 16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Feature projection — replaces Wav2Vec2's CNN + feature_projection.
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(moshi_dim),
            nn.Linear(moshi_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Conv1d positional encoding (Wav2Vec2-style — large receptive field).
        self.conv_pos = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=conv_pos_kernel,
            padding=conv_pos_kernel // 2,
            groups=conv_pos_groups,
        )
        self.conv_pos_gelu = nn.GELU()

        # Pre-transformer LayerNorm + dropout.
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # 12 Transformer encoder layers — kept as a ModuleList instead of
        # nn.TransformerEncoder so we can collect per-layer hidden states.
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass mimicking the Wav2Vec2-base encoder.

        Args:
            x: ``[N, 4096]`` or ``[B, N, 4096]`` — Moshi tokens after interpolation.

        Returns:
            ``[N, 12, 768]`` or ``[B, N, 12, 768]`` — per-layer hidden states.
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)                       # [1, N, 4096]
            squeeze = True

        B, N, _ = x.shape

        # 1. Feature projection 4096 → 768.
        x = self.feature_projection(x)               # [B, N, 768]

        # 2. Convolutional position encoding.
        x_conv = x.transpose(1, 2)                   # [B, 768, N]
        x_conv = self.conv_pos(x_conv)               # [B, 768, N + k//2]
        x_conv = x_conv[:, :, :N]
        x_conv = self.conv_pos_gelu(x_conv)
        x = x + x_conv.transpose(1, 2)               # [B, N, 768]

        # 3. LayerNorm + dropout.
        x = self.layer_norm(x)
        x = self.input_dropout(x)

        # 4. Forward through 12 transformer layers, collecting per-layer states.
        hidden_states: list[torch.Tensor] = []
        for layer in self.transformer_layers:
            x = layer(x)                             # [B, N, 768]
            hidden_states.append(x)

        # 5. Stack → [B, N, 12, 768].
        output = torch.stack(hidden_states, dim=2)
        if squeeze:
            output = output.squeeze(0)               # [N, 12, 768]
        return output

    # ── Checkpoint helpers ──────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> str:
        """Save adapter weights to ``path`` (creates parent dirs if needed)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[Adapter] Saved checkpoint → {path}")
        return path

    def load_checkpoint(self, path: str) -> bool:
        """Load adapter weights from ``path``.

        Returns True if the file was found and loaded, False otherwise.
        """
        if not os.path.isfile(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        print(f"[Adapter] Loaded fine-tuned checkpoint ← {path}")
        return True
