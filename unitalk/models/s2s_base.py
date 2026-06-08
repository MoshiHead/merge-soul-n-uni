"""Engine-agnostic protocol for speech-to-speech backbones.

UniTalk supports multiple S2S models (Moshi Helium, NVIDIA PersonaPlex, …).
They all feed the same downstream pipeline: a 12.5 Hz stream of
``(token_id, transformer_out_cpu, audio_pcm_np, arrival_ts)`` tuples,
where ``transformer_out_cpu`` is the ``[1, 1, 4096]`` Helium hidden
state consumed by ``MoshiToWav2VecAdapter``.

This module defines the contract every engine must honour so that
``server.py`` / ``streaming.py`` / ``flashhead_engine.py`` stay
engine-agnostic — they only see ``S2SEngine`` and the wire-format tuple.
"""

from __future__ import annotations

import queue
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class S2SEngine(Protocol):
    """Minimal interface every S2S backbone implements.

    The concrete engines (``MoshiEngine``, ``PersonaPlexEngine``) ship the
    LM + audio codec internally and emit the wire-format tuple on
    ``token_queue`` at 12.5 Hz.
    """

    # ── State attributes consumed by callers ──────────────────────────
    device: str
    loaded: bool

    # ── Lifecycle ─────────────────────────────────────────────────────
    def load(self) -> None:
        """Load weights + enter streaming context. Idempotent."""
        ...

    def reset_streaming_state(self) -> None:
        """Reset per-session streaming state (cache, offsets) without
        tearing down the streaming context. Called once per new
        WebSocket session."""
        ...

    def run_streaming(
        self,
        token_queue: "queue.Queue",
        mic_queue: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        """Long-running thread loop. Pulls audio frames from ``mic_queue``
        and pushes ``(token_id, transformer_out_cpu, audio_pcm_np,
        arrival_ts)`` tuples onto ``token_queue`` at 12.5 Hz."""
        ...

    def get_latest_text(self) -> str:
        """Best-effort decoded text piece from the engine's text stream.
        Empty string if not available."""
        ...
