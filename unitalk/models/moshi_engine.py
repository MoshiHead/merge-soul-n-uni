"""Moshi Helium S2S engine — owns the LM, Mimi codec, and streaming loop."""

from __future__ import annotations

import queue
import threading
import time
from contextlib import nullcontext

import numpy as np
import torch

from ..settings import MoshiSettings
from ..utils import ensure_moshi_imports, put_latest

_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


class MoshiEngine:
    """Wraps the Moshi LM + Mimi codec and produces tokens at 12.5 Hz.

    Two halves:
      * ``load()`` — one-time model construction (slow).
      * ``run_streaming()`` — long-running thread that consumes mic audio
        and pushes ``(token_id, transformer_out, audio_pcm, arrival_ts)``
        tuples onto a queue at 12.5 Hz.
    """

    def __init__(self, settings: MoshiSettings, moshi_pkg: str, device: str):
        self.settings = settings
        self.moshi_pkg = moshi_pkg

        preset = settings.presets.get(settings.precision)
        if preset is None:
            raise ValueError(f"Unknown moshi precision: {settings.precision}")

        self.precision = settings.precision
        self.hf_repo = settings.repo_override or preset.repo
        self.dtype = _DTYPE_BY_NAME[preset.dtype]
        self.device = device

        self.mimi = None
        self.lm = None
        self.lm_gen = None
        self.frame_size: int | None = None
        self.text_tokenizer = None
        self.loaded = False

        self._text_lock = threading.Lock()
        self._latest_text = ""

        # ``load()`` opens the streaming context once and leaves the
        # engine in a clean state. Resetting again on the first session
        # is redundant work — skip it exactly once. Sessions 2+ still
        # reset normally to wipe inter-conversation state.
        self._first_session = True

    # ── Text passthrough (best-effort) ─────────────────────────────────

    def get_latest_text(self) -> str:
        with self._text_lock:
            return self._latest_text

    def _set_latest_text(self, text: str) -> None:
        if not text:
            return
        with self._text_lock:
            self._latest_text = text

    def _decode_text_piece(self, tokens: torch.Tensor) -> str:
        """Best-effort text-token decoding from Moshi output."""
        tok = self.text_tokenizer
        if tok is None or tokens is None:
            return ""
        try:
            token_id = int(tokens[0, 0, 0].item())
        except Exception:
            return ""
        if token_id <= 0:
            return ""

        for fn in (
            lambda: tok.decode([token_id]),
            lambda: tok.decode(token_id),
            lambda: tok.id_to_piece(token_id),
            lambda: tok.convert_ids_to_tokens([token_id])[0],
        ):
            try:
                piece = fn()
                if piece is not None:
                    piece = str(piece).strip()
                    if piece and piece.lower() != "<pad>":
                        return piece
            except Exception:
                continue
        return ""

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        if self.loaded:
            return

        m = ensure_moshi_imports(self.moshi_pkg)
        LMGen, CheckpointInfo = m["LMGen"], m["CheckpointInfo"]

        print("[Moshi] Loading checkpoint info...")
        info = CheckpointInfo.from_hf_repo(self.hf_repo)

        print("[Moshi] Loading Mimi codec...")
        self.mimi = info.get_mimi(device=self.device)

        print(f"[Moshi] Loading LM ({self.precision}, {self.dtype})...")
        self.lm = info.get_moshi(device=self.device, dtype=self.dtype)
        self.lm.eval()

        if self.precision == "q8":
            fixed = 0
            for module in self.lm.modules():
                if hasattr(module, "weight_scb") and module.weight_scb.dtype != torch.float32:
                    module.weight_scb.data = module.weight_scb.data.float()
                    fixed += 1
            if fixed:
                print(f"[Moshi] Fixed {fixed} QLinear scale buffers → float32")

        self.lm_gen = LMGen(self.lm)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        try:
            self.text_tokenizer = info.get_text_tokenizer()
            print("[Moshi] Text tokenizer loaded.")
        except Exception as e:
            print(f"[Moshi] Text tokenizer not available: {e}")
            self.text_tokenizer = None

        # Enter the streaming context ONCE for the lifetime of the process.
        # This is what the official moshi server (moshi/moshi/moshi/server.py)
        # does: streaming_forever in __init__, then reset_streaming() per
        # session. Previously we re-called streaming_forever per session
        # which interacts badly with the CUDAGraph captures held inside
        # _LMGenState — re-entering streaming creates new CUDAGraphed
        # wrappers but stale graphed_depth state can leak across sessions
        # and trip `assert not lm_model.depformer.is_streaming` on the
        # first depformer_step of session 2.
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        print("[Moshi] Streaming context entered (one-time).")

        self.loaded = True
        print(
            f"[Moshi] Ready. dim={self.lm.dim}, dep_q={self.lm.dep_q}, "
            f"frame_size={self.frame_size}"
        )

    def reset_streaming_state(self) -> None:
        """Reset per-session streaming state (cache, offsets, exec_mask).
        The streaming CONTEXT itself stays open — only state VALUES
        are reset, matching the official moshi server's per-session
        ``reset_streaming()`` call.

        First-session skip: ``load()`` already opened the streaming
        context with a clean state, so the first session re-uses it
        without paying redundant work. Sessions 2+ reset normally.
        """
        if self._first_session:
            self._first_session = False
            print("[Moshi] First session — reusing post-load state.")
            return
        try:
            if self.mimi is not None:
                self.mimi.reset_streaming()
        except Exception as e:
            print(f"[Moshi] mimi.reset_streaming warning: {e}")
        try:
            if self.lm_gen is not None:
                self.lm_gen.reset_streaming()
        except Exception as e:
            print(f"[Moshi] lm_gen.reset_streaming warning: {e}")

    # ── Streaming loop (runs in a dedicated thread) ────────────────────

    @torch.no_grad()
    def run_streaming(
        self,
        token_queue: "queue.Queue",
        mic_queue: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        """Main Moshi streaming loop.

        Produces ``(token_id, transformer_out_cpu, audio_pcm_np, arrival_ts)``
        tuples and pushes them to ``token_queue`` at 12.5 Hz. Each token
        corresponds to 80 ms of audio and maps to exactly 2 video frames
        after the 100 → 200 interpolation.
        """
        assert self.loaded, "Call load() first"
        print("[MoshiThread] Starting streaming...")

        # NOTE: streaming_forever was called once in load(); we do NOT
        # re-enter it here. The per-session reset is expected to have
        # happened in session.run() before this thread starts.

        # Match plain "cuda" AND any "cuda:N" so multi-GPU runs still get
        # the streaming optimization (was previously only "cuda").
        is_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
        cuda_stream = torch.cuda.Stream(device=self.device) if is_cuda else None
        step = 0
        first_frame = True
        token_id = 0

        try:
            while not stop_event.is_set():
                try:
                    mic_chunk = mic_queue.get(timeout=0.2)
                except queue.Empty:
                    mic_chunk = np.zeros(self.frame_size, dtype=np.float32)

                pcm = (
                    torch.from_numpy(mic_chunk)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                )

                ctx = torch.cuda.stream(cuda_stream) if cuda_stream else nullcontext()
                with ctx:
                    codes = self.mimi.encode(pcm)

                    if first_frame:
                        self.lm_gen._step(codes)
                        first_frame = False
                        step += 1
                        del codes, pcm
                        continue

                    result = self.lm_gen._step(codes)
                    if result is None:
                        step += 1
                        del codes, pcm
                        continue

                    tokens, transformer_out = result

                    # The 4096-dim transformer_out is what the token bridge consumes.
                    token_id += 1
                    t_out_cpu = transformer_out.detach().cpu()  # [1, 1, 4096]
                    arrival_ts = time.perf_counter()

                    audio_pcm = None
                    if self.lm.dep_q > 0:
                        out_pcm = self.mimi.decode(tokens[:, 1:])
                        audio_pcm = out_pcm[0, 0].detach().cpu().numpy()
                        del out_pcm

                    text_piece = self._decode_text_piece(tokens)
                    if text_piece:
                        self._set_latest_text(text_piece)

                    put_latest(
                        token_queue,
                        (token_id, t_out_cpu, audio_pcm, arrival_ts),
                    )

                    del tokens, transformer_out, codes, pcm, t_out_cpu

                step += 1
                if step % 500 == 0:
                    print(
                        f"  [MoshiThread] step={step}, token_id={token_id}, "
                        f"token_q={token_queue.qsize()}"
                    )

        except Exception as e:
            print(f"[MoshiThread] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[MoshiThread] Exiting.")
