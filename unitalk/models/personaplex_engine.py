"""PersonaPlex S2S engine — NVIDIA's persona/voice-controlled Moshi finetune.

PersonaPlex (https://github.com/NVIDIA/personaplex) is a finetune of Moshi
Helium that adds two conditioning hooks before the chat loop starts:

  * a text persona (``"You are a helpful billing agent for ACME ISP"``),
    tokenized and prefilled at startup so every reply stays in-character.
  * a voice clone — either a pre-baked ``.pt`` embedding shipped in the HF
    repo (NATF/NATM/VARF/VARM), or a fresh ``.wav`` that the engine
    LUFS-normalizes and mimi-encodes during prefill.

The 4096-dim Helium ``transformer_out`` is preserved (same architecture
as Moshi), so the downstream ``MoshiToWav2VecAdapter`` consumes it
identically. PersonaPlex's ``LMGen.step()`` does *not* return that hidden
state, so this module subclasses ``LMGen`` and captures it via the
``process_transformer_output`` hook — a 4-line, non-invasive patch.

Public API mirrors ``MoshiEngine``:
  * ``load()``
  * ``reset_streaming_state()``
  * ``run_streaming(token_queue, mic_queue, stop_event)``
  * ``get_latest_text()``
"""

from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import nullcontext

import numpy as np
import torch

from ..settings import PersonaPlexSettings
from ..utils import ensure_personaplex_imports, put_latest

_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


class PersonaPlexEngine:
    """PersonaPlex full-duplex S2S engine, wire-compatible with MoshiEngine.

    Two halves:
      * ``load()`` — pulls weights from HF, builds Mimi + LM, runs the
        persona + voice prefill once. Slow (~30 s cold).
      * ``run_streaming()`` — long-running thread that consumes mic
        audio and pushes ``(token_id, transformer_out, audio_pcm,
        arrival_ts)`` tuples onto a queue at 12.5 Hz.
    """

    def __init__(self, settings: PersonaPlexSettings, pplex_pkg: str, device: str):
        self.settings = settings
        self.pplex_pkg = pplex_pkg

        preset = settings.presets.get(settings.precision)
        if preset is None:
            # PersonaPlex only ships a bf16 model today; allow override via repo.
            if settings.repo_override:
                preset = None
            else:
                raise ValueError(
                    f"Unknown personaplex precision: {settings.precision}. "
                    f"Available presets: {list(settings.presets.keys())}"
                )

        self.precision = settings.precision
        self.hf_repo = (
            settings.repo_override
            or (preset.repo if preset else "nvidia/personaplex-7b-v1")
        )
        self.dtype = _DTYPE_BY_NAME[preset.dtype] if preset else torch.bfloat16
        self.device = device

        self.mimi = None
        self.lm = None
        self.lm_gen = None
        self.frame_size: int | None = None
        self.text_tokenizer = None
        self.loaded = False

        self._text_lock = threading.Lock()
        self._latest_text = ""

        # Track whether we've handed the engine to a session yet. ``load()``
        # already runs the (slow) persona + voice prefill and leaves the
        # streaming context in a clean state — calling
        # ``reset_streaming_state`` for session #1 would re-do that work
        # for no reason and add ~1-2s of perceived latency between the
        # user clicking "Start" and the avatar talking. We skip it
        # exactly once, then fall back to the normal reset+re-prefill
        # path for sessions 2+.
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
        """Decode the text token at ``tokens[0, 0, 0]``. Skips PersonaPlex's
        zero_text_code (3) and silence (0), matching offline.py's filter."""
        tok = self.text_tokenizer
        if tok is None or tokens is None:
            return ""
        try:
            token_id = int(tokens[0, 0, 0].item())
        except Exception:
            return ""
        if token_id in (0, 3):
            return ""

        try:
            piece = tok.id_to_piece(token_id)
        except Exception:
            return ""
        if not piece:
            return ""
        piece = piece.replace("▁", " ").strip()
        return piece if piece and piece.lower() != "<pad>" else ""

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        if self.loaded:
            return

        m = ensure_personaplex_imports(self.pplex_pkg)
        LMGen = m["LMGen"]
        get_mimi = m["get_mimi"]
        get_moshi_lm = m["get_moshi_lm"]
        loaders_mod = m["loaders"]
        SentencePieceProcessor = m["SentencePieceProcessor"]
        hf_hub_download = m["hf_hub_download"]

        # ── Build the transformer_out-capturing LMGen subclass ───────
        # We only need the hook, not the full class hierarchy. Done at
        # load() time (not module top-level) because LMGen is imported
        # lazily and the parent class isn't known until then.
        class _PPLMGenWithHiddenState(LMGen):
            """Captures transformer_out so MoshiToWav2VecAdapter can read it."""

            last_transformer_out: "torch.Tensor | None" = None

            @torch.no_grad()
            def process_transformer_output(self, transformer_out, *args, **kwargs):
                # ``transformer_out`` is [B, 1, 4096] — clone immediately
                # because PersonaPlex re-uses the buffer across steps.
                self.last_transformer_out = transformer_out
                return super().process_transformer_output(
                    transformer_out, *args, **kwargs
                )

        # ── Download weights from HuggingFace (gated; needs HF_TOKEN) ─
        print(f"[PersonaPlex] Resolving weights from {self.hf_repo}...")
        try:
            mimi_path = hf_hub_download(self.hf_repo, loaders_mod.MIMI_NAME)
            moshi_path = hf_hub_download(self.hf_repo, loaders_mod.MOSHI_NAME)
            tokenizer_path = hf_hub_download(self.hf_repo, loaders_mod.TEXT_TOKENIZER_NAME)
        except Exception as e:
            raise RuntimeError(
                f"PersonaPlex weight download failed from {self.hf_repo}. "
                f"Did you (a) accept the model license at "
                f"https://huggingface.co/{self.hf_repo} and (b) set HF_TOKEN? "
                f"Underlying error: {e}"
            ) from e

        print(f"[PersonaPlex] Loading Mimi codec...")
        self.mimi = get_mimi(mimi_path, device=self.device)

        print(f"[PersonaPlex] Loading LM ({self.precision}, {self.dtype})...")
        self.lm = get_moshi_lm(moshi_path, device=self.device, dtype=self.dtype)
        self.lm.eval()

        self.text_tokenizer = SentencePieceProcessor(tokenizer_path)
        print(f"[PersonaPlex] Text tokenizer loaded.")

        self.lm_gen = _PPLMGenWithHiddenState(
            self.lm,
            device=self.device,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            frame_rate=self.mimi.frame_rate,
        )
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        # ── Enter streaming ONCE for the process lifetime ───────────
        # Mirrors MoshiEngine.load() — re-entering streaming_forever
        # mid-process interacts badly with CUDAGraph captures.
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        print("[PersonaPlex] Streaming context entered (one-time).")

        # ── Apply voice prompt (.pt embeddings OR .wav clone) ────────
        if self.settings.voice_prompt:
            vp = self._resolve_voice_prompt(self.settings.voice_prompt)
            if vp.endswith(".pt"):
                print(f"[PersonaPlex] Loading voice prompt embeddings: {vp}")
                self.lm_gen.load_voice_prompt_embeddings(vp)
            else:
                print(f"[PersonaPlex] Loading voice prompt audio: {vp}")
                self.lm_gen.load_voice_prompt(vp)
        else:
            print("[PersonaPlex] No voice prompt configured — default voice.")

        # ── Apply text persona prompt ────────────────────────────────
        # NOTE: step_system_prompts() iterates over text_prompt_tokens
        # unconditionally. We use an empty list (not None) when no
        # persona is configured so the iteration is a clean no-op
        # instead of a TypeError.
        if self.settings.text_prompt:
            wrapped = self._wrap_with_system_tags(self.settings.text_prompt)
            self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrapped)
            print(f"[PersonaPlex] Persona prompt set ({len(self.lm_gen.text_prompt_tokens)} tokens).")
        else:
            self.lm_gen.text_prompt_tokens = []
            print("[PersonaPlex] No persona prompt — using empty list (default behavior).")

        # ── Prime the model with voice + persona ─────────────────────
        # mimi gets reset twice (once before, once after voice-prompt
        # encoding) per the canonical offline.py recipe — otherwise the
        # mimi state holds the voice prompt audio and the chat loop
        # starts from the wrong position.
        print("[PersonaPlex] Running system prompt prefill...")
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()
        print("[PersonaPlex] Prefill complete.")

        self.loaded = True
        print(
            f"[PersonaPlex] Ready. dim={self.lm.dim}, dep_q={self.lm.dep_q}, "
            f"frame_size={self.frame_size}"
        )

    @staticmethod
    def _wrap_with_system_tags(text: str) -> str:
        """Mirror PersonaPlex's offline.py wrapper — idempotent."""
        cleaned = text.strip()
        if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
            return cleaned
        return f"<system> {cleaned} <system>"

    # Voice prompt filenames bundled inside upstream voices.tgz on HF.
    # Used to decide whether a missing path can be auto-fetched.
    _BUNDLED_VOICE_PREFIXES = ("NATF", "NATM", "VARF", "VARM")

    # Canonical voice location, relative to the UniTalk project root.
    # Top-level so we never tangle with the vendored personaplex tree.
    _VOICES_SUBDIR = "assets/audio_prompts"

    def _resolve_voice_prompt(self, path: str) -> str:
        """Validate the voice-prompt path; auto-fetch bundled voices.

        Behaviour:
          * If the file exists at ``path``, return it as-is.
          * Otherwise, if ``path`` points at a missing bundled voice
            inside the canonical ``<UniTalk>/assets/audio_prompts/``
            directory, download + extract ``voices.tgz`` and retry.
          * Anything else → ``FileNotFoundError`` with a clear message.

        No bare-name shortcut, no legacy-path fallback — pass a full
        path (absolute or project-relative; the settings layer already
        resolves the relative-vs-absolute distinction).
        """
        if os.path.isfile(path):
            return path

        # Auto-fetch only when the missing file looks like a bundled
        # voice AND it's in the canonical voices dir.
        from ..settings import PROJECT_ROOT
        canonical_dir = os.path.join(PROJECT_ROOT, self._VOICES_SUBDIR)
        basename = os.path.basename(path)
        in_canonical_dir = os.path.abspath(os.path.dirname(path)) == \
            os.path.abspath(canonical_dir)
        is_bundled = any(basename.startswith(p) for p in self._BUNDLED_VOICE_PREFIXES)

        if in_canonical_dir and is_bundled:
            self._ensure_voices_dir(canonical_dir)
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f"Voice prompt not found: {path!r}.\n"
            f"For bundled voices (NATF*/NATM*/VARF*/VARM*), place them at "
            f"{canonical_dir}/<NAME>.pt — or let the engine auto-download "
            f"there on first launch (requires HF_TOKEN + accepted "
            f"https://huggingface.co/{self.hf_repo} license)."
        )

    def _ensure_voices_dir(self, target_dir: str) -> str:
        """Download + extract voices.tgz from HF into ``target_dir``.

        ``voices.tgz`` unpacks as ``voices/<NAME>.pt``; we flatten the
        inner ``voices/`` wrapper so files end up directly under
        ``target_dir``. Idempotent — returns early if any bundled
        voice file is already present.
        """
        os.makedirs(target_dir, exist_ok=True)

        # Already populated? Look for any bundled voice file as evidence.
        if any(
            f.endswith(".pt")
            and any(f.startswith(p) for p in self._BUNDLED_VOICE_PREFIXES)
            for f in os.listdir(target_dir)
        ):
            return target_dir

        m = ensure_personaplex_imports(self.pplex_pkg)
        hf_hub_download = m["hf_hub_download"]
        print(f"[PersonaPlex] No bundled voices in {target_dir} — "
              f"downloading voices.tgz from {self.hf_repo}...")
        try:
            voices_tgz = hf_hub_download(self.hf_repo, "voices.tgz")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download voices.tgz from {self.hf_repo}. "
                f"Confirm HF_TOKEN is set and the model license has been "
                f"accepted at https://huggingface.co/{self.hf_repo}. "
                f"Underlying error: {e}"
            ) from e

        import tarfile
        print(f"[PersonaPlex] Extracting voices.tgz to {target_dir}/...")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=target_dir)

        # voices.tgz unpacks as voices/<NAME>.pt — flatten so files sit
        # directly under target_dir.
        nested = os.path.join(target_dir, "voices")
        if os.path.isdir(nested):
            for entry in os.listdir(nested):
                src = os.path.join(nested, entry)
                dst = os.path.join(target_dir, entry)
                if not os.path.exists(dst):
                    os.rename(src, dst)
            # Best-effort cleanup of the now-empty wrapper dir.
            try:
                os.rmdir(nested)
            except OSError:
                pass

        return target_dir

    def reset_streaming_state(self) -> None:
        """Reset per-session streaming state and re-prime the persona.

        PersonaPlex's persona/voice conditioning lives in the streaming
        state. After a reset the model forgets the persona, so we must
        re-run ``step_system_prompts`` to re-prime. The double-reset of
        mimi mirrors the canonical offline.py flow.

        First-session skip: ``load()`` already left the engine in a
        clean post-prefill state, so the very first session can re-use
        that without paying the prefill cost again. Sessions 2+ go
        through the full reset+re-prefill cycle.
        """
        if self._first_session:
            self._first_session = False
            print("[PersonaPlex] First session — reusing post-load state (skip re-prefill).")
            return
        try:
            if self.mimi is not None:
                self.mimi.reset_streaming()
            if self.lm_gen is not None:
                self.lm_gen.reset_streaming()
                # Re-prime persona for the new session.
                if self.mimi is not None:
                    self.lm_gen.step_system_prompts(self.mimi)
                    self.mimi.reset_streaming()
        except Exception as e:
            print(f"[PersonaPlex] reset_streaming_state warning: {e}")

    # ── Streaming loop (runs in a dedicated thread) ────────────────────

    @torch.no_grad()
    def run_streaming(
        self,
        token_queue: "queue.Queue",
        mic_queue: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        """Main PersonaPlex streaming loop.

        Same wire format as MoshiEngine — produces
        ``(token_id, transformer_out_cpu, audio_pcm_np, arrival_ts)``
        tuples at 12.5 Hz, exactly what the FlashHead bridge expects.

        Two PersonaPlex specifics handled here:
          * ``lm_gen.step()`` returns tokens only — we read
            ``lm_gen.last_transformer_out`` (populated by our subclass
            patch) right after.
          * Audio decode uses ``tokens[:, 1:9]`` (the 8 agent audio
            codebooks), matching offline.py's ``decode_tokens_to_pcm``.
        """
        assert self.loaded, "Call load() first"
        print("[PersonaPlexThread] Starting streaming...")

        is_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
        cuda_stream = torch.cuda.Stream(device=self.device) if is_cuda else None
        step = 0
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
                    # mimi.encode returns [B, K, T] — usually T == 1 for
                    # a frame_size chunk, but step() requires T==1 so we
                    # iterate to stay safe.
                    n_steps = codes.shape[-1]
                    last_tokens = None
                    last_transformer_out = None
                    for c in range(n_steps):
                        last_tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                        # Snapshot the captured hidden state IMMEDIATELY —
                        # the next step() call overwrites it.
                        if self.lm_gen.last_transformer_out is not None:
                            last_transformer_out = self.lm_gen.last_transformer_out

                    if last_tokens is None or last_transformer_out is None:
                        step += 1
                        del codes, pcm
                        continue

                    token_id += 1
                    t_out_cpu = last_transformer_out.detach().cpu()  # [1, 1, 4096]
                    arrival_ts = time.perf_counter()

                    audio_pcm = None
                    if self.lm.dep_q > 0:
                        # PersonaPlex hard-codes channels 1..8 for agent
                        # audio (vs vanilla Moshi's 1: open slice). The
                        # 8 here matches the model's dep_q exactly.
                        out_pcm = self.mimi.decode(last_tokens[:, 1:1 + self.lm.dep_q])
                        audio_pcm = out_pcm[0, 0].detach().cpu().numpy()
                        del out_pcm

                    text_piece = self._decode_text_piece(last_tokens)
                    if text_piece:
                        self._set_latest_text(text_piece)

                    put_latest(
                        token_queue,
                        (token_id, t_out_cpu, audio_pcm, arrival_ts),
                    )

                    del last_tokens, last_transformer_out, codes, pcm, t_out_cpu

                step += 1
                if step % 500 == 0:
                    print(
                        f"  [PersonaPlexThread] step={step}, token_id={token_id}, "
                        f"token_q={token_queue.qsize()}"
                    )

        except Exception as e:
            print(f"[PersonaPlexThread] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[PersonaPlexThread] Exiting.")
