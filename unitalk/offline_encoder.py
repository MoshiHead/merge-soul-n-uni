"""Offline WAV → Moshi ``transformer_out`` token encoder.

Converts any audio file into the 4096-D Moshi token stream that
``MoshiToWav2VecAdapter`` expects, by running audio frame-by-frame through
Moshi's Mimi codec and LM in teacher-forcing mode.

Why teacher-forcing?
    In the live streaming pipeline, Moshi processes mic audio and generates a
    speech response; the adapter maps the LM's internal ``transformer_out``
    (4096-D per 80 ms frame) to face-animation features.  For offline inference
    from a pre-existing WAV file, we feed the audio as if it were the incoming
    speech and capture the same ``transformer_out`` at each step.  The adapter
    was trained on these representations, so offline and live behaviour are
    matched.

Typical use::

    enc = WavToMoshiTokens(moshi_pkg="moshi/moshi",
                           hf_repo="kyutai/moshiko-pytorch-bf16",
                           device="cuda")
    enc.load()                            # downloads ~14 GB on first run
    tokens = enc.encode("speech.wav")    # → [N, 4096] float32 on CPU

The returned tensor feeds directly into
:func:`unitalk.streaming.get_audio_embedding_from_tokens`.
"""
from __future__ import annotations

import torch

from .settings import MOSHI_SR, MOSHI_TOKEN_RATE


class WavToMoshiTokens:
    """Encode an audio file to Moshi ``transformer_out`` tokens at 12.5 Hz.

    Args:
        moshi_pkg:  Absolute (or CWD-relative) path to the Moshi inner package
                    directory (the folder that contains the ``moshi/`` Python
                    package, i.e. the one you'd put on ``sys.path``).
                    Typically ``<project_root>/moshi/moshi``.
        hf_repo:    HuggingFace repo ID for the Moshi checkpoint.
                    Defaults to ``"kyutai/moshiko-pytorch-bf16"``.
        device:     PyTorch device string (``"cuda"``, ``"cuda:0"``, etc.).
    """

    def __init__(
        self,
        moshi_pkg: str,
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        device: str = "cuda",
    ):
        self.moshi_pkg = moshi_pkg
        self.hf_repo = hf_repo
        self.device = device

        self.mimi = None
        self.lm_gen = None
        self.frame_size: int | None = None
        self.loaded = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load Mimi codec + Moshi LM (bf16).

        The first call downloads and caches the checkpoint (~14 GB for bf16).
        Subsequent calls are instant (already cached by HuggingFace).
        """
        if self.loaded:
            return

        from .utils import ensure_moshi_imports
        m = ensure_moshi_imports(self.moshi_pkg)
        LMGen, CheckpointInfo = m["LMGen"], m["CheckpointInfo"]

        print("[WavEncoder] Loading Moshi checkpoint info from HF hub...")
        info = CheckpointInfo.from_hf_repo(self.hf_repo)

        print("[WavEncoder] Loading Mimi codec...")
        self.mimi = info.get_mimi(device=self.device)

        print("[WavEncoder] Loading Moshi LM (bf16) — this may take ~1 min the first time...")
        lm = info.get_moshi(device=self.device, dtype=torch.bfloat16)
        lm.eval()

        self.lm_gen = LMGen(lm)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        # Enter streaming context once for the lifetime of this encoder.
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        self.loaded = True
        print(
            f"[WavEncoder] Ready.  "
            f"frame_size={self.frame_size} samples "
            f"@ {self.mimi.sample_rate} Hz  "
            f"({self.mimi.frame_rate:.1f} Hz tokens)"
        )

    # ── Encoding ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, wav_path: str) -> torch.Tensor:
        """Encode an audio file to ``[N, 4096]`` Moshi tokens.

        The audio is silently resampled to Moshi's 24 kHz sample rate.
        The output is float32 on CPU (move to GPU in the caller if needed).

        Args:
            wav_path: Path to the audio file (WAV, MP3, FLAC, OGG, …).

        Returns:
            ``[N, 4096]`` float32 on CPU, where N ≈ duration_seconds × 12.5.

        Raises:
            RuntimeError: If no tokens could be extracted (audio too short).
        """
        assert self.loaded, "Call load() first"

        pcm = self._load_and_resample(wav_path)   # [1, T] float32 on device
        self._reset_state()

        tokens_out: list[torch.Tensor] = []
        n_frames = pcm.shape[1] // self.frame_size
        first_frame = True

        for i in range(n_frames):
            chunk = pcm[:, i * self.frame_size: (i + 1) * self.frame_size]
            chunk = chunk.unsqueeze(0)             # [1, 1, frame_size]

            codes = self.mimi.encode(chunk)        # [1, n_codebooks, 1]

            if first_frame:
                # The LM needs one warm-up step before it starts emitting
                # transformer_out — matches the behaviour in MoshiEngine.
                self.lm_gen._step(codes)
                first_frame = False
                continue

            result = self.lm_gen._step(codes)
            if result is None:
                continue

            _, t_out = result                      # t_out: [1, 1, 4096]
            tokens_out.append(t_out[0, 0].float().cpu())

        if not tokens_out:
            raise RuntimeError(
                f"[WavEncoder] No tokens extracted from {wav_path!r}. "
                "The audio may be too short (< 160 ms)."
            )

        tokens = torch.stack(tokens_out, dim=0)   # [N, 4096]
        duration_s = tokens.shape[0] / MOSHI_TOKEN_RATE
        print(
            f"[WavEncoder] {wav_path!r}: "
            f"{n_frames} codec frames → {tokens.shape[0]} tokens "
            f"({duration_s:.2f} s)"
        )
        return tokens

    # ── Private helpers ────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Reset streaming state for a fresh encoding pass."""
        for obj in (self.mimi, self.lm_gen):
            if obj is None:
                continue
            try:
                obj.reset_streaming()
            except Exception:
                pass

    def _load_and_resample(self, wav_path: str) -> torch.Tensor:
        """Return ``[1, T]`` float32 at MOSHI_SR on self.device."""
        # Try torchaudio first (fastest); fall back to librosa.
        try:
            import torchaudio
            pcm, sr = torchaudio.load(wav_path)
        except Exception:
            import librosa
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            import numpy as np
            pcm = torch.from_numpy(y.astype("float32")).unsqueeze(0)

        # Collapse to mono float32.
        pcm = pcm.float().mean(0, keepdim=True)   # [1, T]

        # Resample to Moshi's native 24 kHz if needed.
        sr = int(sr)
        if sr != MOSHI_SR:
            try:
                import torchaudio.functional as TF
                pcm = TF.resample(pcm, sr, MOSHI_SR)
            except Exception:
                from .utils import resample_mono_f32
                import numpy as np
                arr = resample_mono_f32(pcm.squeeze(0).numpy(), sr, MOSHI_SR)
                pcm = torch.from_numpy(arr).unsqueeze(0)

        return pcm.to(self.device)
