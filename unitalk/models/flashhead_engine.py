"""FlashHead engine — frozen SoulX-FlashHead avatar pipeline driven by tokens.

Pipeline::

    Adapter output (1, 33, 5, 12, 768)
    → AudioProjModel (FROZEN): (1, 33, 5, 12, 768) → (1, 9, 32, 1536)
    → DiT denoising (FROZEN): 4 steps with audio cross-attention
    → VAE Decode (FROZEN): latent → 33 pixel frames (512×512)
    → Motion frame carry-over (FROZEN): last N frames → encode → next chunk
    → Discard motion overlap → 24 (lite) / 28 (pro) new frames

All frozen components come from standard SoulX-FlashHead checkpoints.
Only the adapter is loaded from the path configured in ``inference.yaml``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from ..distributed import (
    CMD_RESET_REF,
    CMD_RUN_PIPELINE,
    broadcast_cmd,
    broadcast_tensor,
    get_dist_context,
)
from ..settings import (
    DEQUE_SIZE,
    INTERP_TARGET,
    MOSHI_DIM,
    WAV2VEC_DIM,
    WAV2VEC_LAYERS,
)
from ..utils import ensure_flashhead_imports
from .adapter import MoshiToWav2VecAdapter


class FlashHeadTokenEngine:
    """Runs the frozen SoulX-FlashHead pipeline with our adapter as the front-end."""

    def __init__(
        self,
        soulx_root: str,
        ckpt_dir: str,
        wav2vec_dir: str,
        model_type: str,
        ref_image: str,
        adapter_ckpt_path: str,
        base_seed: int = 42,
        device: str = "cuda",
    ):
        self.soulx_root = os.path.abspath(soulx_root)
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.wav2vec_dir = os.path.abspath(wav2vec_dir)
        self.model_type = model_type
        self.ref_image = os.path.abspath(ref_image)
        self.adapter_ckpt_path = os.path.abspath(adapter_ckpt_path)
        self.base_seed = int(base_seed)
        self.device = device

        # Lazily populated by load().
        self.pipeline = None
        self.adapter: MoshiToWav2VecAdapter | None = None
        self.infer_params = None
        self.frame_num: int | None = None
        self.motion_frames_num: int | None = None
        self.slice_len: int | None = None
        self.tokens_per_chunk: int | None = None
        self.tgt_fps: int | None = None
        self._chunk_idx = 0

        # Audio-embedding shape, frozen at load time. Workers pre-allocate
        # against this so they can receive the broadcast without first
        # having to learn the shape from rank 0.
        self._embedding_shape: tuple[int, ...] | None = None
        self._embedding_dtype: torch.dtype | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        if self.pipeline is not None:
            return

        fh = ensure_flashhead_imports(self.soulx_root)

        # Read the distributed layout. Single-GPU runs => world_size=1
        # and SoulX takes the non-USP path. Under torchrun world_size
        # equals --nproc_per_node, and SoulX's get_pipeline calls
        # dist.init_process_group + initializes xfuser sequence-parallel
        # for us. We do NOT init the process group ourselves.
        ctx = get_dist_context()
        world_size = ctx.world_size

        print(
            f"[FlashHead] Loading pipeline... "
            f"(rank={ctx.rank}/{ctx.world_size}, device={self.device})"
        )
        self.pipeline = fh["get_pipeline"](
            world_size=world_size,
            ckpt_dir=self.ckpt_dir,
            wav2vec_dir=self.wav2vec_dir,
            model_type=self.model_type,
        )
        self.infer_params = fh["get_infer_params"]()
        self.frame_num = int(self.infer_params["frame_num"])             # 33
        self.motion_frames_num = int(self.infer_params["motion_frames_num"])
        self.tgt_fps = int(self.infer_params["tgt_fps"])                 # 25
        self.slice_len = self.frame_num - self.motion_frames_num

        # Moshi tokens needed per chunk = slice_len frames × 40ms / 80ms per token.
        # Lite (vae_stride[0]=8): motion=9, slice=24 → 12 tokens (960 ms).
        # Pro  (vae_stride[0]=4): motion=5, slice=28 → 14 tokens (1120 ms).
        self.tokens_per_chunk = self.slice_len // 2

        vae_stride_0 = self.pipeline.config.vae_stride[0]
        print(
            f"[FlashHead] model={self.model_type}, vae_stride[0]={vae_stride_0}, "
            f"frame_num={self.frame_num}, motion={self.motion_frames_num}, "
            f"slice_len={self.slice_len}, tokens_per_chunk={self.tokens_per_chunk}"
        )

        # The adapter is the only trainable component.
        self.adapter = MoshiToWav2VecAdapter(
            moshi_dim=MOSHI_DIM,
            hidden_dim=WAV2VEC_DIM,
            num_layers=WAV2VEC_LAYERS,
            num_heads=12,
            ffn_dim=3072,
        )
        loaded = self.adapter.load_checkpoint(self.adapter_ckpt_path)
        if not loaded:
            print(
                f"[Adapter] No checkpoint found at {self.adapter_ckpt_path}. "
                f"Using random initialization (neutral motion until fine-tuned)."
            )
        self.adapter = self.adapter.to(self.device).eval()

        # Audio embedding shape produced by get_audio_embedding_from_tokens:
        # [1, frame_num, 5, num_wav2vec_layers, wav2vec_dim]. Frozen here
        # so worker ranks know what tensor to allocate for the per-chunk
        # broadcast (they cannot query rank 0 — we'd deadlock).
        self._embedding_shape = (1, self.frame_num, 5, WAV2VEC_LAYERS, WAV2VEC_DIM)
        self._embedding_dtype = self.pipeline.param_dtype

        self.set_reference(self.ref_image)
        self._chunk_idx = 0

        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        total_pipe_params = sum(p.numel() for p in self.pipeline.model.parameters())
        print(
            f"[FlashHead] Ready. tokens_per_chunk={self.tokens_per_chunk}\n"
            f"  Adapter params (trainable):    {adapter_params:,} "
            f"({adapter_params * 4 / 1e6:.1f} MB @ fp32)\n"
            f"  DiT+AudioProj params (frozen): {total_pipe_params:,} "
            f"({total_pipe_params * 2 / 1e9:.2f} GB @ bf16)"
        )
        print(f"  Adapter checkpoint: {'LOADED ✓' if loaded else 'NOT FOUND (random init)'}")

    # ── Worker-side helpers ────────────────────────────────────────────

    @property
    def embedding_shape(self) -> tuple[int, ...]:
        """Shape of the audio embedding broadcast between ranks. Read-
        only; valid after :meth:`load`."""
        assert self._embedding_shape is not None, "call load() first"
        return self._embedding_shape

    @property
    def embedding_dtype(self) -> torch.dtype:
        assert self._embedding_dtype is not None, "call load() first"
        return self._embedding_dtype

    def _run_pipeline_lockstep(self, audio_emb: torch.Tensor):
        """Call SoulX's ``run_pipeline`` on every rank in lockstep.

        Single-GPU: thin no-broadcast pass-through.

        Multi-GPU: the broadcasts are SYMMETRIC. NCCL's ``dist.broadcast``
        with ``src=0`` requires every rank to participate — rank 0 sends,
        others receive. Calling it only on rank 0 deadlocks. Both
        ``warmup`` / ``render_idle_loop`` (where every rank calls this
        method directly) and chunk generation (where rank 0 calls this
        method and workers call the matching collectives from
        :func:`run_worker_loop`) need to see matching broadcasts on every
        rank — so the broadcasts here fire unconditionally under multi-
        GPU. On worker ranks the local ``audio_emb`` is overwritten
        in-place with rank 0's tensor before ``run_pipeline`` is called,
        keeping the xfuser sequence-parallel collective numerically
        identical across ranks.
        """
        fh = ensure_flashhead_imports(self.soulx_root)
        ctx = get_dist_context()
        if ctx.multi_gpu:
            broadcast_cmd(CMD_RUN_PIPELINE)
            broadcast_tensor(audio_emb)
        return fh["run_pipeline"](self.pipeline, audio_emb)

    def set_reference(self, ref_image_path: str) -> None:
        fh = ensure_flashhead_imports(self.soulx_root)
        ref_image_path = os.path.abspath(ref_image_path)
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
        self.ref_image = ref_image_path
        fh["get_base_data"](
            self.pipeline,
            cond_image_path_or_dir=self.ref_image,
            base_seed=self.base_seed,
            use_face_crop=False,
        )
        self.pipeline.reset_person_name(self.pipeline.person_name)
        self._chunk_idx = 0

    def reset_for_next_session(self) -> None:
        # Broadcast the reset to worker ranks so their per-rank pipeline
        # state (latent_motion_frames, person_name) zeroes out in lockstep
        # with rank 0. No-op on single-GPU runs.
        ctx = get_dist_context()
        if ctx.multi_gpu and ctx.is_rank0:
            broadcast_cmd(CMD_RESET_REF)
        self._chunk_idx = 0
        self.pipeline.reset_person_name(self.pipeline.person_name)

    # ── Inference ──────────────────────────────────────────────────────

    @torch.inference_mode()
    def warmup(
        self,
        min_chunks: int = 3,
        max_chunks: int = 12,
        target_ms_per_frame: float = 40.0,
        stability_tolerance: float = 0.15,
    ) -> None:
        """Adaptive warmup that exercises both silence and real-data paths.

        Strategy:
          1. First few chunks use a zero-token deque (silence / idle-loop
             path). This is fast — covers the trivial activation pattern.
          2. Remaining chunks use a Gaussian-random deque so the adapter
             forward + DiT cross-attention hit the same dense kernels as
             live conversation tokens.
          3. Exit when:
             * ``max_chunks`` reached (safety cap), OR
             * ``min_chunks`` done AND we're in the random phase AND the
               last two chunks are within ``stability_tolerance`` AND
               under ``target_ms_per_frame``.

        Why we changed from the old "one fast chunk → exit" rule: the old
        rule converged at chunk 2 with a zero deque (~3 ms/frame), but the
        FIRST real chunk in a live session was 200-320 ms — a 4-5×
        regression the user perceived as the avatar "starting slow". The
        random-token pass warms the same kernel path live tokens hit;
        consecutive-stability ensures we're truly at steady state.
        """
        # Local import avoids a top-level cycle (streaming imports this module).
        from ..streaming import get_audio_embedding_from_tokens

        print(
            f"[FlashHead] Warmup (adaptive): "
            f"min={min_chunks}, max={max_chunks}, "
            f"target ≤{target_ms_per_frame:.0f}ms/frame, "
            f"stability ±{int(stability_tolerance * 100)}%..."
        )

        # Split: at most 2 zero-deque chunks (idle path), the rest random.
        # We always want at least 1 random-pass chunk to exit on.
        zero_chunks = min(2, max(1, min_chunks // 2))

        silence_deque = torch.zeros(DEQUE_SIZE, MOSHI_DIM)
        rng = torch.Generator()
        rng.manual_seed(self.base_seed)

        prev_elapsed_ms: float | None = None
        ms_per_frame = float("inf")
        chunks_done = 0

        for i in range(max_chunks):
            if i < zero_chunks:
                deque_snap = silence_deque
                tag = "silence"
            else:
                # Gaussian std 0.5 keeps values in a reasonable dynamic
                # range — real Moshi/PersonaPlex transformer_out values
                # span roughly this scale in our captured training data.
                deque_snap = torch.randn(
                    DEQUE_SIZE, MOSHI_DIM, generator=rng
                ) * 0.5
                tag = "random"

            t0 = time.perf_counter()
            audio_emb = get_audio_embedding_from_tokens(
                self.adapter,
                deque_snap,
                self.frame_num,
                device=torch.device(self.device),
                dtype=self.pipeline.param_dtype,
            )
            _ = self._run_pipeline_lockstep(audio_emb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ms_per_frame = elapsed_ms / max(1, self.frame_num)
            chunks_done = i + 1

            delta_str = ""
            if prev_elapsed_ms is not None:
                delta_frac = abs(elapsed_ms - prev_elapsed_ms) / max(prev_elapsed_ms, 1.0)
                delta_str = f" Δ={delta_frac * 100:.1f}%"

            print(
                f"  [FlashHead] warmup chunk {chunks_done}/{max_chunks} ({tag}): "
                f"{elapsed_ms:.0f}ms total ({ms_per_frame:.1f}ms/frame){delta_str}"
            )

            # Exit only after: (a) min_chunks reached, (b) we're past the
            # zero-pass into the random pass, (c) last two chunks are
            # within tolerance AND under the target.
            past_zero_pass = i >= zero_chunks
            stable = (
                prev_elapsed_ms is not None
                and abs(elapsed_ms - prev_elapsed_ms) / max(prev_elapsed_ms, 1.0)
                <= stability_tolerance
            )
            if (
                chunks_done >= min_chunks
                and past_zero_pass
                and stable
                and ms_per_frame <= target_ms_per_frame
            ):
                print(
                    f"[FlashHead] Warmup converged at chunk {chunks_done} "
                    f"({ms_per_frame:.1f}ms/frame, stable within "
                    f"±{int(stability_tolerance * 100)}%)."
                )
                break

            prev_elapsed_ms = elapsed_ms
        else:
            print(
                f"[FlashHead] Warmup hit max_chunks={max_chunks} without "
                f"reaching stability. Last: {ms_per_frame:.1f}ms/frame. "
                f"Streaming will still work — consider raising max_chunks "
                f"or relaxing the target."
            )

        # Reset pipeline state after warmup so the first real chunk doesn't
        # carry over warmup motion frames.
        self.pipeline.reset_person_name(self.pipeline.person_name)
        self._chunk_idx = 0
        print("[FlashHead] Warmup complete.")

    @torch.inference_mode()
    def render_idle_loop(self) -> np.ndarray:
        """Render one chunk of "silent listening" frames using a zero
        audio context. Called once after warmup; the result is shipped
        to the client at handshake time and played in a loop while the
        user is mid-utterance (mute state) instead of freezing on the
        last drawn frame.

        Returns:
            ``np.ndarray`` of uint8 frames ``[slice_len, H, W, 3]``.
            For lite: 24 frames. For pro: 28 frames.
        """
        from ..streaming import get_audio_embedding_from_tokens

        silence_deque = torch.zeros(DEQUE_SIZE, MOSHI_DIM)

        audio_emb = get_audio_embedding_from_tokens(
            self.adapter,
            silence_deque,
            self.frame_num,
            device=torch.device(self.device),
            dtype=self.pipeline.param_dtype,
        )
        video = self._run_pipeline_lockstep(audio_emb)
        video = video[self.motion_frames_num:]

        # Reset pipeline state so the next real chunk doesn't carry
        # idle silence motion frames over into the live conversation.
        self.pipeline.reset_person_name(self.pipeline.person_name)
        self._chunk_idx = 0

        return video.detach().cpu().numpy().astype(np.uint8)

    @torch.inference_mode()
    def generate_chunk(self, deque_snapshot: torch.Tensor) -> np.ndarray:
        """Generate a video chunk from the current deque snapshot.

        Args:
            deque_snapshot: ``[DEQUE_SIZE, MOSHI_DIM]`` token buffer snapshot.

        Returns:
            ``np.ndarray`` of uint8 frames ``[N, H, W, 3]``. N = slice_len
            (24 lite / 28 pro) for stream mode.
        """
        from ..streaming import get_audio_embedding_from_tokens

        audio_emb = get_audio_embedding_from_tokens(
            self.adapter,
            deque_snapshot,
            self.frame_num,
            device=torch.device(self.device),
            dtype=self.pipeline.param_dtype,
        )

        video = self._run_pipeline_lockstep(audio_emb)
        # Stream mode: discard motion overlap frames (matches FlashHead default).
        video = video[self.motion_frames_num:]
        self._chunk_idx += 1
        return video.detach().cpu().numpy().astype(np.uint8)
