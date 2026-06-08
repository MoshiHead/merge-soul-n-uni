"""Command-line entry point for the UniTalk streaming server.

Examples
--------
Run with all defaults from ``unitalk/config/inference.yaml``::

    python lets_talk_with_unitalk.py

Override specific options::

    python lets_talk_with_unitalk.py --flash-model-type pro --port 8000

Pass an absolute checkpoint path (works from any cwd)::

    python lets_talk_with_unitalk.py \\
        --adapter-ckpt-path /workspace/UniTalk/checkpoints/adapter_phase2_latest_ep4.pt

Pass a relative path (must be run from the project root)::

    python lets_talk_with_unitalk.py --adapter-ckpt-path checkpoints/adapter_phase2_latest_ep4.pt

Override the served HTML page::

    python lets_talk_with_unitalk.py --static-index unitalk/static/flashtalk_head_rotate.html

Point at a different config file entirely::

    python lets_talk_with_unitalk.py --config path/to/my_config.yaml
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

# CUDA env tweaks must happen before torch is imported by sibling modules.
os.environ.setdefault("NO_CUDA_GRAPH", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import uvicorn  # noqa: E402

from .distributed import (  # noqa: E402
    get_dist_context,
    init_dist_context,
    run_worker_loop,
    shutdown_workers,
)
from .models import resolve_active_adapter_path  # noqa: E402
from .server import create_app, load_engines  # noqa: E402
from .settings import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Settings, load_settings, resolve_path  # noqa: E402
from .streaming import set_rotate_view  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UniTalk — Moshi + SoulX-FlashHead unified streaming server"
    )
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="Path to inference YAML config")

    # Optional overrides — only applied when explicitly passed.
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--ref-image", dest="ref_image", default=None,
                   help="Path to the reference image (abs or relative to project root)")
    p.add_argument("--base-seed", dest="base_seed", type=int, default=None)
    p.add_argument("--flash-model-type", dest="flash_model_type",
                   choices=["lite", "pro"], default=None)
    p.add_argument("--warmup-min-chunks", dest="warmup_min_chunks", type=int, default=None,
                   help="Minimum warmup chunks before adaptive exit can trigger")
    p.add_argument("--warmup-max-chunks", dest="warmup_max_chunks", type=int, default=None,
                   help="Hard cap on warmup chunks")
    p.add_argument("--warmup-target-ms-per-frame", dest="warmup_target_ms_per_frame",
                   type=float, default=None,
                   help="Stop warmup once the last chunk averages ≤ this many ms per frame")
    p.add_argument("--adapter-ckpt-path", dest="adapter_ckpt_path", default=None,
                   help="Full path to the adapter .pt file "
                        "(abs path, or relative to the project root)")
    p.add_argument("--static-index", dest="static_index", default=None,
                   help="Path to the frontend HTML served at / "
                        "(abs path, or relative to the project root)")
    p.add_argument("--moshi-precision", dest="moshi_precision",
                   choices=["q8", "bf16", "fp32"], default=None)
    p.add_argument("--moshi-repo", dest="moshi_repo", default=None)
    p.add_argument("--moshi-device", dest="moshi_device", default=None)
    p.add_argument("--flashhead-device", dest="flashhead_device", default=None)
    p.add_argument("--show-sync", dest="show_sync", action="store_true", default=None)

    # ── S2S engine selection + PersonaPlex overrides ────────────────
    p.add_argument("--s2s-engine", dest="s2s_engine",
                   choices=["moshi", "personaplex"], default=None,
                   help="Select the speech-to-speech backbone "
                        "(overrides s2s.engine in YAML)")
    p.add_argument("--voice-prompt", dest="voice_prompt", default=None,
                   help="PersonaPlex voice prompt — .pt embedding file or "
                        ".wav clip (abs or relative to project root)")
    p.add_argument("--text-prompt", dest="text_prompt", default=None,
                   help="PersonaPlex persona text prompt. Will be wrapped "
                        "<system>...<system> at tokenize time.")
    p.add_argument("--personaplex-precision", dest="personaplex_precision",
                   choices=["bf16"], default=None)
    p.add_argument("--personaplex-repo", dest="personaplex_repo", default=None,
                   help="HuggingFace repo for PersonaPlex weights "
                        "(default: nvidia/personaplex-7b-v1)")
    p.add_argument("--personaplex-device", dest="personaplex_device", default=None)

    # ── VAD / barge-in overrides (yaml is the source of truth; these win
    # when explicitly passed). Use the explicit true/false form so the
    # intent reads cleanly in shell history.
    p.add_argument("--vad-enabled", dest="vad_enabled",
                   choices=["true", "false"], default=None,
                   help="Override vad.enabled from the YAML (true | false)")

    return p.parse_args(argv)


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply CLI overrides on top of YAML settings (only for fields the user set)."""
    if args.host is not None:
        settings.server.host = args.host
    if args.port is not None:
        settings.server.port = args.port
    if args.show_sync is not None:
        settings.server.show_sync = bool(args.show_sync)

    if args.ref_image is not None:
        settings.paths.ref_image = resolve_path(args.ref_image)
    if args.adapter_ckpt_path is not None:
        settings.paths.adapter_ckpt_path = resolve_path(args.adapter_ckpt_path)
    if args.static_index is not None:
        settings.paths.static_index = resolve_path(args.static_index)

    if args.flash_model_type is not None:
        settings.flashhead.model_type = args.flash_model_type
    if args.base_seed is not None:
        settings.flashhead.base_seed = args.base_seed
    if args.warmup_min_chunks is not None:
        settings.flashhead.warmup_min_chunks = args.warmup_min_chunks
    if args.warmup_max_chunks is not None:
        settings.flashhead.warmup_max_chunks = args.warmup_max_chunks
    if args.warmup_target_ms_per_frame is not None:
        settings.flashhead.warmup_target_ms_per_frame = args.warmup_target_ms_per_frame
    if args.flashhead_device is not None:
        settings.flashhead.device = args.flashhead_device

    if args.moshi_precision is not None:
        settings.moshi.precision = args.moshi_precision
    if args.moshi_repo is not None:
        settings.moshi.repo_override = args.moshi_repo
    if args.moshi_device is not None:
        settings.moshi.device = args.moshi_device

    # ── S2S engine + PersonaPlex overrides ──────────────────────────
    if args.s2s_engine is not None:
        settings.s2s.engine = args.s2s_engine

    if settings.personaplex is not None:
        if args.voice_prompt is not None:
            settings.personaplex.voice_prompt = resolve_path(args.voice_prompt)
        if args.text_prompt is not None:
            settings.personaplex.text_prompt = args.text_prompt
        if args.personaplex_precision is not None:
            settings.personaplex.precision = args.personaplex_precision
        if args.personaplex_repo is not None:
            settings.personaplex.repo_override = args.personaplex_repo
        if args.personaplex_device is not None:
            settings.personaplex.device = args.personaplex_device
    elif (
        args.voice_prompt or args.text_prompt or args.personaplex_precision
        or args.personaplex_repo or args.personaplex_device
    ):
        # User passed a PersonaPlex flag without a personaplex block in YAML.
        raise SystemExit(
            "PersonaPlex CLI overrides were passed (--voice-prompt / "
            "--text-prompt / --personaplex-*), but the YAML has no "
            "`personaplex:` block. Add one or remove the overrides."
        )

    if args.vad_enabled is not None:
        settings.vad.enabled = (args.vad_enabled == "true")

    return settings


def _maybe_preprocess_ref_image_for_rotate_mode(settings: Settings) -> None:
    """When the served frontend is ``flashtalk_head_rotate.html``, pad the
    reference image to a square + resize to 512×512 and overwrite
    ``settings.paths.ref_image`` with the padded version. Stash the crop box
    so the browser can strip the padding back out frame-by-frame.

    No-op for any other static_index (default UniTalk behavior is preserved).

    The preprocessing helper lives in ``examples/ref_image_preprocess.py`` so
    this module never grows image-handling logic of its own.
    """
    static_index = settings.paths.static_index or ""
    if not static_index.endswith("flashtalk_head_rotate.html"):
        set_rotate_view(None)
        return

    helper_path = os.path.join(PROJECT_ROOT, "examples", "ref_image_preprocess.py")
    if not os.path.isfile(helper_path):
        print(f"[rotate-mode] Preprocess helper missing at {helper_path}; "
              "frames will be displayed as raw 512x512 squares.")
        set_rotate_view(None)
        return

    # Lazy import so the preprocessor stays optional for non-rotate runs.
    import importlib.util
    mod_name = "_unitalk_ref_image_preprocess"
    spec = importlib.util.spec_from_file_location(mod_name, helper_path)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass inside the loaded module reaches into sys.modules — register
    # the module BEFORE exec_module so its decorators don't see a None entry.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    src = settings.paths.ref_image
    print(f"[rotate-mode] Preprocessing reference image: {src}")
    out_path, box = mod.preprocess_ref_image(src)
    print(f"[rotate-mode] Wrote padded ref image: {out_path}")
    print(f"[rotate-mode] CropBox: {box.to_dict()}")

    settings.paths.ref_image = out_path
    set_rotate_view(box.to_dict())


def _patch_ref_image_path_for_workers(settings: Settings) -> None:
    """For non-rank-0 workers: predict where rank 0's preprocessor will
    write the padded PNG and point ``settings.paths.ref_image`` at it.
    No file IO — rank 0 writes the file before the NCCL handshake, so
    by the time workers actually open it inside ``set_reference``, it
    exists.

    Matches ``examples/ref_image_preprocess.py`` naming:
    ``{stem}_unitalk_512.png``.
    """
    static_index = settings.paths.static_index or ""
    if not static_index.endswith("flashtalk_head_rotate.html"):
        return
    src = settings.paths.ref_image
    stem, _ = os.path.splitext(os.path.abspath(src))
    settings.paths.ref_image = f"{stem}_unitalk_512.png"


def _ensure_port_available(host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
    except OSError as e:
        raise RuntimeError(
            f"Port {host}:{port} unavailable ({e}). Use another --port."
        ) from e
    finally:
        sock.close()


def _print_banner(settings: Settings) -> None:
    """Echo the chosen runtime config so users can sanity-check at startup."""
    model_type = settings.flashhead.model_type
    if model_type == "lite":
        tokens_per_chunk, accumulation_ms, est_gen_ms = 12, 960, 350
    else:
        tokens_per_chunk, accumulation_ms, est_gen_ms = 14, 1120, 600
    est_latency = accumulation_ms + est_gen_ms

    ctx = get_dist_context()
    gpu_mode = (
        f"multi-GPU sequence-parallel ({ctx.world_size}× ranks)"
        if ctx.multi_gpu else "single-GPU"
    )

    engine = settings.s2s.engine
    if engine == "personaplex" and settings.personaplex is not None:
        s2s_precision = settings.personaplex.precision
        s2s_device = settings.personaplex.device
    else:
        s2s_precision = settings.moshi.precision
        s2s_device = settings.moshi.device

    try:
        active_adapter = resolve_active_adapter_path(settings)
    except ValueError as e:
        active_adapter = f"<unresolved: {e}>"

    print("\n" + "=" * 70)
    print("  UniTalk — Streaming Speech-to-Avatar Server")
    print("=" * 70)
    print(f"  GPU Mode          : {gpu_mode}")
    print(f"  Host:Port         : {settings.server.host}:{settings.server.port}")
    print(f"  Flash Model       : {model_type}")
    print(f"  S2S Engine        : {engine}")
    print(f"  S2S Precision     : {s2s_precision}")
    print(f"  S2S Device        : {s2s_device}")
    print(f"  FlashHead Device  : {settings.flashhead.device}")
    if engine == "personaplex" and settings.personaplex is not None:
        vp = settings.personaplex.voice_prompt or "<none>"
        tp = settings.personaplex.text_prompt or "<none>"
        # Truncate long persona prompts for the banner.
        tp_display = tp if len(tp) <= 60 else tp[:57] + "..."
        print(f"  Voice Prompt      : {vp}")
        print(f"  Persona Prompt    : {tp_display}")
    print(f"  Tokens/Chunk      : {tokens_per_chunk}")
    print(f"  Accumulation      : {accumulation_ms}ms")
    print(f"  Est. Gen Time     : ~{est_gen_ms}ms")
    print(f"  Est. 1st Chunk    : ~{est_latency}ms")
    print(f"  Buffer Latency    : dynamic (starts on first chunk arrival)")
    print(f"  Warmup            : adaptive, "
          f"min={settings.flashhead.warmup_min_chunks} "
          f"max={settings.flashhead.warmup_max_chunks} "
          f"target≤{settings.flashhead.warmup_target_ms_per_frame:.0f}ms/frame")
    print(f"  Ref Image         : {settings.paths.ref_image}")
    print(f"  Adapter Ckpt Path : {active_adapter}")
    print(f"  Static Index      : {settings.paths.static_index}")
    print(f"  Show Sync Info    : {settings.server.show_sync}")
    print(f"  VAD (barge-in)    : {'enabled' if settings.vad.enabled else 'disabled'}")
    print("=" * 70 + "\n")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    settings = _apply_overrides(load_settings(args.config), args)

    # Detect torchrun launch. Single-process runs see WORLD_SIZE unset
    # and fall through to the legacy single-GPU path.
    ctx = init_dist_context()
    is_rank0 = ctx.is_rank0

    # Rank 0 owns the WebSocket port + ref-image preprocessing; worker
    # ranks skip both. The NCCL handshake inside flash_engine.load()
    # (which workers hit before they read the padded ref image) serves
    # as an implicit barrier — workers can't read the file until rank 0
    # has entered the same handshake, which happens AFTER preprocessing.
    if is_rank0:
        _ensure_port_available(settings.server.host, settings.server.port)
        _maybe_preprocess_ref_image_for_rotate_mode(settings)
        _print_banner(settings)
    else:
        # Workers still need to know where the padded ref image will
        # land so set_reference() reads the right path. We patch the
        # settings without touching disk — same path math, no IO.
        from .streaming import set_rotate_view
        _patch_ref_image_path_for_workers(settings)
        set_rotate_view(None)   # rotate_view is only consumed by rank 0's WebSocket
        print(f"[rank {ctx.rank}/{ctx.world_size}] worker bootstrap "
              f"on {ctx.device}")

    # Lockstep model load + warmup + idle render. Every rank participates
    # in the FlashHead pipeline calls; only rank 0 keeps the S2S engine
    # + the encoded idle frames.
    s2s_engine, flash_engine, idle_jpeg_b64 = load_engines(settings)

    if is_rank0:
        app = create_app(settings, s2s_engine, flash_engine, idle_jpeg_b64)
        try:
            uvicorn.run(
                app,
                host=settings.server.host,
                port=settings.server.port,
                log_level="info",
            )
        finally:
            # Tell worker ranks to break out of their broadcast loop.
            # Safe to call in single-GPU mode too (no-op).
            shutdown_workers(device=ctx.device)
    else:
        # Worker rank: enter the broadcast loop and stay there until
        # rank 0 sends CMD_STOP (or this process is killed).
        run_worker_loop(flash_engine)


if __name__ == "__main__":
    main()
