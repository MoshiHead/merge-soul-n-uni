"""Configuration and physical constants for the UniTalk pipeline.

Two halves:
  * Constants — properties of the upstream Moshi and Wav2Vec2 architectures.
    Not user-tunable; change only if the underlying models change.
  * Settings  — typed view of ``unitalk/config/inference.yaml``. The single
    source of truth for paths and runtime options.

Modules elsewhere should accept a ``Settings`` (or one of its fields)
rather than reading globals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


# ──────────────────────────────────────────────────────────────────────────
# Physical / architectural constants
# ──────────────────────────────────────────────────────────────────────────

# Moshi Helium
MOSHI_SR = 24000                                # Mimi codec sample rate
MOSHI_TOKEN_RATE = 12.5                         # tokens / second
MOSHI_TOKEN_DURATION_MS = 80                    # ms / token
MOSHI_FRAME_SAMPLES = int(MOSHI_SR / MOSHI_TOKEN_RATE)   # 1920 samples/token
MOSHI_DIM = 4096                                # transformer_out hidden size
SILENCE_THRESHOLD = 0.01

# FlashHead audio context window
DEQUE_DURATION_S = 8                            # sliding window length
DEQUE_SIZE = int(DEQUE_DURATION_S * MOSHI_TOKEN_RATE)    # 100 tokens
FLASHHEAD_FPS = 25
FLASHHEAD_FRAME_MS = 1000 / FLASHHEAD_FPS       # 40 ms / frame
INTERP_TARGET = int(DEQUE_DURATION_S * FLASHHEAD_FPS)    # 200 frame-tokens
FRAME_NUM = 33                                  # FlashHead frames / chunk

# Wav2Vec2-base output format (what FlashHead's AudioProjModel expects)
WAV2VEC_LAYERS = 12
WAV2VEC_DIM = 768


# ──────────────────────────────────────────────────────────────────────────
# YAML-backed Settings
# ──────────────────────────────────────────────────────────────────────────

# Project root = the directory containing the ``unitalk/`` package.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)


def resolve_path(path: str) -> str:
    """Resolve a path against the project root if it's not already absolute.

    Both absolute paths (``/workspace/UniTalk/checkpoints/foo.pt``) and
    project-relative paths (``checkpoints/foo.pt``) are supported.
    """
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


@dataclass
class PathsSettings:
    soulx_root: str
    moshi_pkg: str
    flash_ckpt: str
    flash_wav2vec: str
    ref_image: str
    static_dir: str
    static_index: str
    # Optional explicit adapter path. When None, the active S2S engine's
    # ``default_adapter_path`` is used (see resolve_active_adapter_path()
    # in unitalk/models/__init__.py).
    adapter_ckpt_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "PathsSettings":
        resolved = {}
        for k, v in d.items():
            if v is None:
                resolved[k] = None
            else:
                resolved[k] = resolve_path(v)
        return cls(**resolved)


@dataclass
class ServerSettings:
    host: str
    port: int
    show_sync: bool


@dataclass
class MoshiPreset:
    repo: str
    dtype: str


@dataclass
class MoshiSettings:
    precision: str
    repo_override: str | None
    device: str
    presets: dict[str, MoshiPreset]
    # Engine-specific default adapter path. Resolved against the project
    # root. Used when paths.adapter_ckpt_path is None.
    default_adapter_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "MoshiSettings":
        dap = d.get("default_adapter_path")
        return cls(
            precision=d["precision"],
            repo_override=d.get("repo_override"),
            device=d.get("device", "auto"),
            presets={k: MoshiPreset(**v) for k, v in d["presets"].items()},
            default_adapter_path=resolve_path(dap) if dap else None,
        )


@dataclass
class PersonaPlexSettings:
    """Config for the NVIDIA PersonaPlex S2S engine.

    PersonaPlex is a finetune of Moshi Helium that adds:
      * text persona prompt (wrapped <system>...<system> at tokenize time)
      * voice prompt — either a pre-baked .pt embedding file (NATF/NATM
        /VARF/VARM shipped in the repo's assets) or a raw .wav that
        ``LMGen.load_voice_prompt`` normalizes to -24 LUFS and mimi-encodes
    Both are optional — leaving them None gives a default-persona run.
    """

    precision: str
    repo_override: str | None       # null → presets[precision].repo
    device: str
    pkg_path: str                   # project-relative path to the inner package dir
    voice_prompt: str | None        # absolute or project-relative .pt or .wav
    text_prompt: str | None         # raw persona string (will be wrapped)
    presets: dict[str, MoshiPreset]
    default_adapter_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "PersonaPlexSettings":
        vp = d.get("voice_prompt")
        dap = d.get("default_adapter_path")
        return cls(
            precision=d.get("precision", "bf16"),
            repo_override=d.get("repo_override"),
            device=d.get("device", "auto"),
            pkg_path=resolve_path(d.get("pkg_path", "personaplex/moshi")),
            voice_prompt=resolve_path(vp) if vp else None,
            text_prompt=d.get("text_prompt"),
            presets={k: MoshiPreset(**v) for k, v in d.get("presets", {}).items()},
            default_adapter_path=resolve_path(dap) if dap else None,
        )


@dataclass
class S2SSettings:
    """Top-level switch for which speech-to-speech backbone to load."""

    engine: str = "moshi"            # "moshi" | "personaplex"


@dataclass
class FlashHeadSettings:
    model_type: str
    base_seed: int
    warmup_min_chunks: int
    warmup_max_chunks: int
    warmup_target_ms_per_frame: float
    device: str


@dataclass
class QueueSettings:
    token_max: int
    dispatch_max: int
    mic_max: int


@dataclass
class VADSettings:
    enabled: bool = True
    speech_threshold: float = 0.5
    min_speech_frames: int = 4         # 128 ms of real speech before fire.
    min_silence_frames: int = 25       # 800 ms of silence before unmute.
    cooldown_ms: float = 800.0
    # Optional pre-Silero RMS gate. 0.0 disables (default). Try 0.005-0.01
    # in noisy environments — see vad.py for details.
    energy_gate_rms: float = 0.0


@dataclass
class KeepwarmSettings:
    """Optional background idle-render loop that keeps FlashHead's CUDA
    kernels hot while no client session is active. Costs ~200ms of GPU
    every ``interval_s``; the trade-off is that the first chunk of the
    next session arrives at full speed instead of paying a cold-cache
    penalty. Recommended only when the host has spare GPU headroom.
    """

    enabled: bool = False
    interval_s: float = 5.0


@dataclass
class Settings:
    project: dict[str, Any]
    paths: PathsSettings
    server: ServerSettings
    moshi: MoshiSettings
    flashhead: FlashHeadSettings
    queues: QueueSettings
    vad: VADSettings
    s2s: S2SSettings
    personaplex: PersonaPlexSettings | None = None
    keepwarm: KeepwarmSettings = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.keepwarm is None:
            self.keepwarm = KeepwarmSettings()

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Old YAMLs without an `s2s:` block default to the legacy
        # Moshi-only behavior. Same for personaplex + keepwarm — optional.
        s2s_block = data.get("s2s") or {}
        pplex_block = data.get("personaplex")
        keepwarm_block = data.get("keepwarm") or {}
        return cls(
            project=data.get("project", {}),
            paths=PathsSettings.from_dict(data["paths"]),
            server=ServerSettings(**data["server"]),
            moshi=MoshiSettings.from_dict(data["moshi"]),
            flashhead=FlashHeadSettings(**data["flashhead"]),
            queues=QueueSettings(**data["queues"]),
            vad=VADSettings(**data.get("vad", {})),
            s2s=S2SSettings(engine=s2s_block.get("engine", "moshi")),
            personaplex=(
                PersonaPlexSettings.from_dict(pplex_block) if pplex_block else None
            ),
            keepwarm=KeepwarmSettings(**keepwarm_block),
        )


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "inference.yaml"
)


def load_settings(path: str | None = None) -> Settings:
    """Load settings from a YAML file (default: bundled ``inference.yaml``)."""
    return Settings.from_yaml(path or DEFAULT_CONFIG_PATH)
