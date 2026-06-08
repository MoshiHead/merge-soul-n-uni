"""UniTalk model wrappers: adapter, FlashHead engine, S2S engines, factory."""

from __future__ import annotations

from .adapter import MoshiToWav2VecAdapter
from .flashhead_engine import FlashHeadTokenEngine
from .moshi_engine import MoshiEngine
from .s2s_base import S2SEngine

__all__ = [
    "MoshiToWav2VecAdapter",
    "FlashHeadTokenEngine",
    "MoshiEngine",
    "S2SEngine",
    "make_s2s_engine",
    "resolve_active_adapter_path",
]


def make_s2s_engine(settings, device: str) -> S2SEngine:
    """Factory: pick the active engine based on ``settings.s2s.engine``.

    Only the active engine's source tree is touched by ``sys.path``, so
    the ``moshi`` package-name collision between the two upstream trees
    is handled implicitly (we never load both in the same process).
    """
    engine = settings.s2s.engine
    if engine == "moshi":
        return MoshiEngine(
            settings=settings.moshi,
            moshi_pkg=settings.paths.moshi_pkg,
            device=device,
        )
    if engine == "personaplex":
        if settings.personaplex is None:
            raise ValueError(
                "s2s.engine='personaplex' but no `personaplex:` block in the YAML. "
                "Add one (see unitalk/config/inference.yaml.example) or pick "
                "s2s.engine='moshi'."
            )
        # Lazy import — PersonaPlex pulls in the cloned tree on first import.
        from .personaplex_engine import PersonaPlexEngine
        return PersonaPlexEngine(
            settings=settings.personaplex,
            pplex_pkg=settings.personaplex.pkg_path,
            device=device,
        )
    raise ValueError(
        f"Unknown s2s.engine: {engine!r}. Expected 'moshi' or 'personaplex'."
    )


def resolve_active_adapter_path(settings) -> str:
    """Resolve which adapter `.pt` file the active engine should load.

    Priority:
      1. ``paths.adapter_ckpt_path`` (explicit CLI/YAML override) — returned
         as-is. The user explicitly chose this path; we don't second-guess.
      2. The active engine's ``default_adapter_path``. If the literal file
         exists, return it. Otherwise scan the same directory for tolerant
         fallbacks (so a user who dropped in a file with a slightly
         different name still gets it picked up):
           a. Same basename without the ``_best`` suffix
              (e.g. ``moshi_to_flashhead_adapter.pt``)
           b. Newest ``*_ep{N}.pt`` sibling
           c. Newest ``*_latest.pt`` sibling
         If a fallback is used, the chosen path is returned and the
         caller (FlashHead) prints which file it actually loaded.
      3. Hard error — neither is configured.
    """
    import os
    import glob

    explicit = getattr(settings.paths, "adapter_ckpt_path", None)
    if explicit:
        return explicit

    engine = settings.s2s.engine
    engine_cfg = getattr(settings, engine, None)
    default = getattr(engine_cfg, "default_adapter_path", None) if engine_cfg else None
    if not default:
        raise ValueError(
            f"No adapter checkpoint configured. Set either paths.adapter_ckpt_path "
            f"or {engine}.default_adapter_path in your YAML, or pass "
            f"--adapter-ckpt-path on the command line."
        )

    if os.path.isfile(default):
        return default

    # Tolerant fallback search — same dir, common name variations.
    # Order is deliberate: highest-quality artefact first.
    #
    # Quality hierarchy for adapters:
    #   phase2_best > phase1_best > phase2_ep{N} > phase1_ep{N} >
    #   phase2_latest > phase1_latest > legacy un-phased names
    #
    # This way users who configure phase1_best as the default still
    # auto-upgrade to phase2 weights once Phase 2 finishes training.
    adapter_dir = os.path.dirname(default)
    if not os.path.isdir(adapter_dir):
        return default  # let FlashHead's "NOT FOUND" warning fire

    candidates = []
    new_base = f"{engine}_to_flashhead"            # current canonical
    legacy_base = f"{engine}_to_flashhead_adapter"  # pre-phase-refactor

    # Highest quality first — phase2_best, then phase1_best
    for phase in (2, 1):
        candidates.append(os.path.join(adapter_dir, f"{new_base}_phase{phase}_best.pt"))

    # Per-epoch history (newest mtime first within each phase, phase2 > phase1)
    for phase in (2, 1):
        ep_cands = sorted(
            glob.glob(os.path.join(adapter_dir, f"{new_base}_phase{phase}_ep*.pt")),
            key=os.path.getmtime,
            reverse=True,
        )
        candidates.extend(ep_cands)

    # Latest (lower quality than any best/ep but better than nothing)
    for phase in (2, 1):
        candidates.append(os.path.join(adapter_dir, f"{new_base}_phase{phase}_latest.pt"))

    # Legacy un-phased patterns (pre-refactor; kept for backward compat
    # with previously-trained adapters that didn't include the phase tag).
    candidates.append(os.path.join(adapter_dir, f"{legacy_base}_best.pt"))
    candidates.append(os.path.join(adapter_dir, f"{legacy_base}.pt"))
    legacy_ep = sorted(
        glob.glob(os.path.join(adapter_dir, f"{legacy_base}_ep*.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    candidates.extend(legacy_ep)
    candidates.append(os.path.join(adapter_dir, f"{legacy_base}_latest.pt"))

    for cand in candidates:
        if cand == default:
            continue  # already tried above
        if os.path.isfile(cand):
            print(f"[Adapter] Resolver: {os.path.basename(default)} missing "
                  f"— using {os.path.basename(cand)} instead.")
            return cand

    # Nothing matched — return the configured default so the existing
    # "NOT FOUND (random init)" warning from FlashHead fires cleanly.
    return default
