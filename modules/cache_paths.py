"""Shared Hugging Face cache policy for Dubbing Control Center.

The new project cache is the write target by default. Existing standard and
legacy caches are read-only fallback locations; this module never moves or
removes model files.
"""
from __future__ import annotations

import os
import json
from pathlib import Path


def _value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def hf_cache_paths() -> tuple[Path, Path]:
    """Return (HF_HOME, HF_HUB_CACHE), respecting explicit user overrides."""
    default_home = Path.home() / ".dubbing-control-center" / "models"
    home_value = _value("DUBBING_HF_HOME", "HF_HOME")
    hf_home = Path(home_value).expanduser() if home_value else default_home

    hub_value = _value(
        "DUBBING_HF_HUB_CACHE",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
    )
    hub_cache = Path(hub_value).expanduser() if hub_value else hf_home / "hub"
    return hf_home, hub_cache


def configure_hf_cache() -> tuple[Path, Path]:
    """Set HF environment/constants before importing or constructing HF models."""
    hf_home, hub_cache = hf_cache_paths()
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)

    try:
        import huggingface_hub.constants as constants

        constants.HF_HOME = str(hf_home)
        constants.HF_HUB_CACHE = str(hub_cache)
    except (ImportError, AttributeError):
        pass
    return hf_home, hub_cache


def hf_hub_search_roots() -> list[Path]:
    """Return unique cache roots, newest project/explicit roots before fallbacks."""
    hf_home, configured_hub = hf_cache_paths()
    candidates = [
        configured_hub,
        hf_home / "hub",
        Path.home() / ".dubbing-control-center" / "models" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
        # Read-only compatibility paths used by older ZastTranslate builds.
        Path.home() / ".cache" / "zasttranslate" / "models" / "hub",
        Path.home() / ".cache" / "zasttranslate" / "hub",
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            roots.append(resolved)
    return roots


def hf_snapshots(repo_id: str):
    """Yield local snapshot directories for a Hub repo in search-priority order."""
    repo_dir = "models--" + repo_id.strip().replace("/", "--")
    for root in hf_hub_search_roots():
        snapshot_root = root / repo_dir / "snapshots"
        if not snapshot_root.is_dir():
            continue
        try:
            entries = sorted(
                (entry for entry in snapshot_root.iterdir() if entry.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            continue
        yield from entries


def transformers_weights_complete(snapshot: Path) -> bool:
    """Validate all shards in a Transformers index, or every unindexed weight file."""
    snapshot = Path(snapshot)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, json.JSONDecodeError):
            return False
        shards = {str(name) for name in weight_map.values()}
        return bool(shards) and all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0 for name in shards)

    weights = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("*.bin"))
    return bool(weights) and all(path.is_file() and path.stat().st_size > 0 for path in weights)
