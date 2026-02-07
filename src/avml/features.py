from __future__ import annotations

from pathlib import Path
import hashlib
import math
from typing import Dict, Iterable, List, Optional

# Optional: PE parsing (Windows .exe/.dll). Keep optional so mac users can still run.
try:
    import pefile  # type: ignore
except Exception:
    pefile = None


FEATURE_ORDER: List[str] = [
    "file_size",
    "entropy",
    "is_executable",
    "has_pe_header",
    "pe_num_sections",
    "pe_num_imports",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entropy(path: Path, max_bytes: int = 2_000_000) -> float:
    """
    Shannon entropy over up to max_bytes (default ~2MB) to avoid huge memory usage.
    """
    with path.open("rb") as f:
        data = f.read(max_bytes)

    if not data:
        return 0.0

    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1

    entropy = 0.0
    length = len(data)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def is_windows_executable(path: Path) -> int:
    return int(path.suffix.lower() in [".exe", ".dll"])


def pe_features(path: Path) -> Dict[str, float]:
    """
    Extract a few safe PE-level stats if pefile is available and file is a PE.
    Returns numeric-only features (good for ML).
    """
    # Default values if not a PE or pefile not available
    base = {
        "has_pe_header": 0.0,
        "pe_num_sections": 0.0,
        "pe_num_imports": 0.0,
    }

    if pefile is None:
        return base

    try:
        pe = pefile.PE(str(path), fast_load=True)
        base["has_pe_header"] = 1.0

        # Sections
        base["pe_num_sections"] = float(len(getattr(pe, "sections", []) or []))

        # Imports (need to parse directories)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        imports = 0
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                imports += len(entry.imports)
        base["pe_num_imports"] = float(imports)

        return base
    except Exception:
        return base


def extract_features(file_path: str) -> Dict[str, object]:
    """
    Human/report fields + numeric ML fields. Keep both, but ML will use only FEATURE_ORDER.
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise ValueError(f"Invalid file path: {file_path}")

    feats_num: Dict[str, float] = {
        "file_size": float(path.stat().st_size),
        "entropy": round(file_entropy(path), 3),
        "is_executable": float(is_windows_executable(path)),
    }
    feats_num.update(pe_features(path))

    report_fields: Dict[str, object] = {
        "path": str(path),
        "extension": path.suffix.lower(),
        "sha256": sha256(path),
    }

    # Merge: numeric first + report fields
    return {**feats_num, **report_fields}


def vectorize(features: Dict[str, object]) -> List[float]:
    """
    Convert feature dict -> numeric vector in fixed order for ML.
    """
    vec: List[float] = []
    for key in FEATURE_ORDER:
        val = features.get(key, 0.0)
        try:
            vec.append(float(val))  # type: ignore[arg-type]
        except Exception:
            vec.append(0.0)
    return vec


def iter_files(root: str, recursive: bool = True) -> Iterable[Path]:
    """
    Yield files under root. Useful for batch extraction.
    """
    p = Path(root)
    if p.is_file():
        yield p
        return

    if recursive:
        yield from (x for x in p.rglob("*") if x.is_file())
    else:
        yield from (x for x in p.glob("*") if x.is_file())
