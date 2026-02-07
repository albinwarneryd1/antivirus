from pathlib import Path
import hashlib
import math


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entropy(path: Path) -> float:
    data = path.read_bytes()
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


def extract_features(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise ValueError("Invalid file path")

    return {
        "file_size": path.stat().st_size,
        "entropy": round(file_entropy(path), 3),
        "is_executable": int(path.suffix.lower() in [".exe", ".dll"]),
        "extension": path.suffix.lower(),
        "sha256": sha256(path),
    }
