from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib

from avml.features import FEATURE_ORDER, extract_features
import pandas as pd


@dataclass
class ScanConfig:
    model_path: str = "models/model.joblib"
    schema_path: str = "models/feature_order.joblib"
    threshold: float = 0.5


class FileScanner:
    from avml.features import iter_files
    import pandas as pd

    def __init__(self, config: ScanConfig):
        self.cfg = config
        self.model = joblib.load(self.cfg.model_path)
        self.schema = joblib.load(self.cfg.schema_path)

    def _vector_from_features(self, feats: Dict[str, Any]) -> list[float]:
        order = self.schema if isinstance(self.schema, list) else FEATURE_ORDER
        return [float(feats.get(k, 0.0)) for k in order]

    def scan_file(self, file_path: str) -> Dict[str, Any]:
        feats = extract_features(file_path)
        x = self._vector_from_features(feats)
        order = self.schema if isinstance(self.schema, list) else FEATURE_ORDER
        X = pd.DataFrame([x], columns=order)
        proba = self.model.predict_proba(X)[0]
        
        def scan_folder(self, folder: str, recursive: bool = True, top_n: int = 25) -> list[dict]:
            results = []
            for p in iter_files(folder, recursive=recursive):
                try:
                    r = self.scan_file(str(p))
                    results.append(r)
                except Exception:
                    continue
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return results[:top_n]



        if len(proba) == 2:
            score = float(proba[1])
        else:
            score = float(max(proba))

        verdict = "malicious" if score >= self.cfg.threshold else "benign"

        return {
            "verdict": verdict,
            "score": round(score, 4),
            "threshold": self.cfg.threshold,
            "path": feats.get("path", file_path),
            "sha256": feats.get("sha256"),
            "extension": feats.get("extension"),
            "features": {k: feats.get(k) for k in (self.schema if isinstance(self.schema, list) else FEATURE_ORDER)},
        }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to file to scan")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--schema", default="models/feature_order.joblib")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="", help="Write JSON report to this path")
    parser.add_argument("path", help="File or folder to scan")
    parser.add_argument("--folder", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--recursive", action="store_true")

    args = parser.parse_args()

    cfg = ScanConfig(model_path=args.model, schema_path=args.schema, threshold=args.threshold)
    scanner = FileScanner(cfg)
    report = scanner.scan_file(args.file)

    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
