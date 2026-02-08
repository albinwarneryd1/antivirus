from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from avml.features import FEATURE_ORDER, extract_features, iter_files


@dataclass
class DatasetConfig:
    benign_dir: str = ""
    malicious_dir: str = ""
    input_dir: str = ""
    out_csv: str = "data/processed.csv"
    recursive: bool = True
    label_col: str = "label"
    benign_label: int = 0
    malicious_label: int = 1


class DatasetBuilder:
    def __init__(self, config: DatasetConfig):
        self.cfg = config

    def _row(self, feats: Dict[str, Any], label: int | None) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for k in FEATURE_ORDER:
            row[k] = float(feats.get(k, 0.0))
        row["path"] = feats.get("path")
        row["sha256"] = feats.get("sha256")
        row["extension"] = feats.get("extension")
        if label is not None:
            row[self.cfg.label_col] = int(label)
        return row

    def build(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        if self.cfg.benign_dir and self.cfg.malicious_dir:
            for p in iter_files(self.cfg.benign_dir, recursive=self.cfg.recursive):
                feats = extract_features(str(p))
                rows.append(self._row(feats, self.cfg.benign_label))

            for p in iter_files(self.cfg.malicious_dir, recursive=self.cfg.recursive):
                feats = extract_features(str(p))
                rows.append(self._row(feats, self.cfg.malicious_label))

        elif self.cfg.input_dir:
            for p in iter_files(self.cfg.input_dir, recursive=self.cfg.recursive):
                feats = extract_features(str(p))
                rows.append(self._row(feats, None))

        else:
            raise ValueError("Provide either benign_dir+malicious_dir or input_dir")

        return pd.DataFrame(rows)

    def save(self, df: pd.DataFrame) -> None:
        out = Path(self.cfg.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    def run(self) -> None:
        df = self.build()
        self.save(df)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benign", default="")
    parser.add_argument("--malicious", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--out", default="data/processed.csv")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    cfg = DatasetConfig(
        benign_dir=args.benign,
        malicious_dir=args.malicious,
        input_dir=args.input,
        out_csv=args.out,
        recursive=bool(args.recursive),
        label_col=args.label_col,
    )
    DatasetBuilder(cfg).run()


if __name__ == "__main__":
    main()
