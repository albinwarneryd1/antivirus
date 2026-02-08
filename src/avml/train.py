from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from avml.features import FEATURE_ORDER


@dataclass
class TrainConfig:
    data_csv: str
    label_col: str = "label"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int | None = None
    model_out: str = "models/model.joblib"
    schema_out: str = "models/feature_order.joblib"


class ModelTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config

    def load_data(self) -> pd.DataFrame:
        path = Path(self.cfg.data_csv)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return pd.read_csv(path)

    def build_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required feature columns: {missing}")

        if self.cfg.label_col not in df.columns:
            raise ValueError(f"Label column '{self.cfg.label_col}' not found")

        X = df[FEATURE_ORDER].copy()
        y = df[self.cfg.label_col].copy()
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y if y.nunique() > 1 and y.value_counts().min() >= 2 else None,

        )

        model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            random_state=self.cfg.random_state,
            max_depth=self.cfg.max_depth,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred,
            average="binary" if y.nunique() == 2 else "weighted",
            zero_division=0
        )

        print(cm)
        print(f"precision: {precision:.3f}")
        print(f"recall: {recall:.3f}")
        print(f"f1: {f1:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        return model

    def save(self, model: RandomForestClassifier) -> None:
        Path(self.cfg.model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.cfg.model_out)
        joblib.dump(FEATURE_ORDER, self.cfg.schema_out)

    def run(self) -> None:
        df = self.load_data()
        X, y = self.build_xy(df)
        model = self.train(X, y)
        self.save(model)


def main() -> None:
    cfg = TrainConfig(
        data_csv="data/processed.csv",
        label_col="label",
    )
    ModelTrainer(cfg).run()


if __name__ == "__main__":
    main()
