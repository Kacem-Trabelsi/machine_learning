"""Train models and write joblib artifacts under web_app/artifacts/. Run once after clone."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from medical_pipeline import MedicalBundle

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def train_medical() -> None:
    bundle = MedicalBundle.train_default()
    bundle.save(ART / "medical_deploy_bundle.pkl")
    print("Wrote", ART / "medical_deploy_bundle.pkl")


def train_hospital() -> None:
    project = ROOT.parent
    data_dir = project / "regression_hospital_data _set" / "data" / "processed" / "hospital"
    meta_path = data_dir / "metadata.json"
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    cols = meta["X_train_columns"]
    X_train = X_train[cols]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    artifact = {"model": model, "feature_names": cols, "target_description": "Duration in hospital (days)"}
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ART / "hospital_rf.pkl")
    print("Wrote", ART / "hospital_rf.pkl")


if __name__ == "__main__":
    ART.mkdir(parents=True, exist_ok=True)
    train_medical()
    train_hospital()
