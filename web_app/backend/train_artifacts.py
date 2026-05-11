"""Train models and write joblib artifacts under web_app/artifacts/. Run once after clone."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors

from medical_pipeline import TARGET_MAP, MedicalBundle

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


def train_recommendation_neighbors(k_neighbors: int = 15, metric: str = "euclidean") -> None:
    """Index k-NN sur X_train médical (notebook 04_Recommendation_Medical)."""
    project = ROOT.parent
    data_dir = project / "classification_Medical_data _set" / "data" / "processed" / "medical"
    rec_dir = project / "classification_Medical_data _set" / "recommendation_model"
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; run preparation notebook for medical data first.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    cont = list(meta["continuous_columns"])
    flags = list(meta["flag_columns"])
    cols = cont + flags
    X_train = pd.read_csv(data_dir / "X_train.csv")[cols]
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
    X_test = pd.read_csv(data_dir / "X_test.csv")[cols]
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

    nn = NearestNeighbors(n_neighbors=k_neighbors, metric=metric, algorithm="auto")
    nn.fit(X_train)
    _, idx = nn.kneighbors(X_test, n_neighbors=k_neighbors)
    labs = y_train.iloc[idx.ravel()].to_numpy().reshape(idx.shape)
    y_rec = (labs.mean(axis=1) >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_rec))
    rec_pos = float(recall_score(y_test, y_rec, pos_label=1))
    f1 = float(f1_score(y_test, y_rec))

    bundle = {
        "nearest_neighbors": nn,
        "feature_columns": cols,
        "k_neighbors": k_neighbors,
        "metric": metric,
        "y_train_reference": y_train.reset_index(drop=True),
    }
    rec_dir.mkdir(parents=True, exist_ok=True)
    path_rec = rec_dir / "medical_case_based_neighbor_bundle.pkl"
    path_art = ART / "medical_case_based_neighbor_bundle.pkl"
    joblib.dump(bundle, path_rec)
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path_art)
    sidecar = {
        "artifact": "medical_case_based_neighbor_bundle.pkl",
        "k_neighbors": k_neighbors,
        "metric": metric,
        "processed_data_dir": str(data_dir.resolve()),
        "holdout_neighbor_majority": {"accuracy": acc, "recall_positive": rec_pos, "f1": f1},
        "target_mapping": meta.get("target_mapping", TARGET_MAP),
    }
    (rec_dir / "medical_recommendation_sidecar.json").write_text(
        json.dumps(sidecar, indent=2),
        encoding="utf-8",
    )
    print("Wrote", path_rec)
    print("Wrote", path_art)
    print(f"Hold-out neighbor majority: acc={acc:.4f} recall_pos={rec_pos:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    ART.mkdir(parents=True, exist_ok=True)
    train_medical()
    train_hospital()
    train_recommendation_neighbors()
