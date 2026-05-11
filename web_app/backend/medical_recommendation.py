"""Recommandation par similarité : k plus proches voisins dans l'espace des features médicales prétraitées."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medical_pipeline import MedicalBundle, TARGET_MAP, clinical_engineering

CODE_TO_LABEL = {v: k for k, v in TARGET_MAP.items()}


def recommendation_bundle_paths(project_root: Path) -> tuple[Path, Path]:
    rec_dir = project_root / "classification_Medical_data _set" / "recommendation_model"
    art_dir = project_root / "web_app" / "artifacts"
    return rec_dir / "medical_case_based_neighbor_bundle.pkl", art_dir / "medical_case_based_neighbor_bundle.pkl"


def load_neighbor_bundle(project_root: Path) -> dict[str, Any] | None:
    """Charge le premier bundle disponible (dossier `recommendation_model` ou `web_app/artifacts`)."""
    import joblib

    p1, p2 = recommendation_bundle_paths(project_root)
    for p in (p1, p2):
        if p.exists():
            return joblib.load(p)
    return None


def recommend_from_vitals(
    medical: MedicalBundle,
    bundle: dict[str, Any],
    *,
    age: float,
    gender: int,
    heart_rate: float,
    systolic_bp: float,
    diastolic_bp: float,
    blood_sugar: float,
    ck_mb: float,
    troponin: float,
) -> dict[str, Any]:
    row = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": float(gender),
                "Heart rate": heart_rate,
                "Systolic blood pressure": systolic_bp,
                "Diastolic blood pressure": diastolic_bp,
                "Blood sugar": blood_sugar,
                "CK-MB": ck_mb,
                "Troponin": troponin,
            }
        ]
    )
    eng = clinical_engineering(row)
    Xp = medical.processed_features(eng)
    cols: list[str] = list(bundle["feature_columns"])
    Xq = Xp[cols].to_numpy(dtype=float)
    nn = bundle["nearest_neighbors"]
    k = min(int(bundle["k_neighbors"]), int(bundle["y_train_reference"].shape[0]))
    dist, idx = nn.kneighbors(Xq, n_neighbors=k)
    y_ref: pd.Series = bundle["y_train_reference"]
    neighbors: list[dict[str, Any]] = []
    labels: list[int] = []
    for rank, (d, j) in enumerate(zip(dist[0].tolist(), idx[0].tolist()), start=1):
        code = int(y_ref.iloc[int(j)])
        labels.append(code)
        neighbors.append(
            {
                "rank": rank,
                "distance": float(d),
                "historical_outcome": CODE_TO_LABEL.get(code, str(code)),
                "outcome_code": code,
            }
        )
    frac_pos = float(np.mean(labels)) if labels else 0.0
    majority = 1 if frac_pos >= 0.5 else 0
    return {
        "k_neighbors": k,
        "metric": bundle.get("metric", "euclidean"),
        "neighbor_fraction_positive": frac_pos,
        "neighbor_majority_label": CODE_TO_LABEL.get(majority, str(majority)),
        "neighbor_majority_code": majority,
        "neighbors": neighbors,
    }
