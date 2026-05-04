"""Medical classification: same preparation logic as 01b_DataPreparation_Medical.ipynb."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
HR_MIN, HR_MAX = 25, 250
SBP_MIN, SBP_MAX = 50, 280
EPS = 1e-6

TARGET_MAP = {"negative": 0, "positive": 1}


def _base_dir() -> Path:
    roots = [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[2] / "classification_Medical_data _set"]
    for r in roots:
        if (r / "Medicaldataset.csv").exists():
            return r
    return Path(__file__).resolve().parents[2] / "classification_Medical_data _set"


def _processed_dir() -> Path:
    return _base_dir() / "data" / "processed" / "medical"


def load_metadata() -> dict[str, Any]:
    p = _processed_dir() / "metadata.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def clinical_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_work = df.copy()
    df_work["flag_hr_out_of_range"] = (
        (df_work["Heart rate"] < HR_MIN) | (df_work["Heart rate"] > HR_MAX)
    ).astype(int)
    df_work["flag_sbp_out_of_range"] = (
        (df_work["Systolic blood pressure"] < SBP_MIN) | (df_work["Systolic blood pressure"] > SBP_MAX)
    ).astype(int)
    df_work.loc[df_work["flag_hr_out_of_range"] == 1, "Heart rate"] = np.nan
    df_work.loc[df_work["flag_sbp_out_of_range"] == 1, "Systolic blood pressure"] = np.nan
    df_work["Pulse_Pressure"] = df_work["Systolic blood pressure"] - df_work["Diastolic blood pressure"]
    df_work["CK_MB_Troponin_Ratio"] = df_work["CK-MB"] / np.maximum(df_work["Troponin"], EPS)
    return df_work


def prepare_xy_from_raw_csv() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data_file = _base_dir() / "Medicaldataset.csv"
    df_raw = pd.read_csv(data_file)
    required = [
        "Age",
        "Gender",
        "Heart rate",
        "Systolic blood pressure",
        "Diastolic blood pressure",
        "Blood sugar",
        "CK-MB",
        "Troponin",
        "Result",
    ]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_work = clinical_engineering(df_raw)
    df_work["Result"] = df_work["Result"].astype(str).str.strip().str.lower().map(TARGET_MAP)
    if df_work["Result"].isna().any():
        raise ValueError("Unexpected target labels in Result column")

    X = df_work.drop(columns=["Result"])
    y = df_work["Result"].astype(int)
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def fit_preprocessors(X_train: pd.DataFrame) -> tuple[list[str], list[str], SimpleImputer, RobustScaler, list[str]]:
    meta = load_metadata()
    flag_cols = list(meta["flag_columns"])
    continuous_cols = [c for c in X_train.columns if c not in flag_cols]
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_train_cont_imputed = pd.DataFrame(
        imputer.fit_transform(X_train[continuous_cols]),
        columns=continuous_cols,
        index=X_train.index,
    )
    X_train_cont_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_cont_imputed),
        columns=continuous_cols,
        index=X_train.index,
    )
    X_train_processed = pd.concat([X_train_cont_scaled, X_train[flag_cols].astype(int)], axis=1)
    feature_order = continuous_cols + flag_cols
    X_train_processed = X_train_processed[feature_order]
    return continuous_cols, flag_cols, imputer, scaler, feature_order


def apply_preprocessing(
    X: pd.DataFrame,
    continuous_cols: list[str],
    flag_cols: list[str],
    imputer: SimpleImputer,
    scaler: RobustScaler,
    feature_order: list[str],
) -> pd.DataFrame:
    X_cont_imp = pd.DataFrame(
        imputer.transform(X[continuous_cols]),
        columns=continuous_cols,
        index=X.index,
    )
    X_cont_scl = pd.DataFrame(
        scaler.transform(X_cont_imp),
        columns=continuous_cols,
        index=X.index,
    )
    out = pd.concat([X_cont_scl, X[flag_cols].astype(int)], axis=1)
    return out[feature_order]


@dataclass
class MedicalBundle:
    model: GradientBoostingClassifier
    imputer: SimpleImputer
    scaler: RobustScaler
    continuous_cols: list[str]
    flag_cols: list[str]
    feature_order: list[str]
    target_mapping: dict[str, int]

    def predict_dataframe(self, X_raw_engineered: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xp = apply_preprocessing(
            X_raw_engineered,
            self.continuous_cols,
            self.flag_cols,
            self.imputer,
            self.scaler,
            self.feature_order,
        )
        proba = self.model.predict_proba(Xp)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return pred, proba

    def predict_one(
        self,
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
        pred, proba = self.predict_dataframe(eng)
        label = "positive" if pred[0] == 1 else "negative"
        return {
            "prediction_label": label,
            "prediction_code": int(pred[0]),
            "probability_positive": float(proba[0]),
        }

    @classmethod
    def train_default(cls) -> MedicalBundle:
        X_train, _, y_train, _ = prepare_xy_from_raw_csv()
        continuous_cols, flag_cols, imputer, scaler, feature_order = fit_preprocessors(X_train)
        X_train_p = apply_preprocessing(X_train, continuous_cols, flag_cols, imputer, scaler, feature_order)
        model = GradientBoostingClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            max_depth=3,
            learning_rate=0.1,
        )
        model.fit(X_train_p, y_train)
        meta = load_metadata()
        return cls(
            model=model,
            imputer=imputer,
            scaler=scaler,
            continuous_cols=continuous_cols,
            flag_cols=flag_cols,
            feature_order=feature_order,
            target_mapping=meta.get("target_mapping", TARGET_MAP),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "imputer": self.imputer,
                "scaler": self.scaler,
                "continuous_cols": self.continuous_cols,
                "flag_cols": self.flag_cols,
                "feature_order": self.feature_order,
                "target_mapping": self.target_mapping,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> MedicalBundle:
        d = joblib.load(path)
        return cls(
            model=d["model"],
            imputer=d["imputer"],
            scaler=d["scaler"],
            continuous_cols=list(d["continuous_cols"]),
            flag_cols=list(d["flag_cols"]),
            feature_order=list(d["feature_order"]),
            target_mapping=dict(d["target_mapping"]),
        )
