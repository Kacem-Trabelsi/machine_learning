"""FastAPI backend: classification médicale, recommandation par similarité, régression hôpital."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from medical_pipeline import TARGET_MAP, MedicalBundle
from medical_recommendation import load_neighbor_bundle, recommend_from_vitals

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PROJECT_ROOT = ROOT.parent
STATIC = ROOT / "frontend"
REC_SIDECAR = (
    PROJECT_ROOT
    / "classification_Medical_data _set"
    / "recommendation_model"
    / "medical_recommendation_sidecar.json"
)

app = FastAPI(title="Medical ML API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_medical: MedicalBundle | None = None
_hospital: dict[str, Any] | None = None
_neighbor_bundle: dict[str, Any] | None = None


@app.on_event("startup")
def load_models() -> None:
    global _medical, _hospital, _neighbor_bundle
    ART.mkdir(parents=True, exist_ok=True)
    med_path = ART / "medical_deploy_bundle.pkl"
    if med_path.exists():
        _medical = MedicalBundle.load(med_path)
    h_path = ART / "hospital_rf.pkl"
    if h_path.exists():
        _hospital = joblib.load(h_path)
    _neighbor_bundle = load_neighbor_bundle(PROJECT_ROOT)


class MedicalIn(BaseModel):
    age: float = Field(..., ge=0, le=120)
    gender: int = Field(..., description="0 ou 1 (encodage du jeu de données)")
    heart_rate: float = Field(..., ge=0)
    systolic_bp: float = Field(..., ge=0)
    diastolic_bp: float = Field(..., ge=0)
    blood_sugar: float = Field(..., ge=0)
    ck_mb: float = Field(..., ge=0)
    troponin: float = Field(..., ge=0)


class HospitalIn(BaseModel):
    features: list[float]


@app.get("/api/hospital/random-features")
def hospital_random_features() -> dict[str, Any]:
    """Une ligne tirée au hasard dans X_train — valeurs toujours cohérentes avec le prétraitement."""
    data_dir = PROJECT_ROOT / "regression_hospital_data _set" / "data" / "processed" / "hospital"
    train_csv = data_dir / "X_train.csv"
    if not train_csv.exists():
        raise HTTPException(status_code=404, detail="Hospital training matrix not found")
    if _hospital is None:
        raise HTTPException(status_code=503, detail="Hospital model not trained; run train_artifacts.py")
    names: list[str] = _hospital["feature_names"]
    x_all = pd.read_csv(train_csv)[names]
    if len(x_all) < 1:
        raise HTTPException(status_code=404, detail="Empty X_train")
    row = x_all.sample(n=1, random_state=None).iloc[0]
    return {"feature_names": names, "example": [float(x) for x in row.tolist()]}


@app.get("/api/hospital/example-features")
def hospital_example_features() -> dict[str, Any]:
    data_dir = PROJECT_ROOT / "regression_hospital_data _set" / "data" / "processed" / "hospital"
    train_csv = data_dir / "X_train.csv"
    if not train_csv.exists():
        raise HTTPException(status_code=404, detail="Hospital training matrix not found")
    if _hospital is None:
        raise HTTPException(status_code=503, detail="Hospital model not trained; run train_artifacts.py")
    row = pd.read_csv(train_csv, nrows=1)
    names: list[str] = _hospital["feature_names"]
    row = row[names]
    return {"feature_names": names, "example": [float(x) for x in row.iloc[0].tolist()]}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "medical_loaded": _medical is not None,
        "hospital_loaded": _hospital is not None,
        "recommendation_loaded": _neighbor_bundle is not None,
        "target_mapping": TARGET_MAP,
    }


@app.post("/api/medical/predict")
def medical_predict(body: MedicalIn) -> dict[str, Any]:
    if _medical is None:
        raise HTTPException(
            status_code=503,
            detail="Medical model not found. Run: python web_app/backend/train_artifacts.py",
        )
    return _medical.predict_one(
        age=body.age,
        gender=body.gender,
        heart_rate=body.heart_rate,
        systolic_bp=body.systolic_bp,
        diastolic_bp=body.diastolic_bp,
        blood_sugar=body.blood_sugar,
        ck_mb=body.ck_mb,
        troponin=body.troponin,
    )


@app.get("/api/hospital/feature-names")
def hospital_feature_names() -> dict[str, Any]:
    if _hospital is None:
        raise HTTPException(status_code=503, detail="Hospital model not trained; run train_artifacts.py")
    return {"feature_names": _hospital["feature_names"], "n": len(_hospital["feature_names"])}


@app.post("/api/medical/recommend")
def medical_recommend(body: MedicalIn) -> dict[str, Any]:
    """k plus proches voisins dans X_train : consensus des issues historiques (démonstration)."""
    if _medical is None:
        raise HTTPException(
            status_code=503,
            detail="Medical model not found. Run: python web_app/backend/train_artifacts.py",
        )
    if _neighbor_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Neighbor bundle not found. Run: python web_app/backend/train_artifacts.py",
        )
    out = recommend_from_vitals(
        _medical,
        _neighbor_bundle,
        age=body.age,
        gender=body.gender,
        heart_rate=body.heart_rate,
        systolic_bp=body.systolic_bp,
        diastolic_bp=body.diastolic_bp,
        blood_sugar=body.blood_sugar,
        ck_mb=body.ck_mb,
        troponin=body.troponin,
    )
    clf = _medical.predict_one(
        age=body.age,
        gender=body.gender,
        heart_rate=body.heart_rate,
        systolic_bp=body.systolic_bp,
        diastolic_bp=body.diastolic_bp,
        blood_sugar=body.blood_sugar,
        ck_mb=body.ck_mb,
        troponin=body.troponin,
    )
    meta: dict[str, Any] = {"classifier_hint": clf}
    if REC_SIDECAR.exists():
        try:
            meta["sidecar"] = json.loads(REC_SIDECAR.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta["sidecar"] = None
    return {**out, "meta": meta}


@app.post("/api/hospital/predict")
def hospital_predict(body: HospitalIn) -> dict[str, Any]:
    if _hospital is None:
        raise HTTPException(status_code=503, detail="Hospital model not trained; run train_artifacts.py")
    names: list[str] = _hospital["feature_names"]
    if len(body.features) != len(names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(names)} features, got {len(body.features)}",
        )
    x = pd.DataFrame([body.features], columns=names)
    pred = float(_hospital["model"].predict(x)[0])
    return {"predicted_duration_days": pred}


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend not built")
    return FileResponse(index_path)


@app.get("/styles.css")
def styles_css() -> FileResponse:
    p = STATIC / "styles.css"
    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(p, media_type="text/css")


@app.get("/app.js")
def app_js() -> FileResponse:
    p = STATIC / "app.js"
    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(p, media_type="application/javascript")


@app.get("/config.js", response_model=None)
def config_js() -> Response:
    p = STATIC / "config.js"
    if p.exists():
        return FileResponse(p, media_type="application/javascript")
    return Response(
        content='window.ML_API_BASE = "";',
        media_type="application/javascript",
    )


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
