"""
FastAPI model server for the diamond pricing dashboard.

    uvicorn api:app --reload --port 8000

Endpoints
    GET  /health          liveness + loaded model info
    GET  /summary         the full analysis payload for the dashboard
    GET  /charts/{name}   a generated PNG from artifacts/charts
    POST /predict         predicted price + predicted value segment
    GET  /sample          a real row from the dataset, for demo buttons
    GET  /trace           step-by-step trace of how one price is produced
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from trace import build_trace

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
CHARTS = ART / "charts"

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
GRADE_ENC = {
    "cut": {g: i for i, g in enumerate(CUT_ORDER)},
    "color": {g: i for i, g in enumerate(COLOR_ORDER)},
    "clarity": {g: i for i, g in enumerate(CLARITY_ORDER)},
}
SEGMENTS = ["Budget", "Mid-range", "Premium", "Luxury"]
NUM_FEATS = ["carat", "depth", "table", "x", "y", "z", "volume", "carat_per_volume", "log_carat"]
REG_FEATS = NUM_FEATS + list(GRADE_ENC.keys())

app = FastAPI(title="Diamond Pricing API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_reg = None
_clf = None
_kmeans = None
_summary: dict = {}


@app.on_event("startup")
def load() -> None:
    global _reg, _clf, _kmeans, _summary
    _reg = joblib.load(ART / "reg_model.joblib")
    _clf = joblib.load(ART / "clf_model.joblib")
    _kmeans = joblib.load(ART / "kmeans_model.joblib")
    _summary = json.loads((ART / "dashboard_data.json").read_text(encoding="utf-8"))
    print("Models loaded.")


class PredictIn(BaseModel):
    carat: float = Field(..., gt=0, le=20, description="Weight in carats")
    cut: str
    color: str
    clarity: str
    depth: float = Field(..., gt=0, le=100)
    table: float = Field(..., gt=0, le=100)
    x: float = Field(..., gt=0, le=60, description="Length, mm")
    y: float = Field(..., gt=0, le=60, description="Width, mm")
    z: float = Field(..., gt=0, le=60, description="Depth, mm")

    def validate_grades(self) -> None:
        for field, order in (("cut", CUT_ORDER), ("color", COLOR_ORDER), ("clarity", CLARITY_ORDER)):
            if getattr(self, field) not in order:
                raise HTTPException(422, f"{field} must be one of {', '.join(order)}")


def to_frame(p: PredictIn) -> pd.DataFrame:
    """Reproduce exactly the feature engineering done in train.py."""
    volume = p.x * p.y * p.z
    return pd.DataFrame([{
        "carat": p.carat,
        "depth": p.depth,
        "table": p.table,
        "x": p.x,
        "y": p.y,
        "z": p.z,
        "volume": volume,
        "carat_per_volume": p.carat / volume if volume else np.nan,
        "log_carat": np.log1p(p.carat),
        "cut": GRADE_ENC["cut"][p.cut],
        "color": GRADE_ENC["color"][p.color],
        "clarity": GRADE_ENC["clarity"][p.clarity],
    }])[REG_FEATS]


@app.get("/health")
def health() -> dict:
    best = _summary.get("regression", {})
    return {
        "status": "ok" if _reg is not None else "loading",
        "regression_model": best.get("best_model"),
        "regression_r2": round(best.get("best", {}).get("r2", 0), 4),
        "classification_model": _summary.get("classification", {}).get("best_model"),
        "classification_f1": round(_summary.get("classification", {}).get("f1_weighted", 0), 4),
        "rows": _summary.get("meta", {}).get("rows"),
    }


@app.get("/summary")
def summary() -> dict:
    return _summary


@app.get("/charts/{name}")
def chart(name: str) -> FileResponse:
    if "/" in name or ".." in name:
        raise HTTPException(400, "Bad chart name")
    path = CHARTS / name
    if not path.exists():
        raise HTTPException(404, f"No chart named {name}")
    return FileResponse(path, media_type="image/png")


@app.get("/sample")
def sample() -> dict:
    """A real dataset row, so the demo buttons show genuine predictions."""
    df = pd.read_csv(ART / "_full_clean.csv")
    row = df.sample(n=1, random_state=7).iloc[0]
    return {
        "carat": float(row.carat), "cut": row.cut, "color": row.color,
        "clarity": row.clarity, "depth": float(row.depth), "table": float(row.table),
        "x": float(row.x), "y": float(row.y), "z": float(row.z),
        "actual_price": float(row.price), "actual_segment": row.segment,
    }


@app.post("/predict")
def predict(p: PredictIn) -> dict:
    p.validate_grades()
    frame = to_frame(p)

    price = float(_reg.predict(frame)[0])
    seg_code = int(_clf.predict(frame)[0])
    probs = _clf.predict_proba(frame)[0]
    prob_map = {SEGMENTS[i]: round(float(probs[i]), 4) for i in range(4)}

    # Map the raw inputs onto the same standardised space the clusterer used.
    km = _kmeans
    cluster_frame = pd.DataFrame([{
        "carat": p.carat, "volume": frame["volume"][0],
        "depth": p.depth, "table": p.table,
    }])[km["features"]]
    cluster = int(km["model"].predict(km["scaler"].transform(cluster_frame))[0])

    return {
        "predicted_price": round(price, 2),
        "price_per_carat": round(price / p.carat, 2),
        "predicted_segment": SEGMENTS[seg_code],
        "segment_probabilities": prob_map,
        "cluster": cluster,
        "cluster_profile": next(
            (c for c in _summary["clustering"]["profiles"] if c["cluster"] == cluster), None
        ),
        "input": p.model_dump(),
    }


@app.get("/trace")
def trace(
    carat: float = Query(1.2, gt=0, le=20),
    cut: str = Query("Very Good"),
    color: str = Query("G"),
    clarity: str = Query("VS2"),
) -> dict:
    """Walk one stone through the pipeline, returning each stage's real output.

    The website renders this directly, so every number on screen is measured
    rather than narrated. The logic lives in trace.py so the same code backs
    both the CLI and the API.
    """
    try:
        return build_trace(carat, cut, color, clarity)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
