"""
Per-stone prediction trace.

Runs one real stone through the fitted models and records what each stage
actually computed, so the website can show measured intermediate values
rather than a narrated story. The API imports build_trace() for the live
version; running this file directly writes artifacts/trace.json.

    python trace.py [carat] [cut] [color] [clarity]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
SEGMENTS = ["Budget", "Mid-range", "Premium", "Luxury"]
GRADE_ENC = {
    "cut": {g: i for i, g in enumerate(CUT_ORDER)},
    "color": {g: i for i, g in enumerate(COLOR_ORDER)},
    "clarity": {g: i for i, g in enumerate(CLARITY_ORDER)},
}
NUM_FEATS = ["carat", "depth", "table", "x", "y", "z", "volume", "carat_per_volume", "log_carat"]
FEATS = NUM_FEATS + list(GRADE_ENC.keys())

# The dataset is 53,772 rows and the API is long lived, so parse it once and
# let the cache outlive the request. Loading per request added seconds.
_CACHE: dict = {}


def _load() -> dict:
    if "models" not in _CACHE:
        _CACHE["models"] = {
            "reg": joblib.load(ART / "reg_model.joblib"),
            "clf": joblib.load(ART / "clf_model.joblib"),
            "kmeans": joblib.load(ART / "kmeans_model.joblib"),
        }
        _CACHE["df"] = pd.read_csv(ART / "_full_clean.csv")
    return _CACHE


def build_trace(carat: float, cut: str, color: str, clarity: str) -> dict:
    """Walk one stone through the pipeline and return what each stage produced."""
    if carat <= 0:
        raise ValueError("carat must be positive")
    for value, order in ((cut, CUT_ORDER), (color, COLOR_ORDER), (clarity, CLARITY_ORDER)):
        if value not in order:
            raise ValueError(f"must be one of {', '.join(order)}, got {value!r}")

    cache = _load()
    reg, clf, kmeans = cache["models"]["reg"], cache["models"]["clf"], cache["models"]["kmeans"]
    df = cache["df"]

    # ---- Stage 1: get a real stone matching the description ---------------
    # The user states a weight and three grades, but the model also wants
    # x/y/z/depth/table. Two measured reasons to borrow a real stone rather
    # than synthesise dimensions from textbook diamond physics:
    #
    #  1. Volume tracks carat at 163.5 mm3 per carat in this data (p10 159.4,
    #     p90 167.0), not the 35.9 mm3 that crystal density at 3.52 g/cm3
    #     predicts. x/y/z here are girdle measurements, not a bounding box,
    #     so the physics does not apply.
    #  2. Proportions barely move the answer (about $7,594 to $7,826 across a
    #     wide depth/table sweep), so proportions are not the risk. Choosing
    #     the wrong comparison group is, which stage 5 handles.
    pool = df[(df["carat"] >= carat * 0.9) & (df["carat"] <= carat * 1.1)]
    if len(pool) == 0:
        pool = df.iloc[(df["carat"] - carat).abs().argsort()[:200]]
    exact = pool[(pool["cut"] == cut) & (pool["color"] == color) & (pool["clarity"] == clarity)]
    pick = exact if len(exact) >= 5 else pool
    # Median-priced stone of the group, so the walkthrough is not cherry-picked.
    med_price = pick["price"].median()
    src = pick.iloc[(pick["price"] - med_price).abs().argsort()[:1]].iloc[0]

    x, y, z = float(src["x"]), float(src["y"]), float(src["z"])
    depth = round(float(src["depth"]), 1)
    table = round(float(src["table"]), 1)
    source_price = int(src["price"])

    raw = {
        "carat": carat, "cut": cut, "color": color, "clarity": clarity,
        "depth": depth, "table": table, "x": x, "y": y, "z": z,
    }

    # ---- Stage 2: feature engineering -------------------------------------
    volume = round(x * y * z, 2)
    cpv = round(carat / volume, 6)
    log_carat = round(float(np.log1p(carat)), 4)
    derived = {
        "volume": {"value": volume, "formula": "x * y * z",
                   "note": f"{x} x {y} x {z} mm"},
        "carat_per_volume": {"value": cpv, "formula": "carat / volume",
                             "note": "compactness: how tightly the weight is packed"},
        "log_carat": {"value": log_carat, "formula": "ln(1 + carat)",
                      "note": "tames the long right tail of weight"},
    }

    frame = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table, "x": x, "y": y, "z": z,
        "volume": volume, "carat_per_volume": cpv, "log_carat": log_carat,
        "cut": GRADE_ENC["cut"][cut],
        "color": GRADE_ENC["color"][color],
        "clarity": GRADE_ENC["clarity"][clarity],
    }])[FEATS]

    # ---- Stage 3: ordinal encoding ----------------------------------------
    enc = [
        {"feature": "cut", "value": f"{cut} -> {GRADE_ENC['cut'][cut]}",
         "why": "grades are ordered, so the order is kept instead of 5 separate columns"},
        {"feature": "color", "value": f"{color} -> {GRADE_ENC['color'][color]}",
         "why": f"J is 0 through D being {len(COLOR_ORDER) - 1}"},
        {"feature": "clarity", "value": f"{clarity} -> {GRADE_ENC['clarity'][clarity]}",
         "why": f"I1 is 0 through IF being {len(CLARITY_ORDER) - 1}"},
    ]

    # ---- Stage 4: how a tree model actually reasons ------------------------
    leaf = reg.apply(frame)[0]
    tree_imp = reg.feature_importances_
    contrib = [
        {"feature": FEATS[i], "importance": round(float(tree_imp[i]), 4)}
        for i in np.argsort(tree_imp)[::-1][:4]
    ]
    same_leaf = df.iloc[np.asarray(leaf)]
    near = same_leaf.nsmallest(5, "carat")[["carat", "cut", "color", "clarity", "price"]]
    neighbours = [
        {"carat": round(float(r.carat), 2), "cut": r.cut, "color": r.color,
         "clarity": r.clarity, "price": round(float(r.price))}
        for r in near.itertuples()
    ]

    price = float(reg.predict(frame)[0])
    seg_code = int(clf.predict(frame)[0])
    probs = clf.predict_proba(frame)[0]

    cluster_frame = pd.DataFrame([{"carat": carat, "volume": volume,
                                   "depth": depth, "table": table}])[kmeans["features"]]
    cluster = int(kmeans["model"].predict(
        kmeans["scaler"].transform(cluster_frame))[0])

    # Same stone, all three grades flattened to worst. The gap is what the
    # quality grades added on top of weight alone.
    flat = frame.copy()
    for c in ("cut", "color", "clarity"):
        flat[c] = 0
    flat_price = float(reg.predict(flat)[0])

    # ---- Stage 5: reality check -------------------------------------------
    # Two rules, both learned the hard way:
    #
    #  1. Match the grade, or the gap measures the grade instead of accuracy.
    #  2. Match the WEIGHT tightly, or the gap measures the weight instead.
    #     A 0.63-0.77ct band for a 0.7ct stone holds 5,995 stones and its
    #     median describes something much smaller and cheaper. A +97% "error"
    #     was entirely an artefact of that too-wide window.
    #
    # So: narrowest window that still holds enough stones, then relax the
    # grades one at a time, and finally return null rather than print a
    # number that means nothing.
    comp: dict = {"n": 0, "median": None, "tol": None, "label": "", "rows": None}
    for tol in (0.01, 0.02, 0.05, 0.10, 0.20):
        s = df[(abs(df["carat"] - carat) <= tol) & (df["cut"] == cut) &
               (df["color"] == color) & (df["clarity"] == clarity)]
        if len(s) >= 8:
            comp = {"n": len(s), "median": float(s["price"].median()), "tol": tol,
                    "label": f"{cut} / {color} / {clarity}", "rows": s}
            break
    if comp["median"] is None:  # relax clarity, keep cut and colour
        for tol in (0.05, 0.10, 0.20):
            s = df[(abs(df["carat"] - carat) <= tol) & (df["cut"] == cut) &
                   (df["color"] == color)]
            if len(s) >= 8:
                comp = {"n": len(s), "median": float(s["price"].median()), "tol": tol,
                        "label": f"{cut} / {color}, any clarity", "rows": s}
                break
    if comp["median"] is None:  # just the weight band
        s = df[abs(df["carat"] - carat) <= 0.10]
        if len(s) >= 8:
            comp = {"n": len(s), "median": float(s["price"].median()), "tol": 0.10,
                    "label": "any grade at this weight", "rows": s}

    market_median = round(comp["median"]) if comp["median"] is not None else None
    error_pct = (round((price / market_median - 1) * 100, 1)
                 if market_median else None)

    return {
        "input": raw,
        "derived": derived,
        "encoding": enc,
        "tree": {
            "leaf_size": int(len(same_leaf)),
            "leaf_median_price": round(float(same_leaf["price"].median())),
            "top_features": contrib,
            "neighbours": neighbours,
        },
        "result": {
            "predicted_price": round(price, 2),
            "price_per_carat": round(price / carat, 2),
            "lowest_quality_price": round(flat_price, 2),
            "quality_premium": round(price - flat_price, 2),
            "predicted_segment": SEGMENTS[seg_code],
            "segment_probabilities": {
                SEGMENTS[i]: round(float(probs[i]), 4) for i in range(4)
            },
            "cluster": cluster,
        },
        "reality_check": {
            "source_stone_price": source_price,
            "market_median_price": market_median,
            "comparable_stones": comp["n"],
            "comparison_group": comp["label"],
            "carat_tolerance": comp["tol"],
            "error_pct": error_pct,
            "price_spread": (
                [round(float(comp["rows"]["price"].quantile(0.1))),
                 round(float(comp["rows"]["price"].quantile(0.9)))]
                if comp["rows"] is not None else None
            ),
        },
    }


def main() -> None:
    args = sys.argv[1:]
    payload = build_trace(
        float(args[0]) if len(args) > 0 else 1.2,
        args[1] if len(args) > 1 else "Very Good",
        args[2] if len(args) > 2 else "G",
        args[3] if len(args) > 3 else "VS2",
    )
    (ART / "trace.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {ART / 'trace.json'}")


if __name__ == "__main__":
    main()
