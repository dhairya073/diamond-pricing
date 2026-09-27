"""
Export deployable model artifacts for the browser.

The trained scikit-learn artifacts cannot ship: the regressor is 631 MB and
the classifier 156 MB, both far over GitHub's 100 MB per-file limit and over
Vercel's 250 MB serverless bundle limit. Two things make a static deploy
possible instead:

  1. A depth-capped forest is dramatically smaller for almost no accuracy
     cost. Measured: 80 trees at depth 10 gives R2 0.9808 / MAE $292 against
     the full forest's 0.9824 / $262, for 66x less data.
  2. Random forest inference is just tree traversal. It ports to TypeScript
     in a few dozen lines, so no Python process is needed at all.

This writes web/public/model/:
    reg.json         compact regressor forest
    clf.json         compact classifier forest + class labels
    kmeans.json      scaler params and centroids
    reference.json   the cleaned dataset, for the walkthrough's comparisons
    summary.json     the static analysis payload

    python export_web.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
OUT = ROOT / "web" / "public" / "model"
CHARTS_OUT = ROOT / "web" / "public" / "charts"

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
SEGMENTS = ["Budget", "Mid-range", "Premium", "Luxury"]
FEATS = ["carat", "depth", "table", "x", "y", "z", "volume",
         "carat_per_volume", "log_carat", "cut", "color", "clarity"]
CLUSTER_FEATS = ["carat", "volume", "depth", "table"]

# Deployment model size. Measured against the full forest in measure_compact:
# depth 10 / 80 trees holds R2 within 0.002 while exporting to a few MB.
N_TREES = 80
MAX_DEPTH = 10
SEED = 42


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["volume"] = out["x"] * out["y"] * out["z"]
    out["carat_per_volume"] = out["carat"] / out["volume"]
    out["log_carat"] = np.log1p(out["carat"])
    out["cut"] = out["cut"].map({g: i for i, g in enumerate(CUT_ORDER)})
    out["color"] = out["color"].map({g: i for i, g in enumerate(COLOR_ORDER)})
    out["clarity"] = out["clarity"].map({g: i for i, g in enumerate(CLARITY_ORDER)})
    return out


def export_forest(est) -> dict:
    """Flatten a forest to parallel arrays.

    One set of arrays per tree rather than an object per node: repeating
    "feature"/"threshold" keys on ~137k nodes costs more in JSON than the
    numbers do. Leaves are marked with left == -1, matching sklearn.
    """
    trees = []
    for t in est.estimators_:
        s = t.tree_
        left = s.children_left.astype(int).tolist()
        right = s.children_right.astype(int).tolist()
        # value shape is (n_nodes, n_outputs, n_classes). Regressors have one
        # output and one value per node; classifiers need the whole vector.
        val = s.value
        if val.ndim == 3 and val.shape[1] == 1 and val.shape[2] > 1:
            vals = np.round(val[:, 0, :], 6).tolist()
        else:
            vals = np.round(val[:, 0, 0], 4).tolist()
        trees.append({
            "f": s.feature.astype(int).tolist(),
            "t": np.round(s.threshold, 5).tolist(),
            "l": left,
            "r": right,
            "v": vals,
        })
    return {"trees": trees, "n_features": int(est.n_features_in_)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ART / "_full_clean.csv")
    enc = encode(df)
    X, y = enc[FEATS], df["price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"rows {len(df):,}  train {len(Xtr):,}  test {len(Xte):,}\n")

    # ---- regressor ------------------------------------------------------
    print(f"training regressor: {N_TREES} trees, max_depth {MAX_DEPTH}")
    reg = RandomForestRegressor(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=SEED,
        n_jobs=-1, min_samples_leaf=2,
    )
    reg.fit(Xtr, ytr)
    rp = reg.predict(Xte)
    print(f"  R2 {r2_score(yte, rp):.4f}  MAE ${mean_absolute_error(yte, rp):,.0f}")
    reg_json = export_forest(reg)
    (OUT / "reg.json").write_text(json.dumps(reg_json, separators=(",", ":")), encoding="utf-8")
    print(f"  exported {[p for p in ['reg.json']][0]}\n")

    # ---- classifier -----------------------------------------------------
    # Same depth cap. Price is deliberately absent from the feature list:
    # the tiers are cut from price, so including it would be leakage.
    ytr_seg = pd.qcut(df.loc[Xtr.index, "price"], 4, labels=False)
    yte_seg = pd.qcut(df.loc[Xte.index, "price"], 4, labels=False)
    print(f"training classifier: {N_TREES} trees, max_depth {MAX_DEPTH}")
    clf = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=SEED,
        n_jobs=-1, min_samples_leaf=2,
    )
    clf.fit(Xtr, ytr_seg)
    cp = clf.predict(Xte)
    acc = accuracy_score(yte_seg, cp)
    f1 = f1_score(yte_seg, cp, average="weighted")
    print(f"  accuracy {acc*100:.2f}%  weighted F1 {f1:.4f}")
    clf_json = export_forest(clf)
    clf_json["classes"] = [int(c) for c in clf.classes_]
    clf_json["segments"] = SEGMENTS
    (OUT / "clf.json").write_text(json.dumps(clf_json, separators=(",", ":")), encoding="utf-8")

    # ---- clustering -----------------------------------------------------
    print("\ntraining k-means on size and proportion only, k=2")
    km_frame = pd.DataFrame({
        "carat": df["carat"], "volume": df["x"] * df["y"] * df["z"],
        "depth": df["depth"], "table": df["table"],
    })[CLUSTER_FEATS]
    scaler = StandardScaler().fit(km_frame)
    km = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(scaler.transform(km_frame))
    km_json = {
        "features": CLUSTER_FEATS,
        "mean": [round(float(v), 8) for v in scaler.mean_],
        "scale": [round(float(v), 8) for v in scaler.scale_],
        "centroids": [[round(float(v), 8) for v in c] for c in km.cluster_centers_],
    }
    (OUT / "kmeans.json").write_text(json.dumps(km_json, separators=(",", ":")), encoding="utf-8")

    # ---- reference data for the walkthrough -----------------------------
    # The walkthrough needs real stones: to borrow dimensions, to show which
    # stones landed in the same leaf, and to build an honest comparison group.
    # Array-of-arrays with integer grade codes keeps this around 2 MB, which
    # gzips to under 1 MB, and it is only fetched when that section is opened.
    ref = [
        [
            round(float(r.carat), 2),
            int(CUT_ORDER.index(r.cut)),
            int(COLOR_ORDER.index(r.color)),
            int(CLARITY_ORDER.index(r.clarity)),
            int(r.price),
            round(float(r.depth), 1),
            round(float(r.table), 1),
            round(float(r.x), 2),
            round(float(r.y), 2),
            round(float(r.z), 2),
        ]
        for r in df.itertuples()
    ]
    (OUT / "reference.json").write_text(json.dumps({
        "fields": ["carat", "cut", "color", "clarity", "price",
                   "depth", "table", "x", "y", "z"],
        "cut_order": CUT_ORDER,
        "color_order": COLOR_ORDER,
        "clarity_order": CLARITY_ORDER,
        "rows": ref,
    }, separators=(",", ":")), encoding="utf-8")

    # ---- static analysis payload and charts -----------------------------
    # The summary and the sample stones are imported as modules by the page,
    # so a Server Component can render them with no network call and no base
    # URL to resolve. Only the models and the reference data are fetched.
    DATA_OUT = ROOT / "web" / "data"
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(ART / "dashboard_data.json", DATA_OUT / "summary.json")

    # A fixed, varied set of stones for the "show me one" button. Committing
    # these means that button does not have to pull the 2.3 MB reference file.
    sample_rows = (
        df.sample(n=min(24, len(df)), random_state=SEED)
        .sort_values("carat")
        .reset_index(drop=True)
    )
    samples = [
        {
            "carat": float(r.carat), "cut": r.cut, "color": r.color,
            "clarity": r.clarity, "depth": float(r.depth), "table": float(r.table),
            "x": float(r.x), "y": float(r.y), "z": float(r.z),
            "actual_price": int(r.price),
        }
        for r in sample_rows.itertuples()
    ]
    (DATA_OUT / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    if CHARTS_OUT.exists():
        shutil.rmtree(CHARTS_OUT)
    shutil.copytree(ART / "charts", CHARTS_OUT)

    print("\nwritten:")
    total = 0
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            mb = p.stat().st_size / 1e6
            total += mb
            print(f"  {p.name:<18} {mb:6.2f} MB")
    print(f"  charts/            {sum(f.stat().st_size for f in CHARTS_OUT.iterdir())/1e6:6.2f} MB")
    print(f"\n  model payload total {total:.2f} MB (was 787 MB of joblib)")
    print("\n  metrics the browser model actually achieves:")
    print(f"    regressor  R2 {r2_score(yte, rp):.4f}  MAE ${mean_absolute_error(yte, rp):,.0f}")
    print(f"    classifier accuracy {acc*100:.2f}%  F1 {f1:.4f}")


if __name__ == "__main__":
    main()
