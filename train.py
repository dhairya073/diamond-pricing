"""
Diamond Pricing & Value Segmentation - full training pipeline.

Run:  python train.py

Outputs (all under artifacts/):
    reg_model.joblib       price regressor (best by RMSE)
    reg_metrics.json       baseline + tuned metrics
    clf_model.joblib       value-segment classifier
    clf_metrics.json       classification report + confusion matrix
    kmeans_model.joblib    k-means clusterer
    kmeans_metrics.json    silhouette score + cluster profiles
    feature_importance.csv top feature importances
    importances.json       chart-ready importance data
    segment_summary.csv    per-segment statistics
    charts/*.png           EDA figures used by the dashboard
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "diamonds.csv"
ART = ROOT / "artifacts"
CHARTS = ART / "charts"
ART.mkdir(exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette
# One accent (deep teal) + neutral greys, locked across every chart so the
# dashboard and the report read as one system. Charts are always drawn on a
# white ground: the dashboard gives them a deliberate light plate in both
# colour schemes rather than shipping two sets of PNGs, which would double the
# training run for a purely cosmetic gain.
ACCENT = "#0F766E"
INK = "#1C1917"
MUTED = "#57534E"
WARN = "#B45309"
FIG_BG = "#FFFFFF"

sns.set_theme(style="whitegrid", font_scale=1.05)
sns.set_palette([ACCENT, "#0E7490", "#B45309", "#7C3AED", "#BE123C", "#4D7C0F"])


def save(fig, name: str) -> None:
    """Write a chart and close it, so no call site can leak a figure."""
    fig.savefig(CHARTS / name, dpi=140, facecolor=FIG_BG, edgecolor="none")
    plt.close(fig)


def money(v: float) -> str:
    return f"${v:,.0f}"


# ================================================================ 1. LOAD
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df

# ================================================================ 2. CLEAN
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop data-entry errors and unusable records. Returns a cleaning log."""
    log: dict = {"raw_rows": int(len(df))}

    log["missing_cells"] = int(df.isna().sum().sum())

    # Physical dimensions of 0 are measurement errors, not real stones.
    zero_dims = int(((df[["x", "y", "z"]] <= 0).any(axis=1)).sum())
    log["zero_or_negative_dims"] = zero_dims

    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

    # A diamond whose depth is 0 after cleaning is not measurable.
    df = df[(df["depth"] > 0) & (df["table"] > 0)]

    # Physically implausible: a stone longer than 30mm in any axis.
    extreme = int((df[["x", "y", "z"]] > 30).any(axis=1).sum())
    log["extreme_dimensions_gt_30mm"] = extreme
    df = df[(df[["x", "y", "z"]] <= 30).all(axis=1)]

    # Duplicate rows in the source file.
    dupes = int(df.duplicated().sum())
    log["duplicate_rows"] = dupes
    df = df.drop_duplicates()

    log["clean_rows"] = int(len(df))
    log["removed_rows"] = log["raw_rows"] - log["clean_rows"]
    print(f"Cleaning: {log['raw_rows']:,} -> {log['clean_rows']:,} rows")
    return df.reset_index(drop=True), log


# ================================================================ 3. EDA
def run_eda(df: pd.DataFrame, clean_log: dict) -> dict:
    """Charts plus the numbers quoted in the report narrative."""
    stats: dict = {"cleaning": clean_log, "correlations": {}}

    # ---- price / carat skew -------------------------------------------------
    p_skew = float(df["price"].skew())
    c_skew = float(df["carat"].skew())
    stats["skew"] = {"price": p_skew, "carat": c_skew}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, color in zip(axes, ["price", "carat"], [ACCENT, "#B45309"]):
        sns.histplot(data=df, x=col, ax=ax, bins=45, color=color, kde=True, line_kws={"lw": 1.4})
        ax.set_xlabel(col)
        ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title(f"Price distribution (skew {p_skew:.2f})", fontsize=11, color=INK)
    axes[1].set_title(f"Carat distribution (skew {c_skew:.2f})", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "01_distributions.png")

    # ---- carat vs price (log-log shows the power law) -----------------------
    # 53k points in one scatter reads as a solid block. Bin the two log axes
    # and shade by density instead, so the shape of the cloud is visible.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    hb = ax.hexbin(
        np.log10(df["carat"]), np.log10(df["price"]),
        gridsize=55, cmap=sns.light_palette(ACCENT, as_cmap=True), mincnt=1, linewidths=0,
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("diamonds per bin", fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK)
    cb.outline.set_visible(False)
    # Overlay the fitted power law so the reader can see the residual spread.
    xs = np.linspace(df["carat"].min(), df["carat"].max(), 60)
    slope, intercept = np.polyfit(
        np.log(df["carat"]), np.log(df["price"]), 1
    )
    ax.plot(np.log10(xs), intercept + slope * np.log(xs), color=WARN, lw=1.4, ls="--",
            label=f"fitted power law, slope {slope:.2f}")
    ax.set_xlabel("Carat (log10 scale)")
    ax.set_ylabel("Price, USD (log10 scale)")
    ax.set_title("Price scales roughly as a power of carat", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "02_carat_price.png")

    # r and r^2 for carat alone, on raw and log scales
    r_raw = float(np.corrcoef(df["carat"], df["price"])[0, 1])
    r_log = float(np.corrcoef(np.log(df["carat"]), np.log(df["price"]))[0, 1])
    stats["carat_price_r_raw"] = r_raw
    stats["carat_price_r_log"] = r_log
    # elasticity of price with respect to carat
    stats["price_carat_elasticity"] = float(
        np.polyfit(np.log(df["carat"]), np.log(df["price"]), 1)[0]
    )

    # ---- quality grades ----------------------------------------------------
    order_cut = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    order_color = ["J", "I", "H", "G", "F", "E", "D"]
    order_clarity = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, col, order in zip(axes, ["cut", "color", "clarity"], [order_cut, order_color, order_clarity]):
        sns.boxplot(data=df, x=col, y="price", order=order, ax=ax, color=ACCENT, fliersize=1.2,
                    linewidth=0.9, saturation=0.55)
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("Price, USD (log)" if col == "cut" else "")
        ax.set_title(col.capitalize(), fontsize=11, color=INK)
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
    fig.suptitle("Median price moves with quality grade, spread stays wide at every grade",
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "03_quality_boxplots.png")

    # median price per grade, for the report
    grade_tables = {}
    for col, order in [("cut", order_cut), ("color", order_color), ("clarity", order_clarity)]:
        med = df.groupby(col)["price"].median().reindex(order).dropna()
        cnt = df.groupby(col).size().reindex(order).dropna()
        grade_tables[col] = {
            "order": list(med.index),
            "median_price": [float(v) for v in med.values],
            "count": [int(v) for v in cnt.values],
        }
    stats["grade_tables"] = grade_tables

    # ---- correlation heatmap ----------------------------------------------
    num = df[["carat", "depth", "table", "x", "y", "z", "price"]]
    corr = num.corr()
    stats["correlations"] = {
        "carat_price": float(corr.loc["carat", "price"]),
        "xy_carat": float(corr.loc["x", "carat"]),
        "depth_price": float(corr.loc["depth", "price"]),
        "table_price": float(corr.loc["table", "price"]),
        "z_price": float(corr.loc["z", "price"]),
        "matrix": corr.round(3).to_dict(),
    }

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.light_palette(ACCENT, as_cmap=True),
                ax=ax, square=True, linewidths=0.6, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_title("Numeric correlations", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "04_correlation_heatmap.png")

    # ---- outliers: price per carat -----------------------------------------
    ppc = df["price"] / df["carat"]
    stats["price_per_carat"] = {
        "median": float(ppc.median()),
        "p05": float(ppc.quantile(0.05)),
        "p95": float(ppc.quantile(0.95)),
        "max": float(ppc.max()),
        "skew": float(ppc.skew()),
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    hb = ax.hexbin(
        np.log10(df["carat"]), np.log10(ppc),
        gridsize=50, cmap=sns.light_palette(ACCENT, as_cmap=True), mincnt=1, linewidths=0,
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("diamonds per bin", fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK)
    cb.outline.set_visible(False)
    ax.axhline(np.log10(ppc.median()), color=WARN, lw=1.3, ls="--",
               label=f"median {money(ppc.median())}/ct")
    ax.set_xlabel("Carat (log10 scale)")
    ax.set_ylabel("USD per carat (log10 scale)")
    ax.set_title("Price per carat separates large stones from small ones",
                 fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "05_price_per_carat.png")

    # ---- grade effect once size is held constant ---------------------------
    # Raw median price per grade is misleading: better-graded stones are also
    # bigger on average, and size dominates price. Splitting the data into
    # carat quintiles and comparing price PER CARAT inside each quintile
    # isolates the grade effect from the weight effect.
    df = df.assign(price_per_carat=df["price"] / df["carat"])
    df["carat_band"] = pd.qcut(df["carat"], 5, labels=False, duplicates="drop")
    size_controlled: dict = {"bands": int(df["carat_band"].nunique())}
    for col, order in [("cut", order_cut), ("color", order_color), ("clarity", order_clarity)]:
        piv = (
            df.pivot_table(index=col, columns="carat_band", values="price_per_carat", aggfunc="median")
            .reindex(order)
        )
        worst_over_best = (piv.iloc[0] / piv.iloc[-1]).round(4)
        size_controlled[col] = {
            "worst_grade": order[0],
            "best_grade": order[-1],
            "ratio_by_band": {str(int(k)): float(v) for k, v in worst_over_best.items()},
            "ratio_mean": float(worst_over_best.mean()),
            "ratio_largest_band": float(worst_over_best.iloc[-1]),
        }
    stats["size_controlled_grade_effect"] = size_controlled

    # ---- chart: grade premium by carat band --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, col, order in zip(axes, ["cut", "color", "clarity"], [order_cut, order_color, order_clarity]):
        piv = (
            df.pivot_table(index=col, columns="carat_band", values="price_per_carat", aggfunc="median")
            .reindex(order)
        )
        ratio = (piv.iloc[0] / piv.iloc[-1] * 100).round(1)
        ax.plot(range(len(ratio)), ratio.values, marker="o", color=ACCENT, lw=1.8)
        ax.set_xticks(range(len(ratio)))
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(ratio))])
        ax.set_xlabel("Carat quintile, small to large")
        if col == "cut":
            ax.set_ylabel("Worst grade as % of best")
        else:
            ax.set_ylabel("")
        worst, best = size_controlled[col]["worst_grade"], size_controlled[col]["best_grade"]
        ax.set_title(f"{col.capitalize()}: {worst} vs {best}", fontsize=11, color=INK)
        ax.grid(alpha=0.25)
    fig.suptitle("Grade premium widens as stones get larger, once size is held constant",
                 fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "11_grade_effect_by_size.png")

    # ---- class distribution preview ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    q = pd.qcut(df["price"], 4, labels=["Budget", "Mid-range", "Premium", "Luxury"])
    counts = q.value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color=ACCENT)
    ax.set_xlabel("Price quartile segment")
    ax.set_ylabel("Diamonds")
    ax.set_title("Segment sizes are equal by construction (quartiles)", fontsize=12, color=INK)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 250, f"{v:,}", ha="center", fontsize=9, color=MUTED)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "06_segment_sizes.png")

    return stats


# ============================================== 4. FEATURE ENGINEERING
ORDERS = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}
SEGMENTS = ["Budget", "Mid-range", "Premium", "Luxury"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns shared by the regressor, the classifier and the API."""
    d = df.copy()
    # Physical volume in cubic millimetres. Carat is weight, so two stones of
    # equal weight can have very different volume depending on cut depth.
    d["volume"] = d["x"] * d["y"] * d["z"]
    # Weight-to-volume ratio: a compactness proxy that separates well-cut
    # stones from deep or shallow ones.
    d["carat_per_volume"] = d["carat"] / d["volume"].replace(0, np.nan)
    # log1p keeps the long right tail of price from dominating squared error.
    d["log_price"] = np.log1p(d["price"])
    d["log_carat"] = np.log1p(d["carat"])
    d = d.replace([np.inf, -np.inf], np.nan)
    return d


# ================================================================ 5. REG
GRADE_ENC = {c: {g: i for i, g in enumerate(ORDERS[c])} for c in ORDERS}
NUM_FEATS = ["carat", "depth", "table", "x", "y", "z", "volume", "carat_per_volume", "log_carat"]
REG_FEATS = NUM_FEATS + list(ORDERS.keys())


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode the three quality grades, lowest grade = 0.

    Preserves the natural ranking (Fair < Good < ... < Ideal, J < ... < D,
    I1 < ... < IF) which one-hot encoding would throw away.
    """
    out = df[NUM_FEATS].copy()
    for c, mapping in GRADE_ENC.items():
        out[c] = df[c].map(mapping)
    return out


def run_regression(df: pd.DataFrame, eda: dict) -> tuple[Pipeline, dict]:
    X = encode(df)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"Train {len(X_train):,} / test {len(X_test):,} rows")

    metrics: dict = {"seed": SEED, "n_train": int(len(X_train)), "n_test": int(len(X_test))}

    def score(name: str, model, Xte: pd.DataFrame, yte: pd.Series) -> tuple:
        pred = model.predict(Xte)
        mae = float(mean_absolute_error(yte, pred))
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        r2 = float(r2_score(yte, pred))
        cv = cross_val_score(model, X_train, y_train, cv=3,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
        res = {"model": name, "mae": mae, "rmse": rmse, "r2": r2,
               "cv_rmse_mean": float(-cv.mean()), "cv_rmse_std": float(cv.std())}
        print(f"  {name:<26} MAE {money(mae):>10}  RMSE {money(rmse):>10}  R2 {r2:.4f}")
        return res, pred

    # ---- baseline: linear regression on the raw target ---------------------
    lin = LinearRegression().fit(X_train, y_train)
    lin_res, lin_pred = score("LinearRegression", lin, X_test, y_test)
    metrics["linear"] = lin_res

    # Same coefficients read on a log target: each coefficient becomes the
    # percent change in price for a one-unit change in the feature.
    log_lin = LinearRegression().fit(X_train, np.log(y_train))
    metrics["linear_coefficients"] = {n: float(c) for n, c in zip(REG_FEATS, log_lin.coef_)}
    metrics["linear_r2_log_target"] = float(log_lin.score(X_test, np.log(y_test)))
    metrics["linear_coef_note"] = (
        "Coefficients from a regression of log(price) on the same features. They read as "
        "roughly proportional price effects, which is why the raw-target fit is far worse."
    )

    # ---- stronger models ---------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=SEED
    )
    rf.fit(X_train, y_train)
    rf_res, rf_pred = score("RandomForest", rf, X_test, y_test)
    metrics["random_forest"] = rf_res

    gb = GradientBoostingRegressor(
        random_state=SEED, n_estimators=300, max_depth=3, learning_rate=0.06
    )
    gb.fit(X_train, y_train)
    gb_res, gb_pred = score("GradientBoosting", gb, X_test, y_test)
    metrics["gradient_boosting"] = gb_res

    best_name = min(["random_forest", "gradient_boosting"], key=lambda k: metrics[k]["rmse"])
    best_pred = rf_pred if best_name == "random_forest" else gb_pred
    best_model = rf if best_name == "random_forest" else gb
    metrics["best_model"] = best_name
    metrics["best"] = metrics[best_name]

    # ---- predicted vs actual + residuals ----------------------------------
    # Both panels bin on the log price axis. 10k alpha-blended dots still read
    # as a solid block, which hides the structure the caption is describing.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    lim = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]

    hb = axes[0].hexbin(
        np.log10(y_test.values), np.log10(np.clip(best_pred, 1, None)),
        gridsize=55, cmap=sns.light_palette(ACCENT, as_cmap=True), mincnt=1, linewidths=0,
    )
    cb = fig.colorbar(hb, ax=axes[0], pad=0.02)
    cb.set_label("test stones per bin", fontsize=8.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK)
    cb.outline.set_visible(False)
    axes[0].plot(np.log10(lim), np.log10(lim), color=WARN, lw=1.3, ls="--")
    axes[0].set_xlabel("Actual price (log10)")
    axes[0].set_ylabel("Predicted price (log10)")
    axes[0].set_title(f"Predicted vs actual ({best_name})", fontsize=11, color=INK)
    axes[0].grid(alpha=0.25)

    resid = y_test.values - best_pred
    hb2 = axes[1].hexbin(
        np.log10(np.clip(best_pred, 1, None)), resid,
        gridsize=55, cmap=sns.light_palette("#0E7490", as_cmap=True), mincnt=1, linewidths=0,
    )
    cb2 = fig.colorbar(hb2, ax=axes[1], pad=0.02)
    cb2.set_label("test stones per bin", fontsize=8.5, color=INK)
    cb2.ax.tick_params(labelsize=8, colors=INK)
    cb2.outline.set_visible(False)
    axes[1].axhline(0, color=WARN, lw=1.3, ls="--")
    axes[1].set_xlabel("Predicted price, log10 scale")
    axes[1].set_ylabel("Residual in USD (actual - predicted)")
    axes[1].set_title("Residuals fan out at the top end", fontsize=11, color=INK)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "07_predicted_vs_actual.png")

    # residual by decile of carat: shows systematic underpricing of big stones
    dec = pd.qcut(best_pred, 10, labels=False, duplicates="drop")
    bias = pd.Series(resid).groupby(dec).mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(range(len(bias)), bias.values, color=ACCENT, width=0.65)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xlabel("Decile of predicted price (low to high)")
    ax.set_ylabel("Mean residual, USD")
    ax.set_title("The model under-prices the largest stones", fontsize=11, color=INK)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "08_residual_bias.png")

    # ---- feature importance -------------------------------------------------
    cols = REG_FEATS
    imp = best_model.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh([cols[i] for i in order], imp[order], color=ACCENT, height=0.66)
    ax.set_xlabel("Feature importance (variance reduction)")
    ax.set_title("What actually drives price", fontsize=12, color=INK)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "09_feature_importance.png")

    fi = pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False)
    fi.to_csv(ART / "feature_importance.csv", index=False)

    # n_jobs=1: on Windows the parallel backend cannot pickle the fitted
    # forest, and a single-threaded pass over 10,755 test rows is quick.
    perm = permutation_importance(rf, X_test, y_test, n_repeats=3, random_state=SEED, n_jobs=1)
    perm_mean = {c: float(v) for c, v in zip(X_test.columns, perm.importances_mean)}
    metrics["permutation_importance"] = perm_mean

    # carat, log_carat, volume and y are near-collinear (volume and carat
    # correlate at 0.999), so a tree can move the credit for one signal between
    # them arbitrarily. Per-feature importance is therefore misleading on its
    # own. Grouping the features into what a buyer actually controls gives an
    # honest split of the credit.
    GROUPS = {
        "weight": ["carat", "log_carat", "volume", "y", "x"],
        "clarity": ["clarity"],
        "colour": ["color"],
        "cut": ["cut"],
        "proportions": ["depth", "table", "z", "carat_per_volume"],
    }
    grouped = {}
    for gname, cols in GROUPS.items():
        # Shuffle every member of the group at once and measure the drop.
        rng = np.random.default_rng(SEED)
        scores = []
        for _ in range(3):
            Xp = X_test.copy()
            for c in cols:
                Xp[c] = rng.permutation(Xp[c].values)
            scores.append(r2_score(y_test, rf.predict(Xp)))
        grouped[gname] = float(metrics["best"]["r2"] - np.mean(scores))
    metrics["grouped_importance"] = grouped
    metrics["grouped_importance_note"] = (
        "R2 lost when the whole group is shuffled. Weight is a single signal "
        "spread across five collinear columns, so per-feature numbers "
        "understate it."
    )

    joblib.dump(best_model, ART / "reg_model.joblib")
    return best_model, metrics


# =========================================================== 6. CLASSIFIER
def run_classification(df: pd.DataFrame) -> tuple:
    X = encode(df)
    y = df["segment_code"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    print(f"Classification: train {len(X_train):,} / test {len(X_test):,} rows")
    print("Class balance:", {SEGMENTS[i]: int((y_test == i).sum()) for i in range(4)})

    metrics: dict = {"segments": SEGMENTS, "seed": SEED}
    best = None  # (name, f1, model, confusion_matrix)

    for name, model in [
        ("LogisticRegression", Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ])),
        ("RandomForest", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
    ]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        f1 = float(f1_score(y_test, pred, average="weighted"))
        acc = float(accuracy_score(y_test, pred))
        print(f"  {name:<22} acc {acc:.4f}  weighted F1 {f1:.4f}")
        metrics[name] = {
            "accuracy": acc,
            "f1_weighted": f1,
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1, 2, 3]).tolist(),
            "report": classification_report(y_test, pred, target_names=SEGMENTS, output_dict=True, zero_division=0),
        }
        this_cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
        if best is None or f1 > best[1]:
            best = (name, f1, model, this_cm)

    metrics["best_model"] = best[0]
    metrics["best_f1"] = best[1]
    metrics["confusion_matrix"] = best[3].tolist()
    metrics["classification_report"] = metrics[best[0]]["report"]

    # ---- confusion matrix chart -------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(best[3], annot=True, fmt="d", cmap=sns.light_palette(ACCENT, as_cmap=True),
                ax=ax, square=True, linewidths=0.6, linecolor="white",
                xticklabels=SEGMENTS, yticklabels=SEGMENTS, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted segment")
    ax.set_ylabel("Actual segment")
    ax.set_title(f"Segment confusion matrix ({best[0]})", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "09b_confusion_matrix.png")

    # ---- accuracy by actual segment, to expose the imbalance effect --------
    pred = best[2].predict(X_test)
    per = []
    for i, seg in enumerate(SEGMENTS):
        mask = y_test.values == i
        if mask.sum():
            per.append({
                "segment": seg,
                "n": int(mask.sum()),
                "recall": float((pred[mask] == i).mean()),
                "precision": float((pred == i)[mask].mean()) if (pred == i).any() else 0.0,
            })
    metrics["per_segment"] = per
    metrics["imbalance_note"] = (
        "Quartile segments are equal by construction, so the test set is balanced and "
        "macro-F1 is close to weighted-F1. The real difficulty is at the tier borders: "
        "adjacent quartiles are defined by price alone, so stones either side of a cut-off "
        "can be physically identical and still get different labels."
    )

    joblib.dump(best[2], ART / "clf_model.joblib")
    return best[2], metrics


# ============================================================== 7. K-MEANS
def run_kmeans(df: pd.DataFrame) -> dict:
    feats = ["carat", "volume", "depth", "table"]
    Z = StandardScaler().fit_transform(df[feats])
    search = {}
    best_k, best_sil, best_km = 4, -1, None
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        sil = float(silhouette_score(Z, km.labels_, sample_size=8000, random_state=SEED))
        search[k] = sil
        if sil > best_sil:
            best_k, best_sil, best_km = k, sil, km
    print(f"K-means: k={best_k}, silhouette {best_sil:.3f}")

    df = df.assign(cluster=best_km.labels_)
    profiles = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        profiles.append({
            "cluster": int(c),
            "n": int(len(sub)),
            "median_carat": float(sub["carat"].median()),
            "median_price": float(sub["price"].median()),
            "median_price_per_carat": float((sub["price"] / sub["carat"]).median()),
            "modal_cut": str(sub["cut"].mode().iloc[0]),
            "modal_color": str(sub["color"].mode().iloc[0]),
            "modal_clarity": str(sub["clarity"].mode().iloc[0]),
        })
    # Order clusters by median price so the chart reads left to right.
    profiles.sort(key=lambda p: p["median_price"])

    # Cross-tab cluster vs manual segment: do the two groupings agree?
    crosstab = pd.crosstab(df["cluster"], df["segment"])
    ct = crosstab.to_dict()
    # How often does the cluster's modal segment match the stone's own segment?
    mode_map = crosstab.idxmax(axis=1).to_dict()
    agree = float(np.mean([mode_map[c] == s for c, s in zip(df["cluster"], df["segment"])]))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for p in profiles:
        ax.scatter(p["median_carat"], p["median_price"], s=260, color=ACCENT, zorder=3,
                   edgecolors="white", linewidths=1.4)
        ax.annotate(f"C{p['cluster']}\n{p['n']:,} stones", (p["median_carat"], p["median_price"]),
                    textcoords="offset points", xytext=(0, 20), ha="center", fontsize=8.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Median carat per cluster")
    ax.set_ylabel("Median price per cluster (USD)")
    ax.set_title(f"K-means clusters (k={best_k}) separate mainly by size", fontsize=12, color=INK)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "10_kmeans_clusters.png")

    # Quality separation INSIDE each cluster. Size dominates between
    # clusters, so the interesting question is what still varies once size is
    # held inside one of them.
    within = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        ppc = (sub["price"] / sub["carat"]).groupby(sub["clarity"]).median()
        within.append({
            "cluster": int(c),
            "n": int(len(sub)),
            "clarity_min_grade": str(ppc.idxmin()),
            "clarity_max_grade": str(ppc.idxmax()),
            "clarity_min_ppc": float(ppc.min()),
            "clarity_max_ppc": float(ppc.max()),
            "clarity_spread_x": float(ppc.max() / ppc.min()),
        })

    joblib.dump(
        {"model": best_km, "scaler": StandardScaler().fit(df[feats]), "features": feats, "k": best_k},
        ART / "kmeans_model.joblib",
    )

    return {
        "k": best_k,
        "silhouette": best_sil,
        "silhouette_by_k": {str(k): v for k, v in search.items()},
        "profiles": profiles,
        "within_cluster_clarity": within,
        "cluster_segment_crosstab": {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in ct.items()},
        "modal_segment_agreement": agree,
        "interpretation": (
            f"Fitted on size and proportion with no price input, k-means found {best_k} size bands, "
            f"not quality bands. The modal cut grade is Ideal in all four price quartiles, and the "
            f"two views agree on {agree:.1%} of stones, which is expected from 2 clusters against "
            "4 quartiles. The tiers are size brackets in practice. Within a single cluster, "
            "clarity still separates price sharply, so quality governs the choice inside a size "
            "band while weight governs the choice between them."
        ),
    }


# ======================================================= 8. SEGMENT SUMMARY
def segment_summary(df: pd.DataFrame) -> dict:
    rows = []
    for seg in SEGMENTS:
        sub = df[df["segment"] == seg]
        ppc = sub["price"] / sub["carat"]
        rows.append({
            "segment": seg,
            "n": int(len(sub)),
            "median_price": float(sub["price"].median()),
            "price_min": float(sub["price"].min()),
            "price_max": float(sub["price"].max()),
            "price_std": float(sub["price"].std()),
            "iqr": float(sub["price"].quantile(0.75) - sub["price"].quantile(0.25)),
            "median_carat": float(sub["carat"].median()),
            "median_ppc": float(ppc.median()),
            "best_median_cut": str(sub["cut"].mode().iloc[0]),
            "best_median_color": str(sub["color"].mode().iloc[0]),
            "best_median_clarity": str(sub["clarity"].mode().iloc[0]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(ART / "segment_summary.csv", index=False)
    return {"segments": rows}


# ================================================================ 9. EXPORT
def export_chart_data(eda: dict, reg: dict, clf: dict, km: dict) -> None:
    """Everything the Next.js dashboard needs, in one JSON."""
    imp = pd.read_csv(ART / "feature_importance.csv")
    top = imp.head(8)
    full = pd.read_csv(ART / "_full_clean.csv")
    payload = {
        "meta": {
            "rows": int(len(full)),
            "seed": SEED,
        },
        "eda": {
            "skew": eda["skew"],
            "carat_price_r_log": eda["carat_price_r_log"],
            "carat_price_r_raw": eda["carat_price_r_raw"],
            "elasticity": eda["price_carat_elasticity"],
            "price_per_carat": eda["price_per_carat"],
            "correlations": {k: v for k, v in eda["correlations"].items() if k != "matrix"},
            "grade_tables": eda["grade_tables"],
            "size_controlled_grade_effect": eda["size_controlled_grade_effect"],
        },
        "regression": {
            "best": reg["best"],
            "best_model": reg["best_model"],
            "linear": reg["linear"],
            "random_forest": reg["random_forest"],
            "gradient_boosting": reg["gradient_boosting"],
            "coefficients": reg["linear_coefficients"],
            "permutation": reg["permutation_importance"],
            "grouped": reg["grouped_importance"],
            "grouped_note": reg["grouped_importance_note"],
        },
        "classification": {
            "best_model": clf["best_model"],
            "accuracy": clf[clf["best_model"]]["accuracy"],
            "f1_weighted": clf[clf["best_model"]]["f1_weighted"],
            "logistic": {k: clf["LogisticRegression"][k] for k in ("accuracy", "f1_weighted")},
            "logistic_vs_rf": {k: clf["RandomForest"][k] for k in ("accuracy", "f1_weighted")},
            "confusion_matrix": clf["confusion_matrix"],
            "per_segment": clf["per_segment"],
            "imbalance_note": clf["imbalance_note"],
        },
        "clustering": km,
        "segments": segment_summary(full),
        "feature_importance": [
            {"feature": r.feature, "importance": float(r.importance)} for r in top.itertuples()
        ],
    }
    (ART / "dashboard_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {ART / 'dashboard_data.json'}")


def main() -> None:
    df_raw = load_data()
    df, clean_log = clean(df_raw)
    eda = run_eda(df, clean_log)
    df = add_features(df)
    # Segments are price quartiles, defined once on the full clean set so the
    # cut-off points are stable and reusable by the API.
    df["segment"] = pd.qcut(df["price"], 4, labels=SEGMENTS)
    df["segment_code"] = pd.Categorical(df["segment"], categories=SEGMENTS).codes
    df.to_csv(ART / "_full_clean.csv", index=False)

    reg_model, reg = run_regression(df, eda)
    clf_model, clf = run_classification(df)
    km = run_kmeans(df)
    export_chart_data(eda, reg, clf, km)

    combined = {
        "cleaning": clean_log,
        "regression": reg,
        "classification": clf,
        "clustering": km,
    }
    (ART / "metrics.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("\nArtifacts written to", ART)


if __name__ == "__main__":
    main()
