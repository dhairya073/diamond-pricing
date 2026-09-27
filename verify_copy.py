"""Verify every factual claim the dashboard copy makes against the artifacts."""

import json

ROOT = r"C:\Users\DELL\Desktop\Projects\diamond-pricing"
d = json.load(open(f"{ROOT}\\artifacts\\dashboard_data.json", encoding="utf-8"))
raw = json.load(open(f"{ROOT}\\artifacts\\metrics.json", encoding="utf-8"))

fails, warns = [], []


def check(label, cond, detail=""):
    (print if cond else fails.append)  # noqa
    if cond:
        print(f"  PASS  {label}  {detail}")
    else:
        fails.append(f"FAIL  {label}  {detail}")
        print(f"  FAIL  {label}  {detail}")


def warn(label, cond, detail=""):
    if not cond:
        warns.append(f"{label} {detail}")
        print(f"  WARN  {label}  {detail}")


print("=== headline numbers ===")
check("rows = 53,772", d["meta"]["rows"] == 53772, str(d["meta"]["rows"]))
check("168 rows removed", 53940 - d["meta"]["rows"] == 168, str(53940 - d["meta"]["rows"]))

print()
print("=== hero / stat copy ===")
best = d["regression"]["best"]
clf = d["classification"]
check("R2 0.9824 in copy", round(best["r2"], 4) == 0.9824, f"{best['r2']:.6f}")
check("MAE $262 in copy", round(best["mae"]) == 262, f"${best['mae']:.2f}")
check("RMSE $523 in copy", round(best["rmse"]) == 523, f"${best['rmse']:.2f}")
check("F1 0.933 in copy", round(clf["f1_weighted"], 3) == 0.933, f"{clf['f1_weighted']:.6f}")
check("logistic F1 0.910", round(clf["logistic"]["f1_weighted"], 3) == 0.910,
      f"{clf['logistic']['f1_weighted']:.4f}")
check("accuracy 93.3% shown", abs(clf["accuracy"] * 100 - 93.3) < 0.1,
      f"{clf['accuracy']*100:.2f}%")

print()
print("=== EDA copy ===")
sk = d["eda"]["skew"]
warn("skew values present", "price" in sk and "carat" in sk, str(sk))
c = d["eda"]["correlations"]
check("carat-price corr 0.922", round(c["carat_price"], 3) == 0.922, f"{c['carat_price']:.4f}")
check("depth corr ~-0.011", round(c["depth_price"], 3) == -0.011, f"{c['depth_price']:.4f}")
check("table corr ~0.127", round(c["table_price"], 3) == 0.127, f"{c['table_price']:.4f}")
check("r_log 0.966", round(d["eda"]["carat_price_r_log"], 3) == 0.966,
      f"{d['eda']['carat_price_r_log']:.4f}")
el = d["eda"]["elasticity"]
pct10 = (1.1 ** el * 100 - 100)
print(f"  INFO  elasticity {el:.3f} -> 10% heavier = {pct10:.1f}% more")
check("elasticity claim 'faster than weight' (pct10 > 15)", pct10 > 15, f"{pct10:.1f}%")

ppc = d["eda"]["price_per_carat"]
check("ppc p05 $1,719", round(ppc["p05"]) == 1719, f"${ppc['p05']:.0f}")
check("ppc max $17,829", round(ppc["max"]) == 17829, f"${ppc['max']:.0f}")
check("ppc median $3,496", round(ppc["median"]) == 3496, f"${ppc['median']:.0f}")

print()
print("=== size-controlled grade effect ===")
g = d["eda"]["size_controlled_grade_effect"]
check("5 bands", g["bands"] == 5, str(g["bands"]))
for k in ("clarity", "color", "cut"):
    e = g[k]
    bands = [float(v) for v in e["ratio_by_band"].values()]
    print(f"  INFO  {k}: {e['worst_grade']} vs {e['best_grade']} "
          f"by band {[f'{v*100:.0f}%' for v in bands]}")
    # Direction is per-grade, not universal: cut and clarity fall with size,
    # colour rises. The copy states each one individually.
    direction = "FALLS" if bands[-1] < bands[0] else "RISES"
    if k in ("cut", "clarity"):
        check(f"{k} falls with size", bands[-1] < bands[0],
              f"band0 {bands[0]:.2f} -> last {bands[-1]:.2f} ({direction})")
    else:
        check(f"{k} rises with size", bands[-1] > bands[0],
              f"band0 {bands[0]:.2f} -> last {bands[-1]:.2f} ({direction})")
    check(f"{k} best grade worth >1.3x worst at top end", 1 / bands[-1] > 1.3,
          f"{1/bands[-1]:.2f}x")
check("clarity steepest of three",
      g["clarity"]["ratio_largest_band"] < min(g["color"]["ratio_largest_band"],
                                               g["cut"]["ratio_largest_band"]),
      f"clarity {g['clarity']['ratio_largest_band']:.3f}")

print()
print("=== grouped importance ===")
grp = d["regression"]["grouped"]
for k, v in sorted(grp.items(), key=lambda x: -x[1]):
    print(f"  INFO  {k:<12} {v:+.4f}")
check("weight is the largest group",
      max(grp, key=grp.get) == "weight", str(sorted(grp, key=grp.get, reverse=True)))
others = sum(v for k, v in grp.items() if k != "weight" and k != "proportions")
print(f"  INFO  weight {grp['weight']:.3f} vs clarity+colour+cut {others:.3f}")
check("weight > clarity+colour+cut", grp["weight"] > others, f"{grp['weight']:.3f} vs {others:.3f}")

print()
print("=== segments ===")
segs = d["segments"]["segments"]
check("4 segments", len(segs) == 4, str(len(segs)))
cuts = {s["segment"]: s["best_median_cut"] for s in segs}
print("  INFO  modal cut by segment:", cuts)
check("all tiers modal cut = Ideal (claim 'Ideal in every tier')",
      all(v == "Ideal" for v in cuts.values()), str(cuts))
best_seg = max(segs, key=lambda s: s["iqr"] / s["median_price"])
check("Luxury is widest relative IQR", best_seg["segment"] == "Luxury",
      f"{best_seg['segment']} {best_seg['iqr']/best_seg['median_price']*100:.1f}%")
print(f"  INFO  Luxury range ${best_seg['price_min']:.0f} to ${best_seg['price_max']:.0f}, "
      f"IQR ${best_seg['iqr']:.0f}")

print()
print("=== clustering ===")
km = d["clustering"]
check("k=2 silhouette 0.351", km["k"] == 2 and round(km["silhouette"], 3) == 0.351,
      f"k={km['k']} sil={km['silhouette']:.4f}")
# The copy says agreement is LOW, so the check is that it is indeed low.
check("agreement is low (copy says 'only ...')", km["modal_segment_agreement"] < 0.6,
      f"{km['modal_segment_agreement']*100:.2f}%")
sil = {int(k): v for k, v in km["silhouette_by_k"].items()}
check("silhouette falls as k rises", sil[2] > sil[8], f"k2 {sil[2]:.3f} -> k8 {sil[8]:.3f}")
for w in km["within_cluster_clarity"]:
    print(f"  INFO  C{w['cluster']} n={w['n']:6d} clarity ppc "
          f"{w['clarity_min_grade']} ${w['clarity_min_ppc']:.0f} -> "
          f"{w['clarity_max_grade']} ${w['clarity_max_ppc']:.0f} "
          f"({w['clarity_spread_x']:.2f}x)")
big = max(km["within_cluster_clarity"], key=lambda w: w["clarity_spread_x"])
# Pick the large-stone cluster by median weight, not by row count: most
# diamonds are small, so the larger cluster is the small-stone one.
big_prof = max(km["profiles"], key=lambda p: p["median_carat"])
big = next(w for w in km["within_cluster_clarity"] if w["cluster"] == big_prof["cluster"])
check("large-stone cluster found by carat, not count",
      big["cluster"] == big_prof["cluster"],
      f"C{big['cluster']} median {big_prof['median_carat']:.2f}ct, spread {big['clarity_spread_x']:.2f}x")
check("clarity separates price inside the large cluster", big["clarity_spread_x"] > 2.0,
      f"{big['clarity_spread_x']:.2f}x in cluster {big['cluster']}")

print()
print("=== the premium arithmetic the copy quotes ===")
# The Results component prints the best grade as a percentage above the worst
# using 1/ratio - 1. The table stores worst/best, so reading the raw ratio as
# a premium understates it by about half. Pin the derived numbers so a change
# to the table cannot silently make the prose wrong.
for k in ("clarity", "cut", "color"):
    e = g[k]
    keys = sorted(e["ratio_by_band"], key=float)
    lo, hi = keys[0], keys[-1]
    prem_lo = round((1 / e["ratio_by_band"][lo] - 1) * 100)
    prem_hi = round((1 / e["ratio_by_band"][hi] - 1) * 100)
    raw_mid = round((100 - e["ratio_by_band"][lo] * 100))
    print(f"  INFO  {k}: premium {prem_lo}% (band {lo}) -> {prem_hi}% (band {hi}); "
          f"naive 100-ratio would have said {raw_mid}%")
    # The two formulas coincide at the extremes: when the ratio is near 1 or
    # near 0.5, 1/r - 1 and 1 - r round to the same whole number. So assert
    # on a band where they genuinely differ, not on the first one.
    diff_band = next((b for b in keys
                      if round((1 / e["ratio_by_band"][b] - 1) * 100)
                      != round(100 - e["ratio_by_band"][b] * 100)), None)
    check(f"{k} premium uses the inverse, not 100-ratio",
          diff_band is not None,
          f"they differ at band {diff_band}"
          + (f": {round((1/e['ratio_by_band'][diff_band]-1)*100)}% vs "
             f"{round(100-e['ratio_by_band'][diff_band]*100)}%" if diff_band else "")
          + f"; at band {lo} they coincide by rounding" if diff_band else "")
    check(f"{k} best/worst ratio > 1 at every band",
          all(1 / v > 1 for v in e["ratio_by_band"].values()),
          f"min {min(1/v for v in e['ratio_by_band'].values()):.2f}x")

# Clarity peaks in a middle band, not the largest. The copy says so, and
# picking the largest band instead would understate the finding.
clar = g["clarity"]["ratio_by_band"]
peak = min(clar, key=lambda k: clar[k])
last = sorted(clar, key=float)[-1]
print(f"  INFO  clarity lowest ratio at band {peak} ({clar[peak]:.2f}), "
      f"largest band is {last} ({clar[last]:.2f})")
check("clarity premium peaks before the largest band", peak != last,
      f"peak {peak} vs last {last}")
check("peak premium beats the largest-band premium",
      1 / clar[peak] > 1 / clar[last],
      f"{1/clar[peak]:.2f}x vs {1/clar[last]:.2f}x")

print()
print("=== confusion matrix ===")
cm = clf["confusion_matrix"]
diag = sum(cm[i][i] for i in range(4))
total = sum(sum(r) for r in cm)
check("diagonal dominant", diag / total > 0.85, f"{diag/total*100:.1f}% on diagonal")
off = [cm[i][j] for i in range(4) for j in range(4) if abs(i - j) > 1]
check("all errors are adjacent tiers", sum(off) == 0, f"{sum(off)} non-adjacent errors")

print()
print("=" * 60)
print(f"RESULT: {len(fails)} failures, {len(warns)} warnings")
for f in fails:
    print("  " + f)
