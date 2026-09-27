"""Run trace.py across a spread of stones and check the walkthrough holds up."""

import json
import subprocess
import sys

PY = r"C:\Users\DELL\Desktop\Projects\diamond-pricing\.venv\Scripts\python.exe"
CASES = [
    (0.3, "Fair", "J", "I1"),
    (0.5, "Good", "H", "SI1"),
    (0.9, "Very Good", "F", "VS1"),
    (1.2, "Very Good", "G", "VS2"),
    (1.8, "Premium", "E", "VVS1"),
    (2.5, "Ideal", "D", "IF"),
    (0.7, "Ideal", "D", "IF"),      # a rare combo at a small weight
    (3.4, "Very Good", "G", "VS2"),  # large, ordinary grades
]

print("%-26s %9s %9s %8s %7s" % ("stone", "model", "market", "err%", "n"))
print("-" * 64)
worst = 0.0
for carat, cut, color, clarity in CASES:
    out = subprocess.run(
        [PY, r"C:\Users\DELL\Desktop\Projects\diamond-pricing\trace.py",
         str(carat), cut, color, clarity],
        capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        print("%-26s FAILED: %s" % (f"{carat}ct {cut}/{color}/{clarity}",
                                    (out.stderr or "").strip().splitlines()[-1:]))
        continue
    text = out.stdout
    d = json.loads(text[text.find("{"):text.rfind("}") + 1])
    r, rc = d["result"], d["reality_check"]
    if rc["error_pct"] is None:
        print("%-26s %9.0f %9s %8s %7d  (no comparable group, ref"
              "used one)" % (f"{carat}ct {cut}/{color}/{clarity}",
                             r["predicted_price"], "n/a", "n/a",
                             rc["comparable_stones"]))
        continue
    err = abs(rc["error_pct"])
    worst = max(worst, err)
    print("%-26s %9.0f %9.0f %7.1f%% %7d  %s" % (
        f"{carat}ct {cut}/{color}/{clarity}",
        r["predicted_price"], rc["market_median_price"], rc["error_pct"],
        rc["comparable_stones"], rc["comparison_group"]))

print("-" * 64)
print("worst absolute error vs same-grade market median: %.1f%%" % worst)
