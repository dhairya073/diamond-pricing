"""
Contrast check for the site's colour tokens.

The design has two colour schemes, and small-text tokens tuned for one are
routinely unreadable on the other. This computes WCAG contrast for every
text/background pair the stylesheet actually defines, in both schemes, and
exits non-zero on a failure. Run it after any palette change:

    python verify_contrast.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CSS = Path(__file__).parent / "web" / "app" / "globals.css"


def srgb_to_lin(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return (
        0.2126 * srgb_to_lin(r)
        + 0.7152 * srgb_to_lin(g)
        + 0.0722 * srgb_to_lin(b)
    )


def contrast(fg: str, bg: str) -> float:
    def parse(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    l1, l2 = luminance(parse(fg)), luminance(parse(bg))
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def read_tokens() -> dict[str, str]:
    """Pull the hex values out of the :root block."""
    text = CSS.read_text(encoding="utf-8")
    root = text.split(":root", 1)[1].split("\n}", 1)[0]
    tokens: dict[str, str] = {}
    for name, value in re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{3,8})", root):
        tokens[name] = value
    return tokens


T = read_tokens()

# (label, foreground token, background token, minimum)
PAIRS = [
    # dark scheme, the default
    ("dark: body text on page", "--cream", "--night", 4.5),
    ("dark: lede on page", "--lilac", "--night", 4.5),
    ("dark: lede on panel", "--lilac", "--night-2", 4.5),
    ("dark: caption on panel", "--lilac", "--night-2", 4.5),
    ("dark: eyebrow label", "--lilac", "--night", 4.5),
    ("dark: small index text", "--ink-faint", "--night-2", 4.5),
    ("dark: nav link", "--ink-faint", "--night", 4.5),
    ("dark: accent display number", "--accent", "--night", 3.0),
    ("dark: accent on raised panel", "--accent", "--night-3", 3.0),
    ("dark: key in spec row", "--ink-faint", "--night-2", 4.5),
    ("dark: value in spec row", "--cream", "--night-2", 4.5),
    # light scheme
    ("light: body text on page", "--ink", "--paper", 4.5),
    ("light: body text on panel", "--ink-soft", "--paper-2", 4.5),
    ("light: caption on panel", "--ink-soft", "--paper-2", 4.5),
    ("light: eyebrow label", "--eyebrow-on-light", "--paper", 4.5),
    ("light: eyebrow on panel", "--eyebrow-on-light", "--paper-2", 4.5),
    ("light: small index text", "--faint-on-light", "--paper", 4.5),
    ("light: index on panel", "--faint-on-light", "--paper-2", 4.5),
    ("light: link / hover text", "--accent-text-on-light", "--paper", 4.5),
    ("light: inline code", "--accent-text-on-light", "--paper-2", 4.5),
    ("light: primary button label", "--night", "--accent", 4.5),
    ("light: ghost button label", "--ink", "--paper", 4.5),
]

fails = []
print(f"read {len(T)} tokens from {CSS.name}\n")
for label, fg, bg, need in PAIRS:
    if fg not in T or bg not in T:
        fails.append(f"{label}: missing token {fg if fg not in T else bg}")
        print(f"  FAIL  {label:<38} missing token")
        continue
    r = contrast(T[fg], T[bg])
    ok = r >= need
    if not ok:
        fails.append(f"{label}: {r:.2f} < {need}")
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<38} {r:5.2f}  needs {need}")

print()
if fails:
    print(f"RESULT: {len(fails)} contrast failures")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print(f"RESULT: all {len(PAIRS)} pairs meet their minimum")
