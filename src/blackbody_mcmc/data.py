from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple

# skip blank lines
# replace zero/neg sigma values with 1% of intensity magnitude 
# so likelihood doesn't divide by 0
def load_data(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found.")

    cleaned = []
    bad_lines = []
    with open(path, "r") as fh:
        for lineno, raw in enumerate(fh, start=1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 2:
                bad_lines.append((lineno, "too few columns", parts))
                continue
            cleaned.append((lineno, parts))

    if not cleaned:
        raise ValueError(f"No valid data rows found in {path}")

    # allow rows with >=2 columns,
    # treat missing sigma as NaN for nwo
    wl_list = []
    I_list = []
    sigma_list = []
    skipped = []
    for lineno, parts in cleaned:
        try:
            wl = float(parts[0])
            I = float(parts[1])
            if len(parts) >= 3 and parts[2] != "":
                sigma = float(parts[2])
            else:
                sigma = np.nan
        except ValueError:
            skipped.append((lineno, parts))
            continue
        wl_list.append(wl)
        I_list.append(I)
        sigma_list.append(sigma)

    if skipped:
        print("skipped:")
        for ln, p in skipped[:10]:
            print(f"  Line {ln}: {p}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")

    wl = np.array(wl_list, dtype=float)
    I = np.array(I_list, dtype=float)
    sigma = np.array(sigma_list, dtype=float)

    # Replace NaN or non-positive sigma
    bad_sigma_mask = ~np.isfinite(sigma) | (sigma <= 0)
    if np.any(bad_sigma_mask):
        nbad = np.count_nonzero(bad_sigma_mask)
        fallback = 0.01 * np.maximum(np.abs(I[bad_sigma_mask]), 1.0)
        print(f"Replacing {nbad} zero/invalid sigma values.")
        sigma[bad_sigma_mask] = fallback

    return wl, I, sigma
