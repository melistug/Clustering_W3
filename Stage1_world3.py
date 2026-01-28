# -*- coding: utf-8 -*-
"""
STAGE-1 (World3-03) — FULL VERSION (CSV ONLY)

✔ LHS sampling (lookup olmayan lever'lar)
✔ World3-03 PySD run
✔ Outputs:
   - params.csv
   - time.csv
   - model_doc.csv
   - *_full_df.csv
   - *_dtw_df.csv
   - time_series_long.csv
"""

import os
import re
import difflib
from pathlib import Path
import numpy as np
import pandas as pd
import pysd
import matplotlib
matplotlib.use("Agg")

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\ASUS\Desktop\wrld3-03.mdl"
OUT_DIR    = r"C:\Users\ASUS\Desktop\world3_stage1_full_v5  "

N_SAMPLES = 200
TIME_START = 1900
TIME_END   = 2100
DT = 1
DTW_TARGET_DT = 1.0

# lookup OLMAYAN lever'lar (MODEL_DOC ile doğrulanmış)
BOUNDS_USER = {
    "desired_completed_family_size_normal": (2.0, 3.5),
    "lifetime_perception_delay": (5, 25),
    "social_adjustment_delay": (5, 30),
    "desired_food_ratio": (0.8, 1.3),
    "food_shortage_perception_delay": (1, 10),
    "industrial_capital_output_ratio_1": (2.5, 5.0),
}


# hedef çıktılar (NET)
TARGET_OUTPUTS_USER = [
    "population",
    "industrial_output",
    "persistent_pollution",
]

# =====================================================
# UTILS
# =====================================================
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def canonical(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def lhs(bounds, n, seed=0):
    rng = np.random.default_rng(seed)
    names = list(bounds.keys())
    edges = np.linspace(0, 1, n + 1)
    X = np.zeros((n, len(names)))

    for j, k in enumerate(names):
        lo, hi = bounds[k]
        u = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(u)
        X[:, j] = lo + u * (hi - lo)

    return pd.DataFrame(X, columns=names)

def resolve_names(model):
    doc = model.doc.copy()
    doc.to_csv(os.path.join(OUT_DIR, "model_doc.csv"), index=False)

    canon_map = {}
    for _, r in doc.iterrows():
        canon_map[canonical(r["Real Name"])] = r["Py Name"]
        canon_map[canonical(r["Py Name"])] = r["Py Name"]

    def resolve(name):
        key = canonical(name)
        if key in canon_map:
            return canon_map[key]
        matches = difflib.get_close_matches(key, canon_map.keys(), n=1, cutoff=0.85)
        if matches:
            return canon_map[matches[0]]
        return None

    return resolve

# =====================================================
# MAIN
# =====================================================
def main():
    ensure_dir(OUT_DIR)

    print("1) Model yükleniyor...")
    model = pysd.read_vensim(MODEL_PATH)
    print("   ✔ Model yüklendi.")

    print("2) İsim eşleştirme...")
    resolve = resolve_names(model)

    bounds_py = {}
    for k, v in BOUNDS_USER.items():
        py = resolve(k)
        if py is None:
            raise ValueError(f"Lever bulunamadı: {k}")
        bounds_py[py] = v

    targets_py = []
    for t in TARGET_OUTPUTS_USER:
        py = resolve(t)
        if py is None:
            raise ValueError(f"Target bulunamadı: {t}")
        targets_py.append(py)

    print("   ✔ Target eşleşmeleri:")
    for u, p in zip(TARGET_OUTPUTS_USER, targets_py):
        print(f"     {u:22s} -> {p}")

    print("3) LHS...")
    grid = lhs(bounds_py, N_SAMPLES)
    grid.to_csv(os.path.join(OUT_DIR, "params.csv"), index=False)

    times = np.arange(TIME_START, TIME_END + DT, DT)
    pd.Series(times, name="time").to_csv(os.path.join(OUT_DIR, "time.csv"), index=False)

    dtw_factor = int(round(DTW_TARGET_DT / DT))

    full_series = {v: [] for v in targets_py}
    dtw_series  = {v: [] for v in targets_py}
    long_full = []
    long_dtw  = []

    print("4) Simülasyonlar...")
    for i, row in grid.iterrows():
        print(f"\n---- Run {i+1}/{N_SAMPLES} ----")
        res = model.run(
            params=row.to_dict(),
            return_columns=targets_py,
            return_timestamps=times,
            initial_condition="original",
            progress=False
        )

        for v in targets_py:
            y = res[v].values
            full_series[v].append(y)
            dtw_series[v].append(y[::dtw_factor])

        for ti, t in enumerate(times):
            for v in targets_py:
                long_full.append((i+1, t, v, res[v].values[ti]))

        for ti, t in enumerate(times[::dtw_factor]):
            for v in targets_py:
                long_dtw.append((i+1, t, v, res[v].values[ti]))

    # =================================================
    # SAVE WIDE CSV
    # =================================================
    for u, p in zip(TARGET_OUTPUTS_USER, targets_py):
        df_full = pd.DataFrame(full_series[p], columns=times)
        df_full.index.name = "run"
        df_full.to_csv(os.path.join(OUT_DIR, f"{u}_full_df.csv"))

        df_dtw = pd.DataFrame(dtw_series[p], columns=times[::dtw_factor])
        df_dtw.index.name = "run"
        df_dtw.to_csv(os.path.join(OUT_DIR, f"{u}_dtw_df.csv"))

    # =================================================
    # SAVE LONG CSV
    # =================================================
    df_long_full = pd.DataFrame(long_full, columns=["run", "time", "var", "value"])
    df_long_dtw  = pd.DataFrame(long_dtw,  columns=["run", "time", "var", "value"])

    df_long_full.to_csv(os.path.join(OUT_DIR, "time_series_long.csv"), index=False)
    df_long_dtw.to_csv(os.path.join(OUT_DIR, "time_series_long_dtw.csv"), index=False)

    print("\n✅ STAGE-1 FULL TAMAMLANDI")
    print("   - CSV only")
    print("   - DTW hazır")
    print("   - Stage-2 uyumlu")

# =====================================================
if __name__ == "__main__":
    main()
