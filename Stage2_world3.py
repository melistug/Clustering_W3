# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mt_tree import TreeForecast

# ===================== USER CONFIG ======================
PARAMS_CSV = r"C:\Users\ASUS\Desktop\world3_stage1_full_v5\params.csv"

DF_TABLES_DTW = {
    "POPULATION":        r"C:\Users\ASUS\Desktop\world3_stage1_full_v5\population_dtw_df.csv",
    "INDUSTRIAL_OUTPUT": r"C:\Users\ASUS\Desktop\world3_stage1_full_v5\industrial_output_dtw_df.csv",
    "PERSISTENT_POLLUTION":    r"C:\Users\ASUS\Desktop\world3_stage1_full_v5\persistent_pollution_dtw_df.csv",
}

DF_TABLES_FULL = {
    "POPULATION":        r"C:\Users\ASUS\Desktop\world3_stage1_full_v5\population_full_df.csv",
    "INDUSTRIAL_OUTPUT": r"C:\Users\ASUS\Desktop\world3_stage1_full_v5\industrial_output_full_df.csv",
    "PERSISTENT_POLLUTION":         r"C:\Users\ASUS\Desktop\world3_stage1_full_v5\persistent_pollution_full_df.csv",
}

OUT_DIR  = r"C:\Users\ASUS\Desktop\world3_stage1_full_v5\world3_stage2_out_1102"
FEATURES = [
    "desired_completed_family_size_normal",
    "lifetime_perception_delay",
    "social_adjustment_delay",
    "desired_food_ratio",
    "food_shortage_perception_delay",
    "industrial_capital_output_ratio_1",
]

Y_NORM_MODE = "standardize"

TIME_START = 1900.0
TIME_STEP  = 1.0
# ---------- İSTEĞE BAĞLI ÇIKTI KONTROLÜ ----------
SAVE_SPLITS_CSV = False
SAVE_PER_SAMPLE_RULES_CSV = False

# =======================================================
# ============ GENERIC SAFE CALL HELPERS =================
def try_many_calls(obj, meth_name, positional_arg_candidates=(), kw_candidate_sets=()):
    meth = getattr(obj, meth_name, None)
    if meth is None:
        raise AttributeError(f"{meth_name} not found on {obj}")
    last_err = None
    if not positional_arg_candidates:
        positional_arg_candidates = [()]
    if not kw_candidate_sets:
        kw_candidate_sets = [dict()]
    for pos in positional_arg_candidates:
        for kw in kw_candidate_sets:
            try:
                return meth(*pos, **kw)
            except Exception as e:
                last_err = e
                continue
    raise last_err or RuntimeError(f"All {meth_name} call variants failed.")

# -------------------- DTW (1D) --------------------------
def _dtw_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n, m = len(a), len(b)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])

def dtw_pairwise_1d(Y_2d: np.ndarray, labels=None) -> pd.DataFrame:
    Y = np.asarray(Y_2d, dtype=float)
    N = Y.shape[0]
    D = np.zeros((N, N), dtype=float)
    total_pairs = N*(N-1)//2
    print(f"[info] (1D) DTW pair sayısı: {total_pairs}", flush=True)
    for i in range(N):
        print(f"[progress] satır {i+1}/{N} başlıyor...", flush=True)
        D[i, i] = 0.0
        for j in range(i + 1, N):
            d = _dtw_distance_1d(Y[i], Y[j])
            D[i, j] = d
            D[j, i] = d
        print(f"[progress] satır {i+1}/{N} bitti.", flush=True)
    if labels is None:
        labels = np.arange(N)
    df = pd.DataFrame(D, index=labels, columns=labels)
    return df

# -------------------- DTW (multivariate) ----------------
def normalize_outputs(Y_mv: np.ndarray, mode: str):
    if mode is None:
        return Y_mv
    Y = np.asarray(Y_mv, dtype=np.float64).copy()
    N, A, T = Y.shape

    if mode == 'standardize':
        for a in range(A):
            A2 = Y[:, a, :].astype(np.float64, copy=True)
            maxabs = np.nanmax(np.abs(A2))
            if np.isfinite(maxabs) and maxabs > 1e150:
                power = int(np.floor(np.log10(maxabs)) - 140)
                A2 = A2 / (10.0 ** power)
            mean = 0.0; M2 = 0.0; count = 0
            it = np.nditer(A2, flags=['multi_index'])
            while not it.finished:
                x = float(it[0])
                if np.isfinite(x):
                    count += 1
                    delta = x - mean
                    mean += delta / count
                    M2   += delta * (x - mean)
                it.iternext()
            std = np.sqrt(M2 / max(count,1)) if count > 1 else 1.0
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            Y[:, a, :] = (A2 - mean) / std
        return Y

    if mode == 'global':
        for a in range(A):
            axis_vals = Y[:, a, :]
            mn = float(np.nanmin(axis_vals)); mx = float(np.nanmax(axis_vals))
            span = mx - mn if mx > mn else 1.0
            Y[:, a, :] = (axis_vals - mn) / span
        return Y

    if mode == 'per_series':
        for n in range(N):
            for a in range(A):
                mn = float(np.nanmin(Y[n, a, :]))
                mx = float(np.nanmax(Y[n, a, :]))
                span = mx - mn if mx > mn else 1.0
                Y[n, a, :] = (Y[n, a, :] - mn) / span
        return Y

    raise ValueError("Unknown normalization mode")

def dtw_pairwise_multi_stage2(
    Y_mv: np.ndarray,
    axis_weights=None,
    axis_names=None,
    return_axis_dtws=False,
    row_norm: str = "none",
    labels=None
):
    Y_mv = np.asarray(Y_mv, dtype=float)
    if Y_mv.ndim != 3:
        raise ValueError("Y_mv must have shape (N, A, T)")
    N, A, T = Y_mv.shape
    w = np.ones(A, dtype=float) if axis_weights is None else np.asarray(axis_weights, dtype=float).reshape(-1)
    if w.size != A:
        raise ValueError("axis_weights length must match A")
    if axis_names is None:
        axis_names = [f"AXIS{a}" for a in range(A)]
    if labels is None:
        labels = np.arange(N)

    def _row_minmax_norm(Y2d):
        mn = Y2d.min(axis=1, keepdims=True)
        mx = Y2d.max(axis=1, keepdims=True)
        span = np.where((mx - mn) == 0.0, 1.0, (mx - mn))
        return (Y2d - mn) / span

    print(f"[info] MV-DTW: N={N}, A={A}, T={T}, weights={w.tolist()}, row_norm={row_norm}", flush=True)
    D_total = None
    axis_dtws = {}

    for a in range(A):
        Y_axis = Y_mv[:, a, :]
        Yn = _row_minmax_norm(Y_axis) if row_norm == "per_series" else Y_axis
        D_a = dtw_pairwise_1d(Yn, labels=labels)
        axis_dtws[str(axis_names[a]).upper()] = D_a
        D_total = (w[a] * D_a.values) if D_total is None else (D_total + w[a] * D_a.values)

    df_total = pd.DataFrame(D_total, index=labels, columns=labels)
    return (df_total, axis_dtws) if return_axis_dtws else df_total

# ---------- Rule/constraint formatting --------------
def parse_constraints_from_path_text(path_text: str):
    bounds = {}
    if not isinstance(path_text, str) or not path_text.strip():
        return bounds
    s = path_text.replace("\\n", "\n")
    pat = re.compile(
        r"(?:Rule:\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*(<=|<|>=|>|≤|≥)\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    for m in pat.finditer(s):
        name, op, val = m.group(1), m.group(2), float(m.group(3))
        if op == "≤": op = "<="
        if op == "≥": op = ">="
        bd = bounds.get(name, {"low": -np.inf, "low_inc": False, "high": np.inf, "high_inc": False})
        if op in ("<", "<="):
            if (val < bd["high"]) or (val == bd["high"] and op == "<=" and not bd["high_inc"]):
                bd["high"] = val
                bd["high_inc"] = (op == "<=")
        else:
            if (val > bd["low"]) or (val == bd["low"] and op == ">=" and not bd["low_inc"]):
                bd["low"] = val
                bd["low_inc"] = (op == ">=")
        bounds[name] = bd
    return bounds

def format_constraints_compact(bounds: dict, max_items: int = 14) -> str:
    items = []
    for name, bd in bounds.items():
        lo     = bd.get("low", -np.inf)
        hi     = bd.get("high",  np.inf)
        lo_inc = bd.get("low_inc",  False)
        hi_inc = bd.get("high_inc", False)

        infeasible = False
        if np.isfinite(lo) and np.isfinite(hi):
            if lo > hi:
                infeasible = True
            if lo == hi and (not lo_inc or not hi_inc):
                # (lo,hi) gibi açık uç -> boş küme
                infeasible = True

        if infeasible:
            s = f"⚠ {name}: INFEASIBLE (lo={lo:.6g}{'≤' if lo_inc else '<'} , hi={'≤' if hi_inc else '<'}{hi:.6g})"
        else:
            if np.isfinite(lo) and np.isfinite(hi):
                left  = "≤" if lo_inc else "<"
                right = "≤" if hi_inc else "<"
                s = f"{lo:.6g} {left} {name} {right} {hi:.6g}"
            elif np.isfinite(hi):
                sym = "≤" if hi_inc else "<"
                s = f"{name} {sym} {hi:.6g}"
            elif np.isfinite(lo):
                sym = "≥" if lo_inc else ">"
                s = f"{name} {sym} {lo:.6g}"
            else:
                s = f"{name}: (no bounds)"
        items.append(s)

    items.sort()
    if len(items) > max_items:
        items = items[:max_items] + ["…"]
    return "\n".join("• " + it for it in items)


# --------------- Output helpers --------------------
def print_and_save_splits(tree, X_cols, out_dir, save=True):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    if hasattr(tree, "splits"):
        pos_cands = [(), (getattr(tree, "Tree", None),), ([],)]
        kw_cands  = [dict(), dict(split_list=[])]
        for pos in pos_cands:
            try:
                s = try_many_calls(tree, "splits", [pos], kw_cands)
                if isinstance(s, (list, tuple)):
                    rows = [{"split": str(x)} for x in s]
                elif s is not None:
                    rows = [{"split": str(s)}]
                if rows:
                    break
            except Exception:
                continue
    if not rows:
        if hasattr(tree, "print_tree"):
            try:
                txt = str(tree.print_tree())
                lines = [ln for ln in txt.splitlines() if ln.strip()]
                rows = [{"split": line} for line in lines] or [{"split": "NA"}]
            except Exception:
                rows = [{"split": "NA"}]
        else:
            rows = [{"split": "NA"}]
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(os.path.join(out_dir, "splits.csv"), index=False)
        print(f"[saved] splits.csv (rows: {len(df)})")
    return df

def per_sample_rules(tree, X_df: pd.DataFrame, out_dir: str, max_depth: int, save=True):
    os.makedirs(out_dir, exist_ok=True)
    n = len(X_df)
    leaf_ids = None
    for kw in [ {"depth":max_depth}, {"max_depth":max_depth}, {} ]:
        try:
            leaf_ids = np.asarray(tree.apply(X_df, **kw)).astype(int).reshape(-1)
            break
        except Exception:
            continue
    if leaf_ids is None:
        leaf_ids = np.empty(n, dtype=int)
        for i in range(n):
            ok = False
            for kw in [ {"depth":max_depth}, {"max_depth":max_depth}, {} ]:
                try:
                    leaf_ids[i] = int(tree.apply(X_df.iloc[[i]], **(kw)))
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                leaf_ids[i] = -1
    raw_txt, merged_txt = [], []
    for i in range(n):
        txt = ""
        if hasattr(tree, "printRule"):
            variants = [
                dict(kwargs={"depth":max_depth, "node":getattr(tree,"Tree",None), "leaf_info_list":[]}),
                dict(kwargs={"max_depth":max_depth, "node":getattr(tree,"Tree",None), "leaf_info_list":[]}),
                dict(kwargs={"depth":max_depth}),
                dict(kwargs={"max_depth":max_depth}),
                dict(kwargs={}),
            ]
            for v in variants:
                try:
                    msgs = tree.printRule(X_df.iloc[[i]], **v["kwargs"])
                    txt = "\n".join([str(m) for m in msgs]) if isinstance(msgs, (list, tuple)) else str(msgs)
                    break
                except Exception:
                    continue
        raw_txt.append(txt)
        bounds = parse_constraints_from_path_text(txt)
        merged_txt.append(format_constraints_compact(bounds, max_items=24))
    df = pd.DataFrame({
        "sample_id": X_df.index,
        "leaf_id": leaf_ids,
        "path_text": raw_txt,
        "path_constraints": merged_txt
    })
    if save:
        df.to_csv(os.path.join(out_dir, "per_sample_rules.csv"), index=False)
        print("[saved] per_sample_rules.csv")
    return df, leaf_ids

# ========== Leaf → Cluster adı ==========
def make_cluster_names(leaf_ids: np.ndarray) -> dict:
    lids = np.asarray(leaf_ids).astype(int)
    uniq = np.unique(lids)
    sizes = [(lid, int(np.sum(lids == lid))) for lid in uniq]
    sizes.sort(key=lambda x: -x[1])
    cluster_names = {}
    for rank, (lid, _) in enumerate(sizes, start=1):
        cluster_names[int(lid)] = f"C{rank}"
    return cluster_names

# ========== 3×1 (Üst=ART, Alt=PRT) + altta kurallar ==========
def plot_per_leaf_series_3x1_world3(
    Y_population, Y_industrial, Y_pollution, t, leaf_ids, rules_df, out_dir,
    normalize_series=True,
    pop_label="Population",
    ind_label="Industrial Output",
    pol_label="Persistent Pollution",
    title_fontsize=16,
    line_width=1.8,
    text_fontsize=9,
    show_mean=True,
    show_median=False,
    per_series_alpha=0.18,
    per_series_lw=0.8,
    max_rules_items=24,
    cluster_names=None,
):
    outp = os.path.join(out_dir, "leaf_plots")
    os.makedirs(outp, exist_ok=True)

    # ---------- helpers ----------
    def minmax_norm_rows(A):
        A = np.asarray(A, dtype=float)
        mn = A.min(axis=1, keepdims=True)
        mx = A.max(axis=1, keepdims=True)
        span = np.where((mx - mn) == 0.0, 1.0, (mx - mn))
        return (A - mn) / span

    def _merge_all_from_texts(texts):
        merged = {}
        for s in (texts or []):
            for name, b in parse_constraints_from_path_text(str(s)).items():
                cur = merged.get(name, {"low": -np.inf, "low_inc": False,
                                         "high": np.inf, "high_inc": False})
                if (b["low"] > cur["low"]) or (b["low"] == cur["low"] and b["low_inc"] and not cur["low_inc"]):
                    cur["low"], cur["low_inc"] = b["low"], b["low_inc"]
                if (b["high"] < cur["high"]) or (b["high"] == cur["high"] and b["high_inc"] and not cur["high_inc"]):
                    cur["high"], cur["high_inc"] = b["high"], b["high_inc"]
                merged[name] = cur
        return merged

    # ---------- normalize ----------
    Yp = minmax_norm_rows(Y_population)  if normalize_series else np.asarray(Y_population, dtype=float)
    Yi = minmax_norm_rows(Y_industrial)  if normalize_series else np.asarray(Y_industrial, dtype=float)
    Yl = minmax_norm_rows(Y_pollution)   if normalize_series else np.asarray(Y_pollution,  dtype=float)

    lids = np.unique(np.asarray(leaf_ids).astype(int))

    # ---------- rules per leaf ----------
    leaf_to_rules = {}
    if rules_df is not None and {"leaf_id", "path_text"} <= set(rules_df.columns):
        tmp = rules_df.copy()
        tmp["leaf_id"] = tmp["leaf_id"].astype(int)
        for lid in lids:
            pt = tmp[tmp["leaf_id"] == int(lid)]["path_text"].astype(str).tolist()
            merged = _merge_all_from_texts(pt)
            leaf_to_rules[int(lid)] = format_constraints_compact(
                merged, max_items=max_rules_items
            )

    # ---------- per leaf plotting ----------
    for lid in lids:
        idxs = np.where(np.asarray(leaf_ids).astype(int) == int(lid))[0]
        if idxs.size == 0:
            continue

        fig, (ax1, ax2, ax3, ax_txt) = plt.subplots(
            4, 1,
            figsize=(11.8, 12.2),
            gridspec_kw={"height_ratios": [2.3, 2.3, 2.3, 1.3]},
            constrained_layout=True
        )

        cname = cluster_names.get(int(lid), f"leaf {int(lid)}") if cluster_names else f"leaf {int(lid)}"

        # ===== POPULATION =====
        Y_leaf = Yp[idxs]
        ax1.plot(t, Y_leaf.T, linewidth=per_series_lw, alpha=per_series_alpha)
        mean = Y_leaf.mean(axis=0)
        if show_mean:
            ax1.plot(t, mean, linewidth=line_width, alpha=0.96)
        if show_median:
            ax1.plot(t, np.median(Y_leaf, axis=0),
                     linewidth=line_width * 0.9, linestyle="--", alpha=0.85)
        ax1.set_ylabel(pop_label)
        ax1.set_title(f"{cname} (leaf {int(lid)}) — size={len(idxs)}",
                      fontsize=title_fontsize, pad=8)
        ax1.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)
        ax1.tick_params(labelbottom=False)

        stats1 = f"{pop_label} — first={mean[0]:.6g} | peak={mean.max():.6g} | last={mean[-1]:.6g}"

        # ===== INDUSTRIAL OUTPUT =====
        Y_leaf = Yi[idxs]
        ax2.plot(t, Y_leaf.T, linewidth=per_series_lw, alpha=per_series_alpha)
        mean = Y_leaf.mean(axis=0)
        if show_mean:
            ax2.plot(t, mean, linewidth=line_width, alpha=0.96)
        if show_median:
            ax2.plot(t, np.median(Y_leaf, axis=0),
                     linewidth=line_width * 0.9, linestyle="--", alpha=0.85)
        ax2.set_ylabel(ind_label)
        ax2.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)
        ax2.tick_params(labelbottom=False)

        stats2 = f"{ind_label} — first={mean[0]:.6g} | peak={mean.max():.6g} | last={mean[-1]:.6g}"

        # ===== POLLUTION =====
        Y_leaf = Yl[idxs]
        ax3.plot(t, Y_leaf.T, linewidth=per_series_lw, alpha=per_series_alpha)
        mean = Y_leaf.mean(axis=0)
        if show_mean:
            ax3.plot(t, mean, linewidth=line_width, alpha=0.96)
        if show_median:
            ax3.plot(t, np.median(Y_leaf, axis=0),
                     linewidth=line_width * 0.9, linestyle="--", alpha=0.85)
        ax3.set_ylabel(pol_label)
        ax3.set_xlabel("Time")
        ax3.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)

        stats3 = f"{pol_label} — first={mean[0]:.6g} | peak={mean.max():.6g} | last={mean[-1]:.6g}"

        # ===== RULES =====
        ax_txt.axis("off")
        stats_block = (
            "Stats (means)\n"
            f"• {stats1}\n"
            f"• {stats2}\n"
            f"• {stats3}"
        )
        rules_text = leaf_to_rules.get(int(lid), "")
        full_block = stats_block + (("\n\nRules (merged)\n" + rules_text) if rules_text else "")

        ax_txt.text(
            0.02, 0.98, full_block,
            transform=ax_txt.transAxes,
            ha="left", va="top",
            fontsize=text_fontsize,
            family="monospace"
        )

        suffix = "_norm" if normalize_series else "_real"
        fpath = os.path.join(outp, f"leaf_{int(lid)}_WORLD3_3x1{suffix}.png")
        fig.savefig(fpath, dpi=220)
        plt.close(fig)
        print(f"[saved] {fpath}")

def plot_pcp_cluster_envelopes_raw(
    X_df: pd.DataFrame,
    leaf_ids: np.ndarray,
    cluster_names: dict,
    out_dir: str,
    fname: str = "pcp_cluster_envelopes_raw.png",
    figsize=(12, 7),
    lw_minmax: float = 2.6,
    lw_median: float = 2.2,
    alpha_minmax: float = 0.95,
    alpha_median: float = 0.9,
    show_median: bool = True,
    show_iqr_band: bool = False,   # istersen Q1-Q3 band
    pad_frac: float = 0.06,
    max_clusters: int | None = None,   # None = hepsi, yoksa en büyük K cluster
):
    """
    Cluster-envelope Parallel Coordinates (RAW-scale axes, geometry normalized per feature)

    Her cluster için:
      - min çizgisi (feature bazında)
      - max çizgisi
      - opsiyonel median çizgisi
      - (opsiyonel) IQR band (Q1-Q3)

    NOT:
      - Eksen geometrisi için [0,1] normalizasyon yapılır (feature bazında).
      - Tick etiketleri RAW değerlerdir.
    """
    os.makedirs(out_dir, exist_ok=True)

    cols = X_df.columns.tolist()
    X = X_df[cols].to_numpy(dtype=float)
    xs = np.arange(len(cols))

    # --- global RAW min/max for axis scaling + ticks ---
    x_min = X_df[cols].min(axis=0).astype(float)
    x_max = X_df[cols].max(axis=0).astype(float)
    span = (x_max - x_min).replace(0, np.nan)

    # pad
    for c in cols:
        mn, mx = float(x_min[c]), float(x_max[c])
        if not np.isfinite(mx - mn) or mx == mn:
            dv = 1.0 if mn == 0 else abs(mn) * 0.1
            mn -= dv; mx += dv
        p = (mx - mn) * pad_frac
        x_min[c] = mn - p
        x_max[c] = mx + p

    def norm_val(col, v):
        mn = float(x_min[col]); mx = float(x_max[col])
        sp = mx - mn if mx > mn else 1.0
        return (v - mn) / sp

    # --- labels (leaf -> Ck) ---
    labels = np.array([cluster_names.get(int(l), f"C{int(l)}") for l in leaf_ids])
    uniq = np.unique(labels)

    # --- cluster sizes: büyükten küçüğe sırala ---
    sizes = {cl: int(np.sum(labels == cl)) for cl in uniq}
    uniq_sorted = sorted(uniq, key=lambda cl: -sizes[cl])
    if max_clusters is not None:
        uniq_sorted = uniq_sorted[:int(max_clusters)]

    # --- colors ---
    cmap = plt.cm.tab10
    color_map = {cl: cmap(i % 10) for i, cl in enumerate(uniq_sorted)}

    fig, ax = plt.subplots(figsize=figsize)

    # --- draw vertical axes + RAW ticks ---
    for i, col in enumerate(cols):
        # main vertical axis
        ax.axvline(i, color="black", linewidth=1.6, zorder=3)

        ymin, ymax = float(x_min[col]), float(x_max[col])
        ticks = np.linspace(ymin, ymax, 6)
        for t in ticks:
            y = norm_val(col, t)
            ax.plot([i - 0.04, i], [y, y], color="black", linewidth=1.0, zorder=4)
            ax.text(i - 0.06, y, f"{t:.3g}", ha="right", va="center", fontsize=9)

    # --- plot envelopes per cluster ---
    for cl in uniq_sorted:
        idxs = np.where(labels == cl)[0]
        Xc = X[idxs, :]  # RAW

        # per-feature stats (RAW)
        vmin = np.nanmin(Xc, axis=0)
        vmax = np.nanmax(Xc, axis=0)
        vmed = np.nanmedian(Xc, axis=0)
        q1 = np.nanquantile(Xc, 0.25, axis=0)
        q3 = np.nanquantile(Xc, 0.75, axis=0)

        # normalize for geometry
        ymin_line = np.array([norm_val(cols[j], vmin[j]) for j in range(len(cols))])
        ymax_line = np.array([norm_val(cols[j], vmax[j]) for j in range(len(cols))])
        ymed_line = np.array([norm_val(cols[j], vmed[j]) for j in range(len(cols))])
        yq1 = np.array([norm_val(cols[j], q1[j]) for j in range(len(cols))])
        yq3 = np.array([norm_val(cols[j], q3[j]) for j in range(len(cols))])

        colr = color_map[cl]

        # min/max lines
        ax.plot(xs, ymin_line, color=colr, linewidth=lw_minmax, alpha=alpha_minmax, zorder=2)
        ax.plot(xs, ymax_line, color=colr, linewidth=lw_minmax, alpha=alpha_minmax, zorder=2)

        # optional iqr band
        if show_iqr_band:
            ax.fill_between(xs, yq1, yq3, color=colr, alpha=0.12, zorder=1)

        # median
        if show_median:
            ax.plot(xs, ymed_line, color=colr, linewidth=lw_median, alpha=alpha_median,
                    linestyle="--", zorder=3)

    # --- x labels ---
    pretty_cols = [c.replace("_", "\n") for c in cols]
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty_cols, fontsize=9)

    # --- layout cleanup ---
    ax.set_xlim(-0.4, len(cols) - 0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- title ---
    title = "Parallel Coordinates — Cluster envelopes (min/max) + median (RAW-scale axes)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=22)

    # --- legend (bottom) ---
    legend_elements = [
        Line2D([0], [0], color=color_map[cl], lw=3,
               label=f"{cl} (n={sizes[cl]})")
        for cl in uniq_sorted
    ]
    leg = ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=min(len(uniq_sorted), 6),
        frameon=True,
        fontsize=10,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_edgecolor("black")

    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# -------- Numberline / Feature distribution ----------
def plot_numberlines_from_df(
    X: pd.DataFrame,
    out_path: str,
    max_params: int = 8,
    layout_cols: int = 2,
    dpi: int = 240,
    pad_frac: float = 0.05,
    show_iqr: bool = True,
):
    import math
    rng = np.random.default_rng(42)
    cols = list(X.columns[:max_params]) if max_params else list(X.columns)
    n = len(cols)
    rows = math.ceil(n / layout_cols) if layout_cols else 1
    mins = X[cols].min(axis=0).astype(float)
    maxs = X[cols].max(axis=0).astype(float)
    for c in cols:
        if mins[c] == maxs[c]:
            dv = 1.0 if mins[c] == 0 else abs(mins[c]) * 0.1
            mins[c] -= dv; maxs[c] += dv
        span = maxs[c] - mins[c]
        mins[c] -= span * 0.05
        maxs[c] += span * 0.05
    fig, axes = plt.subplots(rows, layout_cols, figsize=(10, 2.4*rows + 1.4), constrained_layout=True)
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])
    for k, c in enumerate(cols):
        ax = axes[k]
        v = X[c].astype(float).to_numpy()
        lo, hi = float(mins[c]), float(maxs[c])
        for sp in ("left","right","top","bottom"):
            ax.spines[sp].visible = True
        ax.spines["left"].visible = False
        ax.spines["right"].visible = False
        ax.yaxis.set_visible(False)
        ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.hlines(0, lo, hi, linewidth=2)
        if show_iqr and v.size:
            q1, q3 = np.quantile(v, [0.25, 0.75])
            ax.axvspan(q1, q3, ymin=0.36, ymax=0.64, alpha=0.22)
        if v.size:
            med = float(np.median(v))
            ax.vlines(med, -0.22, 0.22, linewidth=1.6)
        if v.size:
            jitter = (rng.random(len(v)) - 0.5) * 0.16
            ax.scatter(v, jitter, s=18, alpha=0.9, edgecolors="none")
            meanv = float(np.mean(v))
            ax.plot([meanv], [0], marker="o", markersize=7)
        ax.text(lo, -0.46, f"{lo:.4g}", ha="left",  va="top", fontsize=8)
        ax.text(hi, -0.46, f"{hi:.4g}", ha="right", va="top", fontsize=8)
        ax.set_xlabel(str(c), fontsize=10)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xlim(lo, hi)
        ax.add_patch(plt.Rectangle((lo, -0.5),  hi-lo, 1.0, fill=False, linewidth=0.8, alpha=0.9))
    for r in range(n, rows*layout_cols):
        axes[r].axis("off")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[saved] {out_path}")

def plot_numberlines_by_cluster(X, leaf_ids, out_dir, max_params=8, layout_cols=2):
    outp = os.path.join(out_dir, "numberlines_by_cluster")
    os.makedirs(outp, exist_ok=True)
    unique_leaves = np.unique(leaf_ids)
    for rank, lid in enumerate(unique_leaves, start=1):
        idxs = np.where(leaf_ids == lid)[0]
        if idxs.size == 0:
            continue
        X_sub = X.iloc[idxs].copy()
        out_path = os.path.join(outp, f"leaf_{int(lid)}_C{rank}_numberlines.png")
        plot_numberlines_from_df(X_sub, out_path, max_params=max_params, layout_cols=layout_cols)

# ================== EXCEL LOGGING ==================
def log_dtw_reorder_to_excel(dist_df, leaf_ids, rules_df=None, out_path="dtw_reordered_log.xlsx"):
    uniq = np.unique(leaf_ids)
    clusters_raw = [list(np.where(leaf_ids == lid)[0]) for lid in uniq]
    sizes = [len(c) for c in clusters_raw]
    order = np.argsort([-s for s in sizes])
    ordered_clusters = [sorted(clusters_raw[i]) for i in order]
    ordered_labels = [dist_df.index[s] for cl in ordered_clusters for s in cl]
    D = dist_df.loc[ordered_labels, ordered_labels]

    # xlsxwriter yoksa openpyxl fallback
    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(out_path, engine=engine) as writer:
        rows = [{"cluster_id": f"C{k}", "size": len(cl),
                 "members": ",".join(map(lambda z: str(dist_df.index[z]), cl))}
                for k, cl in enumerate(ordered_clusters, start=1)]
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="clusters")

        def _intra_mean(df, idxs):
            if len(idxs) <= 1:
                return 0.0
            sub = df.values[np.ix_(idxs, idxs)]
            triu = sub[np.triu_indices_from(sub, k=1)]
            return float(np.mean(triu)) if triu.size else 0.0
        rows2 = [{"cluster_id": f"C{k}", "size": len(cl), "intra_mean": _intra_mean(dist_df, cl)}
                 for k, cl in enumerate(ordered_clusters, start=1)]
        pd.DataFrame(rows2).to_excel(writer, index=False, sheet_name="cluster_stats")

        smp = pd.DataFrame({"sample_id": dist_df.index, "leaf_id": leaf_ids})
        if rules_df is not None:
            rule_txt = None
            if "path_constraints" in rules_df.columns:
                rule_txt = rules_df["path_constraints"].astype(str)
            elif "path_text" in rules_df.columns:
                rule_txt = rules_df["path_text"].astype(str).map(
                    lambda s: format_constraints_compact(parse_constraints_from_path_text(s), max_items=999)
                )
            if rule_txt is not None:
                smp = smp.merge(pd.DataFrame({"sample_id": rules_df["sample_id"], "rule_short": rule_txt}),
                                on="sample_id", how="left")
        smp.to_excel(writer, index=False, sheet_name="samples")

        pd.DataFrame({"ordered_labels": ordered_labels}).to_excel(writer, index=False, sheet_name="order")

        D_rounded = D.round(2)
        D_rounded.to_excel(writer, sheet_name="dtw_reordered", index=True)

        # süslemeler yalnızca xlsxwriter ile
        if engine == "xlsxwriter":
            wb, ws = writer.book, writer.sheets["dtw_reordered"]
            n = D.shape[0]
            start_row, start_col = 1, 1
            end_row, end_col = start_row + n - 1, start_col + n - 1
            ws.conditional_format(start_row, start_col, end_row, end_col, {
                "type": "3_color_scale",
                "min_color": "#1f3b4d",
                "mid_color": "#2f8fab",
                "max_color": "#f1e05a",
            })
            border_fmt = wb.add_format({"top": 2, "bottom": 2, "left": 2, "right": 2})
            cur = 0
            for cl in ordered_clusters:
                size = len(cl)
                r0 = start_row + cur
                c0 = start_col + cur
                r1 = r0 + size - 1
                c1 = c0 + size - 1
                ws.conditional_format(r0, c0, r1, c1, {"type": "no_errors", "format": border_fmt})
                cur += size
            numfmt = wb.add_format({"num_format": "0.00"})
            ws.set_column(0, 0, 12)
            ws.set_column(start_col, end_col, 6, numfmt)
    print(f"[saved] {out_path}")
    return ordered_labels, ordered_clusters, D

# --------------- EVRENSEL OKUYUCU + LOADER (DÜZELTİLMİŞ) ---------------
def _drop_indexish(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Desteklenmeyen uzantı: {ext} ({path})")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Tablo formatı bekleniyordu: {path}")
    return _drop_indexish(df)

def _extract_timeseries_matrix(dfo: pd.DataFrame, n_expected: int):
    """
    dfo: art_full_df / prt_full_df gibi tablo.
    - run / sample_id / id / index gibi index sütunlarını ayıklar.
    - Zaman başlıklarını (0.0, 0.125, ...) float'a çevirir ve yalnızca onları alır.
    - Hiçbiri float değilse TIME_STEP ile sentetik t üretir.
    Döndürür: Y (N x T) numpy, t (T,) numpy
    """
    df = dfo.copy()

    # 1) index/kimlik kolonlarını topla
    idx_like = []
    for c in list(df.columns):
        lc = str(c).strip().lower()
        if lc in ("run", "sample_id", "id", "index"):
            idx_like.append(c)
    if idx_like:
        c0 = idx_like[0]
        vals = df[c0].values
        try:
            if len(np.unique(vals)) == len(vals):
                df = df.set_index(c0, drop=True)
        except Exception:
            pass
        df = df.drop(columns=idx_like, errors="ignore")

    # 2) float'a çevrilebilen zaman sütunlarını seç
    floatable = []
    for c in df.columns:
        try:
            float(c)
            floatable.append(c)
        except Exception:
            continue

    if floatable:  # gerçek zaman başlıklarını kullan
        t = np.array([float(c) for c in floatable], dtype=float)
        df = df[floatable].copy()
        df.columns = t
    else:
        # Uç durum: hepsi isimsel → sentetik zaman
        T = df.shape[1]
        t = TIME_START + np.arange(T, dtype=float) * TIME_STEP
        df.columns = t

    # 3) satır sayısı kontrolü
    if df.shape[0] != n_expected:
        raise ValueError(f"Satır sayısı uyumsuz: params N={n_expected}, tablo N={df.shape[0]}")

    # 4) zaman adımı hakkında bilgi
    if len(t) >= 3:
        dts = np.unique(np.round(np.diff(t), 9))
        if len(dts) != 1 or not np.isclose(dts[0], TIME_STEP):
            print(f"[warn] Zaman adımı dosyada {dts} (grafikte dosya zamanları kullanılacak).")

    return df.to_numpy(dtype=float), t

def load_from_tables(params_csv: str, features, df_paths: dict):
    # === X (parametreler)
    dfp = pd.read_csv(params_csv)
    dfp = _drop_indexish(dfp)

    if features is None:
        num = dfp.select_dtypes(include=[np.number]).columns.tolist()
        if not num:
            raise ValueError("params.csv içinde sayısal kolon yok; FEATURES belirtin.")
        X = dfp[num].copy()
    else:
        for f in features:
            if f not in dfp.columns:
                raise ValueError(f"Feature '{f}' params.csv içinde yok.")
        X = dfp[list(features)].copy()

    N = len(X)
    axis_names = list(df_paths.keys())
    Y_list, t_common = [], None

    # === Her ekseni oku ve temizle
    for ax in axis_names:
        raw = _read_any(df_paths[ax])
        raw = _drop_indexish(raw)
        Y_axis, t_axis = _extract_timeseries_matrix(raw, n_expected=N)

        if t_common is None:
            t_common = t_axis
        else:
            if len(t_axis) != len(t_common):
                raise ValueError(f"T uzunluğu eksenler arasında eşit olmalı: {len(t_common)} vs {len(t_axis)} ({ax})")
        Y_list.append(Y_axis)

    # === (N, A, T)
    Y_mv = np.stack(Y_list, axis=1)   # axis sırası: axis_names
    t = t_common

    # X index → run id (0..N-1)
    X = X.copy()
    X.index = np.arange(N)

    print(f"[load] N={N} runs, A={len(axis_names)} axes, T={len(t)} steps, features={list(X.columns)}")
    return X, Y_mv, t, axis_names, None

# --------- Ek görselleştirmeler (opsiyonel) ----------
def plot_dtw_reordered_heatmap(dist_df, leaf_ids, out_dir, fname="dtw_reordered_heatmap.png"):
    uniq = np.unique(leaf_ids)
    clusters_raw = [list(np.where(leaf_ids == lid)[0]) for lid in uniq]
    sizes = [len(c) for c in clusters_raw]
    order = np.argsort([-s for s in sizes])
    ordered_clusters = [sorted(clusters_raw[i]) for i in order]
    ordered_labels = [dist_df.index[s] for cl in ordered_clusters for s in cl]
    D = dist_df.loc[ordered_labels, ordered_labels].values
    plt.figure(figsize=(7, 6.5))
    plt.imshow(D, aspect='auto', interpolation='nearest')
    plt.colorbar(label="DTW distance")
    plt.title("DTW distance heatmap (cluster ordered)")
    cur = 0
    for cl in ordered_clusters:
        sz = len(cl)
        plt.gca().add_patch(plt.Rectangle((cur-0.5, cur-0.5), sz, sz, fill=False, linewidth=1.2))
        cur += sz
    plt.xticks([]); plt.yticks([])
    path = os.path.join(out_dir, fname)
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close()
    print(f"[saved] {path}")

# --------- CLUSTER ÜYELERİNİ DIŞA AKTAR ----------
def select_cluster_indices(leaf_ids, cluster_names: dict, select=None, leaf_id=None):
    lids = np.asarray(leaf_ids).astype(int)
    if leaf_id is not None:
        return np.where(lids == int(leaf_id))[0]
    if isinstance(select, str) and select:
        target_lids = [lid for lid, cname in cluster_names.items() if cname == select]
        if not target_lids:
            return np.array([], dtype=int)
        mask = np.zeros_like(lids, dtype=bool)
        for lid in target_lids:
            mask |= (lids == int(lid))
        return np.where(mask)[0]
    return np.array([], dtype=int)

def export_cluster_member_params(X_df: pd.DataFrame, idxs: np.ndarray, out_dir: str, cluster_label: str):
    os.makedirs(out_dir, exist_ok=True)
    sub = X_df.iloc[idxs].copy()
    sub.insert(0, "sample_id", sub.index)
    csv_path  = os.path.join(out_dir, f"{cluster_label}_params.csv")
    xlsx_path = os.path.join(out_dir, f"{cluster_label}_params.xlsx")
    sub.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        sub.to_excel(writer, index=False, sheet_name="params")
    print(f"[saved] {csv_path}")
    print(f"[saved] {xlsx_path}")

def export_cluster_member_plots(
    Y_mv,
    t,
    idxs: np.ndarray,
    axis_names,
    out_dir: str,
    cluster_label: str,
    normalize=False,
    show_cluster_mean=True
):
    if idxs.size == 0:
        print("[warn] export_cluster_member_plots: boş indeks kümesi")
        return

    base_dir = os.path.join(out_dir, f"cluster_members_{cluster_label}{'_norm' if normalize else '_real'}")
    os.makedirs(base_dir, exist_ok=True)

    name_to_idx = {name.upper(): i for i, name in enumerate(axis_names)}
    i_pop = name_to_idx["POPULATION"]
    i_ind = name_to_idx["INDUSTRIAL_OUTPUT"]
    i_pol = name_to_idx["PERSISTENT_POLLUTION"]

    Y_pop_all = Y_mv[:, i_pop, :]
    Y_ind_all = Y_mv[:, i_ind, :]
    Y_pol_all = Y_mv[:, i_pol, :]

    mean_pop = Y_pop_all[idxs].mean(axis=0) if show_cluster_mean else None
    mean_ind = Y_ind_all[idxs].mean(axis=0) if show_cluster_mean else None
    mean_pol = Y_pol_all[idxs].mean(axis=0) if show_cluster_mean else None

    def _row_minmax(x):
        x = np.asarray(x, dtype=float)
        mn, mx = float(np.min(x)), float(np.max(x))
        span = mx - mn if mx > mn else 1.0
        return (x - mn) / span

    for gi in idxs:
        yp = Y_pop_all[gi].astype(float)
        yi = Y_ind_all[gi].astype(float)
        yl = Y_pol_all[gi].astype(float)

        m_pop, m_ind, m_pol = mean_pop, mean_ind, mean_pol

        if normalize:
            yp = _row_minmax(yp)
            yi = _row_minmax(yi)
            yl = _row_minmax(yl)
            if show_cluster_mean and mean_pop is not None:
                m_pop = _row_minmax(mean_pop)
                m_ind = _row_minmax(mean_ind)
                m_pol = _row_minmax(mean_pol)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11.2, 9.4), constrained_layout=True)

        ax1.plot(t, yp, linewidth=2.2)
        if show_cluster_mean and m_pop is not None:
            ax1.plot(t, m_pop, linewidth=2.6, alpha=0.9)
        ax1.set_title(f"{cluster_label} — sample_idx={gi} ({'norm' if normalize else 'real'})", pad=6)
        ax1.set_ylabel("Population")
        ax1.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)
        ax1.tick_params(labelbottom=False)

        ax2.plot(t, yi, linewidth=2.2)
        if show_cluster_mean and m_ind is not None:
            ax2.plot(t, m_ind, linewidth=2.6, alpha=0.9)
        ax2.set_ylabel("Industrial Output")
        ax2.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)
        ax2.tick_params(labelbottom=False)

        ax3.plot(t, yl, linewidth=2.2)
        if show_cluster_mean and m_pol is not None:
            ax3.plot(t, m_pol, linewidth=2.6, alpha=0.9)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Persistent Pollution")
        ax3.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)

        fpath = os.path.join(base_dir, f"{cluster_label}_sample_{gi:04d}.png")
        fig.savefig(fpath, dpi=220)
        plt.close(fig)
        print(f"[saved] {fpath}")

# ------------------------- MAIN FLOW -----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # === 1) DTW/AĞAÇ VERİSİ: dtw_df'ler ===
    X_dtw, Y_mv_raw_dtw, t_dtw, axis_names_dtw, _ = load_from_tables(PARAMS_CSV, FEATURES, DF_TABLES_DTW)

    # === 2) GRAFİK VERİSİ: full_df'ler ===
    X_full, Y_mv_raw_full, t_full, axis_names_full, _ = load_from_tables(PARAMS_CSV, FEATURES, DF_TABLES_FULL)

    # Tutarlılık kontrolleri
    if set(axis_names_dtw) != set(axis_names_full):
        raise ValueError("DTW ve FULL eksen setleri farklı.")
    axis_names = axis_names_dtw
    assert X_dtw.shape[0] == X_full.shape[0], "DTW ve FULL satır sayıları farklı."
    X = X_dtw.copy()  # (parametreler aynı)
    sample_labels = X.index.to_numpy()

    print(f"[flow] DTW/ağaç: {len(sample_labels)} run | Plotlar: FULL zaman serileri ile.")

    # DTW için normalize et
    Y_mv_dtw = normalize_outputs(Y_mv_raw_dtw, Y_NORM_MODE)

    # 3) DTW (1D ve MV)
    if Y_mv_dtw.shape[1] == 1:
        print("[info] Single axis detected → 1D DTW")
        dist_df = dtw_pairwise_1d(Y_mv_dtw[:, 0, :], labels=sample_labels)
        axis_dtws = {str(axis_names_dtw[0]).upper(): dist_df}
    else:
        print("[info] Multiple axes detected → MV-DTW (sum of per-axis DTW)")
        dist_df, axis_dtws = dtw_pairwise_multi_stage2(
            Y_mv_dtw,
            axis_weights=None,
            axis_names=axis_names_dtw,
            return_axis_dtws=True,
            row_norm="none",
            labels=sample_labels
        )

    # 4) Ağaç ve kurallar

    tree = TreeForecast(
        target_type='multi', max_features=None, max_target=1, max_depth=5,
        min_samples_leaf=20, min_samples_split=50, split_style='custom',
        target_diff=False, lambda_decay=0.5, obj_weights = np.array([1.0, 0.1, 0.1]),
        verbose=False
    )
    print("[debug] X shape:", X.shape)
    print("[debug] X dtypes:\n", X.dtypes)
    print("[debug] X NaNs:\n", X.isna().sum())
    print("[debug] X nunique:\n", X.nunique(dropna=False))

    print("[debug] fitting tree...")
    tree.fit(X, dist_df)

    # root terminal mı?
    print("[debug] root is_terminal:", getattr(tree.Tree, "is_terminal", None))

    # kaç split var?
    try:
        spl = tree.splits(tree.Tree, [])
        print("[debug] n_splits:", len(spl))
        if len(spl) > 0:
            print("[debug] first split:", spl[0])
    except Exception as e:
        print("[debug] splits() failed:", e)

    # leaf sayısı kaç?
    leaf_ids_dbg = np.asarray(tree.apply(X, depth=tree.max_depth)).astype(int).reshape(-1)
    u, c = np.unique(leaf_ids_dbg, return_counts=True)
    print("[debug] unique leaf count:", len(u))
    print("[debug] leaf sizes:", list(zip(u.tolist(), c.tolist()))[:30])

    print("[info] Ağaç eğitiliyor (DTW ile)...")

    print_and_save_splits(tree, list(X.columns), OUT_DIR, save=SAVE_SPLITS_CSV)
    DEPTH_FOR_LEAF = tree.max_depth  # veya 15
    rules_df, leaf_ids = per_sample_rules(tree, X, OUT_DIR, max_depth=DEPTH_FOR_LEAF, save=SAVE_PER_SAMPLE_RULES_CSV)

    # Cluster adları
    cluster_names = make_cluster_names(leaf_ids)
    # === Stage-1 lever'ların TAMAMI (RAW) ===
    df_params_all = pd.read_csv(PARAMS_CSV)

    # sadece Stage-1 lever'ları seç
    stage1_cols = [
        "desired_completed_family_size_normal",
        "lifetime_perception_delay",
        "social_adjustment_delay",
        "desired_food_ratio",
        "food_shortage_perception_delay",
        "industrial_capital_output_ratio_1",
    ]

    df_stage1 = df_params_all[stage1_cols].copy()

    # indexleri leaf_ids ile hizala
    df_stage1.index = X.index

    plot_pcp_cluster_envelopes_raw(
        X_df=df_stage1,
        leaf_ids=leaf_ids,
        cluster_names=cluster_names,
        out_dir=OUT_DIR,
        fname="pcp_cluster_envelopes_raw.png",
        show_median=True,
        show_iqr_band=False,
        max_clusters=10
    )

    # 5) 3×1 grafikler: WORLD3 FULL seriler (real ve z-score)

    axis_names = axis_names_full
    Y_mv_full_real = Y_mv_raw_full
    Y_mv_full_std = normalize_outputs(Y_mv_raw_full, 'standardize')

    # --- WORLD3 eksen indeksleri ---
    name_to_idx = {name.upper(): i for i, name in enumerate(axis_names)}

    # ZORUNLU: WORLD3 stokları
    i_pop = name_to_idx.get("POPULATION")
    i_ind = name_to_idx.get("INDUSTRIAL_OUTPUT")
    i_pol = name_to_idx.get("PERSISTENT_POLLUTION")

    missing = [k for k, v in {
        "POPULATION": i_pop,
        "INDUSTRIAL_OUTPUT": i_ind,
        "PERSISTENT_POLLUTION": i_pol
    }.items() if v is None]

    if missing:
        raise ValueError(f"WORLD3 için gerekli stoklar bulunamadı: {missing}")

    # --- REAL ---
    plot_per_leaf_series_3x1_world3(
        Y_population=Y_mv_full_real[:, i_pop, :],
        Y_industrial=Y_mv_full_real[:, i_ind, :],
        Y_pollution=Y_mv_full_real[:, i_pol, :],
        t=t_full,
        leaf_ids=leaf_ids,
        rules_df=rules_df,
        out_dir=os.path.join(OUT_DIR, "plots_real"),
        normalize_series=False,
        pop_label="Population (real)",
        ind_label="Industrial Output (real)",
        pol_label="Persistent Pollution (real)",
        show_mean=True,
        show_median=False,
        max_rules_items=24,
        cluster_names=cluster_names
    )

    # --- Z-SCORE ---
    plot_per_leaf_series_3x1_world3(
        Y_population=Y_mv_full_std[:, i_pop, :],
        Y_industrial=Y_mv_full_std[:, i_ind, :],
        Y_pollution=Y_mv_full_std[:, i_pol, :],
        t=t_full,
        leaf_ids=leaf_ids,
        rules_df=rules_df,
        out_dir=os.path.join(OUT_DIR, "plots_zscore"),
        normalize_series=False,  # zaten standardize edildi
        pop_label="Population (z-score)",
        ind_label="Industrial Output (z-score)",
        pol_label="Persistent Pollution (z-score)",
        show_mean=True,
        show_median=False,
        max_rules_items=24,
        cluster_names=cluster_names
    )

    # --- Excel logları (DTW mesafesine göre) ---
    log_dtw_reorder_to_excel(
        dist_df, leaf_ids, rules_df,
        out_path=os.path.join(OUT_DIR, "dtw_reordered_log.xlsx")
    )
    for ax_name, D_ax in axis_dtws.items():
        log_dtw_reorder_to_excel(
            D_ax, leaf_ids, rules_df,
            out_path=os.path.join(OUT_DIR, f"dtw_{ax_name}_1d.xlsx")
        )

    try:
        plot_dtw_reordered_heatmap(dist_df, leaf_ids, OUT_DIR)
    except Exception as e:
        print("[warn] heatmap:", e)

if __name__ == "__main__":
    main()
