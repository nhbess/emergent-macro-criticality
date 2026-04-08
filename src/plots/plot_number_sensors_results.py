"""
n_sensors_changes analysis: build CSV once, plot any time.

Data methods  (slow, run once):
    make_data_criticality_sensors(root_path)  → criticality.csv

Plot methods  (fast, load CSV):
    plot_criticality_by_sensors(root_path, p_c_micro)  → criticality_summary.png
"""

import os
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.utils import _data_artifact_path, _plot_artifact_path, get_p_c_micro


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data method — criticality score from power-law fits
# ---------------------------------------------------------------------------

def make_data_criticality_sensors(root_path: str) -> None:
    """Build criticality.csv from per-experiment data_fits/summary.csv.

    Mirrors make_data_heatmap from analysis_vision_radius.py but groups by n_exteroceptor.
    Reads each experiment's summary.csv, attaches n_exteroceptor + vision_distance, and
    computes the criticality score:
        D = |α_S − 3/2| + |α_T − 2| + KS_S + KS_T
    (smaller D = better; D=0 means perfect mean-field power laws)
    Saves: criticality.csv
    """
    exp_dirs   = sorted(d for d in glob.glob(os.path.join(root_path, "*")) if os.path.isdir(d))
    rows_list  = []

    for exp_dir in exp_dirs:
        meta_path    = os.path.join(exp_dir, "metadata.yaml")
        summary_path = os.path.join(exp_dir, "data_fits", "summary.csv")
        if not os.path.exists(meta_path) or not os.path.exists(summary_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        n_ext = int(meta.get("n_exteroceptor", meta.get("n_sensors", 0)))
        vd    = int(meta.get("vision_distance", 0))

        df = pd.read_csv(summary_path)
        df.insert(0, "n_exteroceptor", n_ext)
        df.insert(1, "vision_distance", vd)
        rows_list.append(df)

    if not rows_list:
        raise RuntimeError(f"No data found under {root_path}")

    result = pd.concat(rows_list, ignore_index=True)

    alpha_s = result["alpha_size"].astype(float)
    alpha_d = result["alpha_duration"].astype(float)
    ks_s    = result["ks_size"].astype(float)
    ks_d    = result["ks_duration"].astype(float)
    result["D"] = (alpha_s - 1.5).abs() + (alpha_d - 2.0).abs() + ks_s + ks_d

    keep = [c for c in ["n_exteroceptor", "vision_distance", "p", "n_complete",
                         "alpha_size", "alpha_duration", "ks_size", "ks_duration", "D"]
            if c in result.columns]
    result = (result[keep]
              .sort_values(["vision_distance", "n_exteroceptor", "p"])
              .reset_index(drop=True))

    out = _data_artifact_path(root_path, "criticality.csv")
    result.to_csv(out, index=False)
    print(f"Saved {len(result)} rows -> {out}")


# ---------------------------------------------------------------------------
# Plot methods
# ---------------------------------------------------------------------------

def plot_criticality_by_sensors(root_path: str, p_c_micro: float | None = None) -> None:
    """Show how criticality quality changes with n_exteroceptor.

    Loads criticality.csv produced by make_data_criticality_sensors.

    Saves:
        media/<root basename>/<root basename>_criticality_summary.png

    Figure layout: two stacked panels across all vision distances.
      - Top: D_min vs number of sensors
      - Bottom: p* (argmin D) vs number of sensors
    """
    csv_path = _data_artifact_path(root_path, "criticality.csv")
    if not os.path.exists(csv_path):
        print("  [plot_criticality_by_sensors] criticality.csv not found "
              "— run make_data_criticality_sensors first")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("  [plot_criticality_by_sensors] empty CSV, skipping")
        return

    vd_vals  = sorted(df["vision_distance"].unique())
    vd_norm  = plt.Normalize(vmin=min(vd_vals), vmax=max(vd_vals))
    vd_cmap  = plt.colormaps["viridis"]

    sensor_ticks = sorted(df["n_exteroceptor"].unique())

    W = 5
    H = 3
    fig2, (ax_d, ax_p) = plt.subplots(2, 1, figsize=(W, H),
                                       sharex=True, layout="constrained")
    for vd in vd_vals:
        df_vd = df[df["vision_distance"] == vd]
        rows  = []
        for s in sorted(df_vd["n_exteroceptor"].unique()):
            grp = df_vd[df_vd["n_exteroceptor"] == s].dropna(subset=["D"])
            if grp.empty:
                continue
            best = grp.loc[grp["D"].idxmin()]
            rows.append({"n_exteroceptor": s,
                         "D_min":  float(best["D"]),
                         "p_star": float(best["p"])})
        if not rows:
            continue
        sd   = pd.DataFrame(rows).sort_values("n_exteroceptor")
        col  = vd_cmap(vd_norm(vd))
        ax_d.plot(sd["n_exteroceptor"], sd["D_min"],  color=col, lw=1.8, marker="o", ms=6)
        ax_p.plot(sd["n_exteroceptor"], sd["p_star"], color=col, lw=1.8, marker="o", ms=6)

    ax_d.set_ylabel(r"Best $\mathcal{L}$")
    ax_d.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    ax_p.set_ylabel(r"$p^*$")
    ax_p.set_xlabel("Number of sensors $(E)$")
    ax_p.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax_p.set_xticks(sensor_ticks)

    if p_c_micro is not None:
        ax_p.axhline(p_c_micro, linestyle="--", color="gray", linewidth=1,
                     label=r"$p_c^{\mathrm{micro}}$")
        ax_p.legend(fontsize=9, frameon=True)

    sm2 = plt.cm.ScalarMappable(cmap=vd_cmap, norm=vd_norm)
    sm2.set_array([])
    fig2.colorbar(sm2, ax=[ax_d, ax_p], label=r"$v_r$", fraction=0.04, pad=0.02, shrink=1.0)

    out2 = _plot_artifact_path(root_path, "criticality_summary.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {out2}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root_path = "experiments_data/n_sensors"
    p_c = get_p_c_micro(root_path)

    #make_data_criticality_sensors(root_path)
    plot_criticality_by_sensors(root_path, p_c_micro=p_c)
