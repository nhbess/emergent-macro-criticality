import os
import glob
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import yaml
import torch
from tqdm import tqdm
from scipy.special import zeta as _hurwitz_zeta

from metrics import summarize_metrics
from plots.utils import _data_artifact_path, _plot_artifact_path, get_p_c_micro


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Data methods
# ---------------------------------------------------------------------------

def make_data_heatmap(root_path: str) -> None:
    """Build summary_fits.csv from power-law fit results.

    For each experiment dir reads metadata.yaml + data_fits/summary.csv,
    attaches vision_distance, and computes the criticality score
        D = |α_S − 3/2| + |α_T − 2| + KS_S + KS_T
    Saves: summary_fits.csv
    """
    exp_dirs = sorted(d for d in glob.glob(os.path.join(root_path, "*")) if os.path.isdir(d))
    rows = []
    for exp_dir in exp_dirs:
        meta_path    = os.path.join(exp_dir, "metadata.yaml")
        summary_path = os.path.join(exp_dir, "data_fits", "summary.csv")
        if not os.path.exists(meta_path) or not os.path.exists(summary_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        df = pd.read_csv(summary_path)
        df.insert(0, "vision_distance", meta["vision_distance"])
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No experiment data found under {root_path}")

    result = pd.concat(rows, ignore_index=True)
    

    EUCLIDEAN = False
    if EUCLIDEAN:
        result["D"] = (
            (  (result["alpha_size"].astype(float)    - 1.5) ** 2
            + (result["alpha_duration"].astype(float) - 2.0) ** 2
            +  result["ks_size"].astype(float)               ** 2
            +  result["ks_duration"].astype(float)           ** 2
        ) ** 0.5
    )
    
    else:
        result["D"] = (
            (result["alpha_size"].astype(float) - 1.5).abs()
            + (result["alpha_duration"].astype(float) - 2.0).abs()
            + result["ks_size"].astype(float)
            + result["ks_duration"].astype(float)
        )
    cols = [c for c in ["p", "vision_distance", "alpha_size", "alpha_duration",
                        "ks_duration", "ks_size", "D"] if c in result.columns]
    result = result[cols].sort_values(["vision_distance", "p"]).reset_index(drop=True)

    out = _data_artifact_path(root_path, "summary_fits.csv")
    result.to_csv(out, index=False)
    print(f"Saved {len(result)} rows -> {out}")


def make_data_macrometrics(root_path: str) -> None:
    """Compute σ_eff, susceptibility, and activity entropy from state .npz files.

    For each experiment dir reads all data_states/p_*.npz files,
    averages over repetitions per p value.
    Saves: macro_metrics.csv  (one row per vision_distance × p)
    Columns: vision_distance, n_agents, n_neurons, p,
             sigma_eff_mean/std, susceptibility_mean/std, entropy_mean/std
    """
    exp_dirs = sorted(d for d in glob.glob(os.path.join(root_path, "*")) if os.path.isdir(d))
    rows = []

    for exp_dir in exp_dirs:
        meta_path  = os.path.join(exp_dir, "metadata.yaml")
        states_dir = os.path.join(exp_dir, "data_states")
        if not (os.path.exists(meta_path) and os.path.isdir(states_dir)):
            continue
        state_files = sorted(glob.glob(os.path.join(states_dir, "p_*.npz")))
        if not state_files:
            continue

        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        by_p: Dict[float, List[str]] = {}
        for path in state_files:
            d = np.load(path, allow_pickle=False)
            by_p.setdefault(float(d["p"]), []).append(path)
            d.close()

        vd        = int(meta["vision_distance"])
        n_agents  = int(meta.get("n_agents",  meta.get("n_agents_total",  0)))
        n_neurons = int(meta.get("n_neurons", meta.get("n_neurons_total", 0)))

        for p_val in tqdm(sorted(by_p), desc=f"vd={vd}"):
            sigs, chis, ents = [], [], []
            for path in by_p[p_val]:
                d      = np.load(path, allow_pickle=False)
                states = torch.from_numpy(d["states"].astype(np.float32))
                d.close()
                m = summarize_metrics(states)
                sigs.append(float(m["sigma_eff_mean"]))
                chis.append(float(m["susceptibility_mean"]))
                ents.append(float(m["activity_level_entropy_mean"]))
            rows.append({
                "vision_distance":     vd,
                "n_agents":            n_agents,
                "n_neurons":           n_neurons,
                "p":                   p_val,
                "sigma_eff_mean":      float(np.mean(sigs)),
                "sigma_eff_std":       float(np.std(sigs)),
                "susceptibility_mean": float(np.mean(chis)),
                "susceptibility_std":  float(np.std(chis)),
                "entropy_mean":        float(np.mean(ents)),
                "entropy_std":         float(np.std(ents)),
            })

    result = pd.DataFrame(rows).sort_values(["vision_distance", "p"]).reset_index(drop=True)
    out = _data_artifact_path(root_path, "macro_metrics.csv")
    result.to_csv(out, index=False)
    print(f"Saved {len(result)} rows -> {out}")


# ---------------------------------------------------------------------------
# Plot methods
# ---------------------------------------------------------------------------

def _kappa_params(root_path: str) -> tuple[int, float]:
    """Return (N, area) from the first metadata.yaml found under root_path.

    Used to convert vision distances to expected neighbour counts:
        κ(vr) = (N − 1) · π·vr² / area
    """
    for exp_dir in sorted(d for d in glob.glob(os.path.join(root_path, "*"))
                          if os.path.isdir(d)):
        meta_path = os.path.join(exp_dir, "metadata.yaml")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                m = yaml.safe_load(f)
            N    = int(m["n_agents"])
            area = float(m["map_size"][0]) * float(m["map_size"][1])
            return N, area
    raise FileNotFoundError(f"No metadata.yaml found under {root_path}")


def _analytic_kappa_per_vr(vr: np.ndarray, n_agents: int, area: float) -> np.ndarray:
    """Expected neighbour count on the toroidal map, κ(v_r) = (N − 1) π v_r² / area."""
    return (float(n_agents) - 1.0) * np.pi * np.asarray(vr, dtype=float) ** 2 / float(area)


def plot_heatmap(root_path: str, p_c_micro: float | None = None,
                 use_kappa: bool = False) -> None:
    """Load summary_fits.csv and plot the criticality heatmap.

    x = p^micro, y = vision_distance (or κ if use_kappa=True), color = D.
    Brighter = more critical.

    Saved as:
        criticality_heatmap.png       (use_kappa=False)
        criticality_heatmap_kappa.png (use_kappa=True)
    """

    #CMAP = 'plasma_r'
    CMAP = 'inferno_r'
    #CMAP = 'magma_r'
    #CMAP = 'viridis_r'

    df = pd.read_csv(_data_artifact_path(root_path, "summary_fits.csv"))

    ONLY_10 = False # if this is true it will only plot multiple of 10 for the vision distance
    if ONLY_10:
        df = df[df["vision_distance"] % 10 == 0]

    valid = df.dropna(subset=["D"])
    if valid.empty:
        print("  [plot_heatmap] no finite D, skipping")
        return

    pivot  = valid.pivot(index="vision_distance", columns="p", values="D")
    pivot  = pivot.sort_index(axis=0).sort_index(axis=1)
    grid   = pivot.values
    vr_vals = pivot.index.values.astype(float)
    p_vals  = pivot.columns.values.astype(float)

    # optionally remap Y axis from vr → κ
    if use_kappa:
        N, area = _kappa_params(root_path)
        y_vals  = (N - 1) * np.pi * vr_vals**2 / area
        ylabel  = r"Avg. degree $\kappa(v_r)$"
    else:
        y_vals  = vr_vals
        ylabel  = r"Vision radius ($v_r$)"
        ytick_fmt = lambda v: f"{int(v)}" if float(v).is_integer() else f"{v:g}"

    def _edges(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 1:
            return np.array([vals[0] - 0.5, vals[0] + 0.5])
        diffs = np.diff(vals)
        edges = np.empty(vals.size + 1)
        edges[1:-1] = vals[:-1] + diffs / 2
        edges[0]    = vals[0]  - diffs[0]  / 2
        edges[-1]   = vals[-1] + diffs[-1] / 2
        return edges

    p_edges   = _edges(p_vals);  y_edges   = _edges(y_vals)
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(p_edges, y_edges, grid, cmap=CMAP, shading="auto")

    nan_mask = np.isnan(pivot.values)
    if nan_mask.any():
        ri, ci  = np.where(nan_mask)
        lo_mask = y_vals[ri] <= np.percentile(y_vals, 40)
        hi_mask = y_vals[ri] >= np.percentile(y_vals, 70)
        if lo_mask.any():
            ax.text(p_centers[ci[lo_mask]].mean() * 0.6, y_centers[ri[lo_mask]].mean()*0.5,
                    "Silence", color="black", fontsize=10, ha="center", va="center", zorder=5)
        if hi_mask.any():
            ax.text(p_centers[ci[hi_mask]].mean() * 1.1, y_centers[ri[hi_mask]].mean()*0.7,
                    "Saturation", color="black", fontsize=10, ha="center", va="center", zorder=5)

    n_xticks = min(6, len(p_vals))
    step = max(1, len(p_vals) // n_xticks)
    keep = np.arange(len(p_vals))[::step]
    ax.set_xticks(p_centers[keep])
    ax.set_xticklabels([f"{p_vals[i]:.3f}" for i in keep], rotation=45, ha="right")
    n_yticks = min(6, len(y_vals))
    if use_kappa:
        # Pick evenly spaced κ targets and interpolate to pixel (y_center) positions,
        # so labels are round numbers rather than a subsampled non-uniform κ grid.
        # Pick a "nice" step size (nearest power-of-10 multiple of 1, 2, or 5)
        kappa_range = y_vals.max() - y_vals.min()
        raw_step    = kappa_range / (n_yticks - 1)
        magnitude   = 10 ** np.floor(np.log10(raw_step))
        nice_step   = int(magnitude * min([1, 2, 5, 10],
                          key=lambda m: abs(m * magnitude - raw_step)))
        nice_step   = max(nice_step, 1)
        first_tick  = int(np.ceil(y_vals.min() / nice_step)) * nice_step
        tick_targets = np.arange(first_tick,
                                 int(np.floor(y_vals.max())) + 1,
                                 nice_step, dtype=int)
        # Always include κ=1 (percolation transition) and the nearest nice
        # multiple to the actual maximum so the top of the axis is labelled
        max_nice = int(np.round(y_vals.max() / nice_step)) * nice_step
        tick_targets = np.unique(np.sort(np.append(tick_targets, [1, max_nice])))
        tick_targets = tick_targets[tick_targets != 0]
        tick_pos     = np.interp(tick_targets.astype(float), y_vals, y_centers)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels([str(v) for v in tick_targets])
    else:
        v_step = max(1, len(y_vals) // n_yticks)
        v_keep = np.arange(len(y_vals))[::v_step]
        ax.set_yticks(y_centers[v_keep])
        ax.set_yticklabels([ytick_fmt(y_vals[i]) for i in v_keep])

    if p_c_micro is not None:
        ax.axvline(p_c_micro, linestyle="--", color="black", linewidth=1,
                   label=r"$p_c^{\mathrm{micro}}$")
        ax.legend(loc="upper right", fontsize=12)

    ax.set_xlabel(r"$p^{\mathrm{micro}}$")
    ax.set_ylabel(ylabel)
    ax.set_title(r"$\mathcal{L}(p^{\mathrm{micro}},v_r)$  (brighter = more critical)")
    plt.colorbar(im, ax=ax, label=r"$\mathcal{L}(p^{\mathrm{micro}},v_r)$")
    fig.tight_layout()
    fname = "criticality_heatmap_kappa.png" if use_kappa else "criticality_heatmap.png"
    out = _plot_artifact_path(root_path, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


def _cell_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Bin edges so cell i is centered at ``centers[i]`` (handles uneven spacing, e.g. κ ∝ v_r²)."""
    c = np.asarray(centers, dtype=float)
    if c.size == 0:
        return np.array([0.0, 1.0])
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=float)
    e = np.empty(c.size + 1, dtype=float)
    e[0] = c[0] - (c[1] - c[0]) / 2
    e[-1] = c[-1] + (c[-1] - c[-2]) / 2
    e[1:-1] = (c[:-1] + c[1:]) / 2
    return e


def plot_sigma_eff_phase_landscape(
    root_path: str,
    *,
    surface_3d: bool = True,
    use_kappa: bool = False,
) -> None:
    """2D heatmaps and optional 3D surface: macro :math:`\\sigma_{\\mathrm{eff}}` over
    analytic :math:`\\sigma^{\\mathrm{micro}}` and either vision radius or expected degree :math:`\\kappa`.

    Reads ``macro_metrics.csv``. If ``use_kappa`` is True, the vertical axis is
    :math:`\\kappa=(N{-}1)\\pi v_r^2/A` (``n_agents``, ``map_size`` from ``metadata.yaml``).

    Output files add suffix ``_kappa`` when ``use_kappa`` is True.

    * ``sigma_eff_landscape_heatmaps.png`` / ``..._heatmaps_kappa.png`` —
      :math:`x=\\sigma^{\\mathrm{micro}}`, :math:`y=v_r` or :math:`y=\\kappa`. Uses
      ``pcolormesh`` with explicit cell edges so uneven :math:`\\kappa` spacing matches contours.
    * ``sigma_eff_landscape_surface_3d.png`` / ``..._surface_3d_kappa.png`` if
      ``surface_3d``.
    """
    csv_path = _data_artifact_path(root_path, "macro_metrics.csv")
    if not os.path.exists(csv_path):
        print("  [plot_sigma_eff_phase_landscape] macro_metrics.csv not found — "
              "run make_data_macrometrics first")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("  [plot_sigma_eff_phase_landscape] empty CSV, skipping")
        return

    n_neurons_vals = df["n_neurons"].dropna().astype(int).unique()
    if len(n_neurons_vals) != 1:
        print("  [plot_sigma_eff_phase_landscape] warning: multiple n_neurons, "
              f"using {int(n_neurons_vals[0])}")
    n_neurons = int(n_neurons_vals[0])
    if n_neurons < 2:
        print("  [plot_sigma_eff_phase_landscape] n_neurons < 2, skipping")
        return

    vds = np.sort(df["vision_distance"].unique())
    p_vals = np.sort(df["p"].unique())
    sig_micro = p_vals * (n_neurons - 1)
    n_v, n_p = len(vds), len(p_vals)

    z_eff = np.full((n_p, n_v), np.nan, dtype=float)
    for i, p in enumerate(p_vals):
        for j, vd in enumerate(vds):
            sub = df[(df["vision_distance"] == vd) & (df["p"] == p)]
            if len(sub):
                z_eff[i, j] = float(sub["sigma_eff_mean"].iloc[0])

    z_delta = z_eff - sig_micro[:, np.newaxis]

    if use_kappa:
        try:
            n_map, area = _kappa_params(root_path)
        except FileNotFoundError as e:
            print(f"  [plot_sigma_eff_phase_landscape] use_kappa=True needs N, area: {e}")
            return
        y_centers = _analytic_kappa_per_vr(vds, n_map, area)
        y_label = r"Avg. degree $\kappa = (N{-}1)\pi v_r^2/A$"
        title_top = r"$\sigma_{\mathrm{eff}}(\sigma^{\mathrm{micro}}, \kappa)$"
        out_suffix = "_kappa"
    else:
        y_centers = vds.astype(float)
        y_label = r"Vision radius $v_r$"
        title_top = r"$\sigma_{\mathrm{eff}}(\sigma^{\mathrm{micro}}, v_r)$"
        out_suffix = ""

    # pcolormesh: explicit x/y edges so κ(v_r) non-uniform spacing is not squashed (imshow assumes
    # uniform pixel height in data coordinates). Z[j,i] = σ_eff at (σ^micro[i], y_centers[j]).
    xe_sig = _cell_edges_from_centers(sig_micro)
    ye_y = _cell_edges_from_centers(y_centers)

    # Contour on cell-center grid (same as pcolormesh shading='flat' value coords)
    Xcnt, Ycnt = np.meshgrid(sig_micro.astype(float), y_centers.astype(float))

    # --- heatmaps ---
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.2, 7.4), sharex=True)
    z0 = np.ma.masked_invalid(z_eff.T)
    m0 = ax0.pcolormesh(
        xe_sig,
        ye_y,
        z0,
        shading="flat",
        cmap="magma",
        zorder=1,
    )
    fig.colorbar(m0, ax=ax0, label=r"$\sigma_{\mathrm{eff}}$ (macro)")
    ax0.contour(
        Xcnt,
        Ycnt,
        np.asarray(z_eff.T, dtype=float),
        levels=[1.0],
        colors="black",
        linewidths=1.25,
        linestyles="-",
        zorder=5,
    )
    ax0.axvline(1.0, color="black", ls=":", lw=1.0)
    ax0.plot(
        [],
        [],
        color="black",
        ls="-",
        lw=1.25,
        label=r"Contour: $\sigma_{\mathrm{eff}} = 1$",
    )
    ax0.plot(
        [],
        [],
        color="black",
        ls=":",
        lw=1.0,
        label=r"Reference: $\sigma^{\mathrm{micro}} = 1$",
    )
    ax0.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax0.set_ylabel(y_label)
    ax0.set_title(
        title_top + "\n"
        r"Colour $=$ surface height; solid black $=$ contours (legend)",
        fontsize=9,
    )

    d_min = float(np.nanmin(z_delta))
    d_max = float(np.nanmax(z_delta))
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_min >= d_max:
        d_min, d_max = -1.0, 1.0
    norm_d = Normalize(vmin=d_min, vmax=d_max)
    z1 = np.ma.masked_invalid(z_delta.T)
    m1 = ax1.pcolormesh(
        xe_sig,
        ye_y,
        z1,
        shading="flat",
        cmap="cividis",
        norm=norm_d,
        zorder=1,
    )
    fig.colorbar(m1, ax=ax1, label=r"$\sigma_{\mathrm{eff}} - \sigma^{\mathrm{micro}}$")
    ax1.contour(
        Xcnt,
        Ycnt,
        np.asarray(z_delta.T, dtype=float),
        levels=[0.0],
        colors="black",
        linewidths=1.1,
        linestyles="-",
        zorder=5,
    )
    ax1.axvline(1.0, color="black", ls=":", lw=1.0)
    ax1.plot(
        [],
        [],
        color="black",
        ls="-",
        lw=1.1,
        label=r"Contour: $\sigma_{\mathrm{eff}} - \sigma^{\mathrm{micro}} = 0$",
    )
    ax1.plot(
        [],
        [],
        color="black",
        ls=":",
        lw=1.0,
        label=r"Reference: $\sigma^{\mathrm{micro}} = 1$",
    )
    ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(r"$\sigma^{\mathrm{micro}} = p^{\mathrm{micro}} (N_{\mathrm{neurons}}-1)$")
    ax1.set_title(
        r"$\sigma_{\mathrm{eff}} - \sigma^{\mathrm{micro}}$" + "\n"
        r"Colour $=$ surface height; solid black $=$ contours (legend)",
        fontsize=9,
    )

    fig.tight_layout()
    out_hm = _plot_artifact_path(root_path, f"sigma_eff_landscape_heatmaps{out_suffix}.png")
    fig.savefig(out_hm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_hm}")

    if not surface_3d:
        return

    # --- 3D surface (same orientation as heatmaps: X = σ^micro, Y = v_r or κ) ---
    X, Y = np.meshgrid(sig_micro.astype(float), y_centers.astype(float))
    Z = np.ma.masked_invalid(z_eff.T)
    fig3 = plt.figure(figsize=(6.5, 5.2))
    ax3 = fig3.add_subplot(111, projection="3d")
    surf = ax3.plot_surface(
        X, Y, Z, cmap="magma", linewidth=0, antialiased=True, alpha=0.95, rstride=1, cstride=1
    )
    fig3.colorbar(surf, ax=ax3, shrink=0.55, aspect=12, label=r"$\sigma_{\mathrm{eff}}$")
    ax3.set_xlabel(r"$\sigma^{\mathrm{micro}}$")
    ax3.set_ylabel(r"$\kappa$" if use_kappa else r"$v_r$")
    ax3.set_zlabel(r"$\sigma_{\mathrm{eff}}$")
    ax3.view_init(elev=28, azim=-58)
    fig3.tight_layout()
    out_3d = _plot_artifact_path(root_path, f"sigma_eff_landscape_surface_3d{out_suffix}.png")
    fig3.savefig(out_3d, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved -> {out_3d}")


# ---------------------------------------------------------------------------
# Multi-vision comparison plot
# ---------------------------------------------------------------------------

def _load_vd_records(
    root_path: str,
    df_fits: "pd.DataFrame",
    vision_distances: List[int],
) -> list:
    """Load npz fit data for a list of vision distances (internal helper).

    Returns a list of dicts with keys: vd, best_p, D, d (NpzFile).
    """
    records = []
    for vd in sorted(vision_distances):
        sub = df_fits[df_fits["vision_distance"] == vd].dropna(subset=["D"])
        if sub.empty:
            print(f"  vd={vd}: no valid D in summary_fits.csv, skipping")
            continue
        best_row = sub.loc[sub["D"].idxmin()]
        best_p   = float(best_row["p"])
        D_val    = float(best_row["D"])

        # *_{vd}vision_distance avoids matching e.g. 135 when asking for 35
        exp_dirs = glob.glob(os.path.join(root_path, f"*_{vd}vision_distance"))
        if not exp_dirs:
            print(f"  vd={vd}: experiment directory not found, skipping")
            continue
        fits_dir = os.path.join(exp_dirs[0], "data_fits")

        npz_path = os.path.join(fits_dir, f"fit_p_{best_p:.6f}.npz")
        if not os.path.exists(npz_path):
            candidates = sorted(glob.glob(os.path.join(fits_dir, "fit_p_*.npz")))
            if not candidates:
                print(f"  vd={vd}: no fit files found, skipping")
                continue
            npz_path = min(
                candidates,
                key=lambda fp: abs(float(os.path.basename(fp)[6:-4]) - best_p),
            )

        # load GoF p-values from gof_pvalues.csv if present
        gof_p_s = np.nan;  gof_p_d = np.nan
        gof_csv = os.path.join(fits_dir, "gof_pvalues.csv")
        if os.path.exists(gof_csv):
            gof_df  = pd.read_csv(gof_csv)
            gof_row = gof_df[np.isclose(gof_df["p"], best_p, atol=5e-5, rtol=1e-3)]
            if not gof_row.empty:
                gof_p_s = float(gof_row["gof_p_size"].iloc[0])
                gof_p_d = float(gof_row["gof_p_duration"].iloc[0])

        records.append({"vd": vd, "best_p": best_p, "D": D_val,
                        "d": np.load(npz_path, allow_pickle=False),
                        "gof_p_s": gof_p_s, "gof_p_d": gof_p_d})
    return records


def _smooth_power_law_curve(alpha: float, xmin: int, n_fit: int, n_complete: int,
                             x_max: float, n_pts: int = 400) -> tuple:
    """Return (x, y) for a smooth theoretical power-law PMF curve.

    Computes  P(x) = tail_mass · x^{-alpha} / ζ(alpha, xmin)
    on a dense log-spaced grid from xmin to x_max,
    where tail_mass = n_fit / n_complete rescales to match the empirical PDF.
    """
    if xmin < 1 or alpha <= 1.0 or n_complete <= 0:
        return None, None
    tail_mass = n_fit / n_complete
    norm      = _hurwitz_zeta(alpha, float(xmin))
    if not np.isfinite(norm) or norm <= 0:
        return None, None
    x = np.geomspace(xmin, x_max, n_pts)
    y = tail_mass * x ** (-alpha) / norm
    return x, y


def _plot_vd_series(ax_sz, ax_dur, records, cmap_name, all_x_sz, all_y_sz,
                    all_x_dur, all_y_dur, linestyle="-", lw=1.6, alpha_line=1.0,
                    fit_lw=1.5, fit_alpha=0.9, scatter=False,
                    fit_x_sz=None, fit_y_sz=None,
                    fit_x_dur=None, fit_y_dur=None):
    """Draw size and duration series for a list of records onto ax_sz / ax_dur.

    Colors are evenly distributed across the colormap (one step per record),
    regardless of the actual numeric values of the vision distances.

    Returns a list of row-dicts for the summary table:
        {color, vd, best_p, n_complete, alpha_s, ks_s, alpha_d, ks_d}
    """
    if not records:
        return []
    n      = len(records)
    cmap   = plt.colormaps[cmap_name]
    # evenly-spaced colours capped at 0.80 to avoid the near-white end of the colormap
    colors = [cmap(0.80 * i / max(n - 1, 1)) for i in range(n)]

    table_rows = []

    for i, rec in enumerate(records):
        vd, best_p, d = rec["vd"], rec["best_p"], rec["d"]
        color      = colors[i]
        n_complete = int(d["n_complete"]) if "n_complete" in d.files else None

        alpha_s = np.nan;  ks_s = np.nan
        alpha_d = np.nan;  ks_d = np.nan

        # --- size ---
        has_size = (
            "t_vals_size" in d.files and d["t_vals_size"].size > 0
            and "pdf_size"  in d.files and d["pdf_size"].size  > 0
            and "alpha_size" in d.files and np.isfinite(float(d["alpha_size"]))
        )
        if has_size:
            t_s, p_s = d["t_vals_size"], d["pdf_size"]
            idx = np.argsort(t_s);  t_s, p_s = t_s[idx], p_s[idx]
            alpha_s  = float(d["alpha_size"])
            ks_s     = float(d["ks_size"]) if "ks_size" in d.files and np.isfinite(float(d["ks_size"])) else np.nan
            xmin_s   = int(d["xmin_size"]) if "xmin_size" in d.files and int(d["xmin_size"]) >= 1 else None
            n_fit_s  = int(d["n_fit_size"]) if "n_fit_size" in d.files else None

            if scatter:
                ax_sz.scatter(t_s, p_s, s=8, color=color, alpha=0.45, zorder=2)
                if xmin_s and n_fit_s and n_complete:
                    xs, ys = _smooth_power_law_curve(alpha_s, xmin_s, n_fit_s, n_complete,
                                                     float(t_s[-1]) * 1.5)
                    if xs is not None:
                        ax_sz.plot(xs, ys, linestyle=linestyle, color=color,
                                   linewidth=lw, alpha=alpha_line, zorder=3)
            else:
                ax_sz.plot(t_s, p_s, color=color, linewidth=lw, linestyle=linestyle,
                           alpha=alpha_line)
                if xmin_s and n_fit_s and n_complete:
                    xs, ys = _smooth_power_law_curve(alpha_s, xmin_s, n_fit_s, n_complete,
                                                     float(t_s[-1]) * 1.5)
                    if xs is not None:
                        ax_sz.plot(xs, ys, linestyle="-", color=color,
                                   linewidth=fit_lw, alpha=fit_alpha, zorder=3)
                        if fit_x_sz is not None:
                            fit_x_sz.extend(xs.tolist()); fit_y_sz.extend(ys.tolist())

            vm = np.isfinite(p_s) & (p_s > 0)
            all_x_sz.extend(t_s[vm].tolist());  all_y_sz.extend(p_s[vm].tolist())

        # --- duration ---
        has_dur = (
            "t_vals_duration" in d.files and d["t_vals_duration"].size > 0
            and "pdf_duration"  in d.files and d["pdf_duration"].size  > 0
            and "alpha_duration" in d.files and np.isfinite(float(d["alpha_duration"]))
        )
        if has_dur:
            t_d, p_d = d["t_vals_duration"], d["pdf_duration"]
            idx = np.argsort(t_d);  t_d, p_d = t_d[idx], p_d[idx]
            alpha_d  = float(d["alpha_duration"])
            ks_d     = float(d["ks_duration"]) if "ks_duration" in d.files and np.isfinite(float(d["ks_duration"])) else np.nan
            xmin_d   = int(d["Tmin_duration"]) if "Tmin_duration" in d.files and int(d["Tmin_duration"]) >= 1 else None
            n_fit_d  = int(d["n_fit_duration"]) if "n_fit_duration" in d.files else None

            if scatter:
                ax_dur.scatter(t_d, p_d, s=8, color=color, alpha=0.45, zorder=2)
                if xmin_d and n_fit_d and n_complete:
                    xd, yd = _smooth_power_law_curve(alpha_d, xmin_d, n_fit_d, n_complete,
                                                     float(t_d[-1]) * 1.5)
                    if xd is not None:
                        ax_dur.plot(xd, yd, linestyle=linestyle, color=color,
                                    linewidth=lw, alpha=alpha_line, zorder=3)
            else:
                ax_dur.plot(t_d, p_d, color=color, linewidth=lw, linestyle=linestyle,
                            alpha=alpha_line)
                if xmin_d and n_fit_d and n_complete:
                    xd, yd = _smooth_power_law_curve(alpha_d, xmin_d, n_fit_d, n_complete,
                                                     float(t_d[-1]) * 1.5)
                    if xd is not None:
                        ax_dur.plot(xd, yd, linestyle="-", color=color,
                                    linewidth=fit_lw, alpha=fit_alpha, zorder=3)
                        if fit_x_dur is not None:
                            fit_x_dur.extend(xd.tolist()); fit_y_dur.extend(yd.tolist())

            vm = np.isfinite(p_d) & (p_d > 0)
            all_x_dur.extend(t_d[vm].tolist());  all_y_dur.extend(p_d[vm].tolist())

        table_rows.append(dict(
            color=color, vd=vd, best_p=best_p, n=n_complete,
            alpha_s=alpha_s, ks_s=ks_s, alpha_d=alpha_d, ks_d=ks_d,
            gof_p_s=rec.get("gof_p_s", np.nan),
            gof_p_d=rec.get("gof_p_d", np.nan),
        ))
        d.close()

    return table_rows


def _table_legend(ax, all_rows: list, alpha_col: str, ks_col: str, gof_col: str,
                  alpha_label: str, ks_label: str, ref_handle=None,
                  loc: str = "upper right", p_c_micro: float | None = None) -> None:
    """Replace ax.legend() with a monospace-aligned table inside the legend box.

    Columns: vr | p* | n | <alpha_label> | <ks_label> | p-val
    p-val is the Clauset bootstrap GoF p-value; shown as '—' if not yet computed.
    p_c^micro is appended to the reference slope entry at the bottom.
    """
    from matplotlib.lines import Line2D

    # ── column widths (chars) ──────────────────────────────────────────────────
    W_VR = 5;  W_P = 7;  W_N = 6;  W_A = 5;  W_K = 6;  W_G = 5
    sep = "  "

    def _row(vr, p, n, a, k, g):
        return (f"{str(vr):>{W_VR}}{sep}"
                f"{str(p):>{W_P}}{sep}"
                f"{str(n):>{W_N}}{sep}"
                f"{str(a):>{W_A}}{sep}"
                f"{str(k):>{W_K}}{sep}"
                f"{str(g):>{W_G}}")

    header_txt    = _row("vr", "p*", "n", alpha_label, ks_label, "p-val")
    header_handle = Line2D([0], [0], color="none")

    handles = [header_handle]
    labels  = [header_txt]

    for row in all_rows:
        n_str = f"{row['n']:,}"  if row["n"] is not None else "—"
        a_val = row[alpha_col];  k_val = row[ks_col];  g_val = row.get(gof_col, np.nan)
        a_str = f"{a_val:.2f}"  if np.isfinite(a_val) else "—"
        k_str = f"{k_val:.3f}"  if np.isfinite(k_val) else "—"
        g_str = f"{g_val:.2f}"  if np.isfinite(g_val) else "—"
        labels.append(_row(row["vd"], f"{row['best_p']:.4f}", n_str, a_str, k_str, g_str))
        handles.append(Line2D([0], [0], color=row["color"], linewidth=2.2, linestyle="-"))

    if ref_handle is not None:
        pc_suffix = (rf"   $p_c^{{\mathrm{{micro}}}}={p_c_micro:.4f}$"
                     if p_c_micro is not None else "")
        handles.append(ref_handle[0])
        labels.append(ref_handle[1] + pc_suffix)

    ax.legend(
        handles, labels,
        prop={"family": "monospace", "size": 7},
        loc=loc,
        frameon=True, framealpha=0.9, edgecolor="0.65",
        handlelength=1.6, handletextpad=0.6,
        borderpad=0.6,
    )


def plot_critical_distributions_by_vision(
    root_path: str,
    vision_distances_good: List[int],
    vision_distances_bad: List[int] | None = None,
    p_c_micro: float | None = None,
    out_name: str = "critical_distributions_by_vision.png",
    cmap_good: str = "plasma",
    cmap_bad:  str = "Reds",
    scatter: bool = False,
    plot_table: bool = True,
    out_subdir: str = "",
) -> None:
    """Plot cascade size and duration distributions at the most critical p for each vision distance.

    Two groups can be supplied:
      • vision_distances_good  — most critical VDs (solid lines, plasma colormap by default).
      • vision_distances_bad   — least critical VDs (dotted lines, Reds colormap by default).

    Both share the same axes, making the contrast immediately visible.
    A theoretical reference slope (α_S=1.5, α_T=2.0) is anchored at the geometric
    centre of all plotted data.

    Args:
        root_path:             Directory containing summary_fits.csv and experiment folders.
        vision_distances_good: VDs to show as "good" (most critical).
        vision_distances_bad:  VDs to show as "bad" (least critical). Optional.
        p_c_micro:             Theoretical micro critical p (shown in legend if provided).
        out_name:              Output filename (saved inside out_subdir).
        cmap_good:             Colormap for the good VDs.
        cmap_bad:              Colormap for the bad VDs.
        scatter:               If False (default), empirical data as lines + dashed fit.
                               If True, empirical data as scatter dots + solid fit line.
        plot_table:            If True (default), show full table legend (vr, p*, n, α, KS, p-val).
                               If False, show a minimal legend with only the vr label and α reference.
        out_subdir:            Ignored (all outputs are saved in media/<root_path basename>).
    """
    
    
    ALPHA_FITTED_LINE = 1
    ALPHA_REFERENCE_LINE = 1
    ALPHA_CURVE = 0.7
    
    
    csv_path = _data_artifact_path(root_path, "summary_fits.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"summary_fits.csv not found in {root_path}; run make_data_heatmap first.")

    df = pd.read_csv(csv_path)

    all_requested = list(vision_distances_good) + list(vision_distances_bad or [])
    df_sub = df[df["vision_distance"].isin(all_requested)]
    if df_sub.empty:
        print(f"  [plot_critical_distributions_by_vision] none of {all_requested} found in summary_fits.csv")
        return

    W, GR = 4.2, 1.618
    fig, (ax_sz, ax_dur) = plt.subplots(2, 1, figsize=(W, W * 1.7 / GR))

    all_x_sz,  all_y_sz  = [], []
    all_x_dur, all_y_dur = [], []
    fit_x_sz,  fit_y_sz  = [], []
    fit_x_dur, fit_y_dur = [], []

    records_good = _load_vd_records(root_path, df_sub, vision_distances_good)
    rows_good = _plot_vd_series(ax_sz, ax_dur, records_good, cmap_good,
                                all_x_sz, all_y_sz, all_x_dur, all_y_dur,
                                linestyle="-", lw=1.2, scatter=scatter,
                                alpha_line=ALPHA_CURVE, fit_alpha=ALPHA_FITTED_LINE,
                                fit_x_sz=fit_x_sz, fit_y_sz=fit_y_sz,
                                fit_x_dur=fit_x_dur, fit_y_dur=fit_y_dur)

    rows_bad = []
    if vision_distances_bad:
        records_bad = _load_vd_records(root_path, df_sub, vision_distances_bad)
        rows_bad = _plot_vd_series(ax_sz, ax_dur, records_bad, cmap_bad,
                                   all_x_sz, all_y_sz, all_x_dur, all_y_dur,
                                   linestyle=(0, (3, 1)), lw=1.0,
                                   alpha_line=ALPHA_CURVE, fit_lw=0.8,
                                   fit_alpha=ALPHA_FITTED_LINE, scatter=scatter,
                                   fit_x_sz=fit_x_sz, fit_y_sz=fit_y_sz,
                                   fit_x_dur=fit_x_dur, fit_y_dur=fit_y_dur)

    # reference mean-field slopes — anchored on fit-line points, spanning full x range
    def _draw_ref_slope(ax, fit_x, fit_y, all_x, exponent):
        xs = fit_x if fit_x else all_x
        ys = fit_y if fit_y else []
        if not xs or not ys:
            return None
        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)
        x_gm  = np.exp(np.mean(np.log(x_arr[x_arr > 0])))
        y_gm  = np.exp(np.mean(np.log(y_arr[y_arr > 0])))
        # span the full plotted x range
        all_x_arr = np.array(all_x, dtype=float)
        x_min = all_x_arr[all_x_arr > 0].min()
        x_max = all_x_arr[all_x_arr > 0].max()
        x_ref = np.geomspace(x_min, x_max, 300)
        y_ref = y_gm * (x_ref / x_gm) ** (-exponent)
        line, = ax.plot(x_ref, y_ref, color="0.30", linestyle=(0, (6, 2)),
                        linewidth=1.4, alpha=ALPHA_REFERENCE_LINE, zorder=5)
        return line

    ref_sz  = _draw_ref_slope(ax_sz,  fit_x_sz,  fit_y_sz,  all_x_sz,  1.5)
    ref_dur = _draw_ref_slope(ax_dur, fit_x_dur, fit_y_dur, all_x_dur, 2.0)

    all_rows = rows_good + rows_bad
    from matplotlib.lines import Line2D as _L2D

    if plot_table:
        _table_legend(ax_sz,  all_rows, "alpha_s", "ks_s", "gof_p_s", "\u03b1_S", "KS_S",
                      ref_handle=(_L2D([0],[0], color="0.35", lw=2, ls=(0,(6,2))),
                                  r"$\alpha_S=1.5$  ref") if ref_sz  else None,
                      loc="lower left", p_c_micro=p_c_micro)
        _table_legend(ax_dur, all_rows, "alpha_d", "ks_d", "gof_p_d", "\u03b1_T", "KS_T",
                      ref_handle=(_L2D([0],[0], color="0.35", lw=2, ls=(0,(6,2))),
                                  r"$\alpha_T=2.0$  ref") if ref_dur else None,
                      loc="lower left", p_c_micro=p_c_micro)
    else:
        # minimal legend: one coloured entry per vr + the α reference line
        for ax, ref_line, ref_label in [
            (ax_sz,  ref_sz,  r"$\alpha_S=1.5$"),
            (ax_dur, ref_dur, r"$\alpha_T=2.0$"),
        ]:
            n   = len(all_rows)
            cmap = plt.colormaps[cmap_good]
            handles = [_L2D([0],[0], color=cmap(0.80 * i / max(n-1,1)), lw=2)
                       for i in range(n)]
            labels  = [f"$v_r={r['vd']}$" for r in all_rows]
            if ref_line is not None:
                handles.append(_L2D([0],[0], color="0.35", lw=2, ls=(0,(6,2))))
                labels.append(ref_label)
            ax.legend(handles, labels, fontsize=8, loc="lower left",
                      frameon=True, framealpha=0.9, edgecolor="0.65")

    for ax, xlabel, ylabel in [
        (ax_sz,  r"Avalanche size $S$",     r"$P(S)$"),
        (ax_dur, r"Avalanche duration $T$", r"$P(T)$"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    fig.tight_layout()

    _ = out_subdir
    out = _plot_artifact_path(root_path, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")

    # save table data as CSV alongside the plot
    if all_rows:
        tbl_rows = []
        for r in all_rows:
            tbl_rows.append({
                "vd":       r["vd"],
                "p_star":   r["best_p"],
                "n":        r["n"],
                "alpha_S":  r["alpha_s"],
                "KS_S":     r["ks_s"],
                "gof_p_S":  r["gof_p_s"],
                "alpha_T":  r["alpha_d"],
                "KS_T":     r["ks_d"],
                "gof_p_T":  r["gof_p_d"],
            })
        tbl_path = _data_artifact_path(root_path, out_name.replace(".png", "_table.csv"))
        pd.DataFrame(tbl_rows).to_csv(tbl_path, index=False, float_format="%.4f")
        print(f"Saved -> {tbl_path}")


# ---------------------------------------------------------------------------
# Activity-band plot
# ---------------------------------------------------------------------------

def plot_integrated_criticality(root_path: str, p_c_micro: float | None = None,
                                use_kappa: bool = False) -> None:
    """For each p, integrate 1/L across all active vision radii (or κ).

    Since L is a quality score where lower = better criticality, integrating
    1/L gives an "integrated quality": higher values mean the system is robustly
    critical across a wide range of vision radii / neighbour counts.
    NaN rows (silence / saturation) are excluded.

    When use_kappa=False (default): plain discrete sum Σ_{vr} 1/L.
    When use_kappa=True:            trapezoidal integral ∫ (1/L) dκ, where
        κ(vr) = (N − 1) · π · vr² / area  is the expected neighbour count.

    Saved as:
        integrated_criticality.png       (use_kappa=False)
        integrated_criticality_kappa.png (use_kappa=True)
    """
    csv_path = _data_artifact_path(root_path, "summary_fits.csv")
    df = pd.read_csv(csv_path)

    if use_kappa:
        N, area = _kappa_params(root_path)

    p_vals       = sorted(df["p"].unique())
    integrated   = []
    truncated    = []   # True if the upper active vr hits the global measurement ceiling

    vr_global_max = df["vision_distance"].max()

    for p in p_vals:
        sub    = df[df["p"] == p].sort_values("vision_distance")
        active = sub.dropna(subset=["D"])
        valid  = active[active["D"] > 0].sort_values("vision_distance")

        if valid.empty:
            integrated.append(0.0)
        elif use_kappa:
            vr_vals  = valid["vision_distance"].values.astype(float)
            kappa    = (N - 1) * np.pi * vr_vals**2 / area
            inv_D    = 1.0 / valid["D"].values
            # trapezoidal integration over κ
            integrated.append(float(np.trapezoid(inv_D, x=kappa)))
        else:
            integrated.append(float((1.0 / valid["D"]).sum()))

        # Truncated if there are active rows AND the highest active vr equals the
        # global ceiling (meaning the band was cut off, not ended by saturation)
        if not active.empty and active["vision_distance"].max() >= vr_global_max:
            truncated.append(True)
        else:
            truncated.append(False)

    p_arr      = np.array(p_vals,    dtype=float)
    sum_arr    = np.array(integrated, dtype=float)
    trunc_mask = np.array(truncated, dtype=bool)

    GOLDEN_RATIO = 1.618
    HEIGHT = 2.5
    WIDTH = HEIGHT * GOLDEN_RATIO
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    # Shade the truncated p region in the background.
    # Extend the right edge to the first non-truncated p so the shading aligns
    # with the dashed segment that bridges the truncated/complete boundary.
    if trunc_mask.any():
        p_trunc = p_arr[trunc_mask]
        trunc_indices = np.where(trunc_mask)[0]
        last_trunc_idx = trunc_indices[-1]
        right_edge = (p_arr[last_trunc_idx + 1]
                      if last_trunc_idx + 1 < len(p_arr)
                      else p_arr[last_trunc_idx])
        ax.axvspan(p_trunc.min(), right_edge,
                   color="grey", alpha=0.2, zorder=0,
                   label="$v_r$ data limit")

    # Solid line for complete data, dashed for truncated
    # Split into contiguous solid / dashed segments
    for i in range(len(p_arr) - 1):
        seg_p = p_arr[i:i+2]
        seg_v = sum_arr[i:i+2]
        if trunc_mask[i] or trunc_mask[i+1]:
            ax.plot(seg_p, seg_v, color="black", lw=2,
                    linestyle="--", zorder=3)
        else:
            ax.plot(seg_p, seg_v, color="black", lw=2,
                    linestyle="-", zorder=3)

    if p_c_micro is not None:
        ax.axvline(p_c_micro, linestyle="--", color="black", linewidth=1,
                   label=r"$p_c^{\mathrm{micro}}$")

    ax.legend(fontsize=8)
    ax.set_xlabel(r"$p^{\mathrm{micro}}$")
    if use_kappa:
        ax.set_ylabel(r"$\sum_{\kappa} 1/\mathcal{L}$")
    else:
        ax.set_ylabel(r"$\sum_{v_r} 1/\mathcal{L}$")

    fig.tight_layout()
    fname = "integrated_criticality_kappa.png" if use_kappa else "integrated_criticality.png"
    out = _plot_artifact_path(root_path, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root_path = "experiments_data/vision_radii"
    p_c = get_p_c_micro(root_path)

    #make_data_heatmap(root_path)
    #make_data_macrometrics(root_path)

    plot_sigma_eff_phase_landscape(root_path)
    plot_sigma_eff_phase_landscape(root_path, use_kappa=True)

    
    plot_critical_distributions_by_vision(
        root_path,
        vision_distances_good=[35, 50, 100],
        vision_distances_bad=[],
        p_c_micro=p_c,
        plot_table=False,
        cmap_good="inferno",
    )

    plot_heatmap(root_path, p_c_micro=p_c, use_kappa=False)
    plot_heatmap(root_path, p_c_micro=p_c, use_kappa=True)
    plot_integrated_criticality(root_path, p_c_micro=p_c, use_kappa=False)
    plot_integrated_criticality(root_path, p_c_micro=p_c, use_kappa=True)
    