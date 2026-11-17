from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_TOPOLOGIES = ["USNET", "GERMAN", "TI"]
LOAD_TICKS = [5, 10, 15, 20, 25]

DEFAULT_METRICS: List[Tuple[str, str, str, str]] = [
    ("RBP_mean", "RBP_ci95", "Request Blocking Probability", "RBP"),
    ("BBP_mean", "BBP_ci95", "Bandwidth Blocking Probability", "BBP"),
    ("throughput_mean", "throughput_ci95", "Normalized Throughput", "Throughput"),
    ("accepted_load_mean", "accepted_load_ci95", "Accepted Load", "Accepted Load"),
    ("FSU_mean", "FSU_ci95", "Frequency Slot Utilization", "FSU"),
    ("avg_LFM_mean", "avg_LFM_ci95", "Average Fragment Density Metric", "Average FDM"),
    ("avg_EBFM_mean", "avg_EBFM_ci95", "Average Entropy-Based Fragmentation Metric", "avg_EBFM"),
    ("avg_K_mean", "avg_K_ci95", "Average Coding Paths Used", "avg_K"),
    ("avg_r_over_g_mean", "avg_r_over_g_ci95", "Average Redundancy Ratio", "avg_r/g"),
    (
        "actual_aggregate_accepted_load_erlangs_mean",
        "actual_aggregate_accepted_load_erlangs_ci95",
        "Aggregate Accepted Load",
        "Aggregate accepted load (Erlangs)",
    ),
    (
        "mean_active_carried_gbps_mean",
        "mean_active_carried_gbps_ci95",
        "Mean Active Carried Traffic",
        "Mean active carried traffic (Gbps)",
    ),
    (
        "mean_allocated_slot_hops_per_accepted_request_mean",
        "mean_allocated_slot_hops_per_accepted_request_ci95",
        "Mean Slot-Hop Cost per Accepted Request",
        "Slot-hops per accepted request",
    ),
    (
        "slot_hops_per_accepted_gbps_mean",
        "slot_hops_per_accepted_gbps_ci95",
        "Slot-Hop Cost per Accepted Gbps",
        "Slot-hops per accepted Gbps",
    ),
    (
        "carried_gbps_per_occupied_slot_hop_mean",
        "carried_gbps_per_occupied_slot_hop_ci95",
        "Carried-Traffic Slot-Hop Efficiency",
        "Carried Gbps per occupied slot-hop",
    ),
    (
        "occupied_slot_hops_per_carried_gbps_mean",
        "occupied_slot_hops_per_carried_gbps_ci95",
        "Occupied Slot-Hop Cost per Carried Gbps",
        "Occupied slot-hops per carried Gbps",
    ),
    (
        "mean_decision_time_us_mean",
        "mean_decision_time_us_ci95",
        "Mean Decision Time",
        "Mean decision time (us)",
    ),
    (
        "p95_decision_time_us_mean",
        "p95_decision_time_us_ci95",
        "95th-Percentile Decision Time",
        "95th-percentile decision time (us)",
    ),
]

IMPROVEMENT_METRICS: List[Tuple[str, str, str]] = [
    ("RBP_mean", "RBP", "lower"),
    ("BBP_mean", "BBP", "lower"),
    ("throughput_mean", "Throughput", "higher"),
    ("FSU_mean", "FSU", "higher"),
    ("avg_LFM_mean", "FDM", "lower"),
    ("avg_EBFM_mean", "EBFM", "lower"),
]

TABLE_METRICS: List[Tuple[str, str, str]] = [
    ("RBP_mean", "RBP", "lower"),
    ("BBP_mean", "BBP", "lower"),
    ("FSU_mean", "FSU", "higher"),
    ("avg_LFM_mean", "FDM", "lower"),
    ("throughput_mean", "Throughput", "higher"),
]

SCHEME_DISPLAY_NAMES: Dict[str, str] = {
    "k1_rsa": "K1-RSA",
    "k3_rsa": "K3-RSA",
    "fsa_rsa_ssfm": "FSA-RSA(SSFM)",
    "fsa_k3_rsa": "FSA-K3-RSA",
    "rb_scsp_arsa": "RB-SCSP-ARSA",
    "mp_ss_rsa": "MP-SS-RSA",
    "adaptive_mp_ss_rsa": "A-MP-SS-RSA",
    "adaptive_mp_ss_rsa_repetition": "A-MP-SS-RSA-Repetition",
    "aca_rsa": "ACA-RSA",
    # Legacy names remain readable for older result files.
    "ara_rsa": "ACA-RSA",
    "ara_k3_rsa": "ACA-K3-RSA",
    "p_aca_rsa": "P-ACA-RSA",
}


def scheme_label(scheme: str) -> str:
    return SCHEME_DISPLAY_NAMES.get(str(scheme), str(scheme))


SCHEME_STYLES: Dict[str, Dict[str, object]] = {
    "aca_rsa": {"marker": "o", "linestyle": "-"},
    "ara_rsa": {"marker": "o", "linestyle": "-"},
    "fsa_rsa_ssfm": {"marker": "s", "linestyle": "--"},
    "k1_rsa": {"marker": "^", "linestyle": ":"},
    "k3_rsa": {"marker": "D", "linestyle": "-."},
    "mp_ss_rsa": {"marker": "P", "linestyle": (0, (5, 2))},
    "adaptive_mp_ss_rsa": {"marker": "h", "linestyle": (0, (5, 1, 1, 1))},
    "adaptive_mp_ss_rsa_repetition": {"marker": "H", "linestyle": (0, (3, 1))},
    "p_aca_rsa": {"marker": "X", "linestyle": (0, (1, 2))},
    "rb_scsp_arsa": {"marker": "*", "linestyle": (0, (3, 1, 1, 1))},
    "ara_k3_rsa": {"marker": "v", "linestyle": (0, (7, 2, 1, 2))},
    "fsa_k3_rsa": {"marker": "<", "linestyle": (0, (2, 2, 6, 2))},
}

DEFAULT_SCHEME_STYLES: List[Dict[str, object]] = [
    {"marker": "o", "linestyle": "-"},
    {"marker": "s", "linestyle": "--"},
    {"marker": "^", "linestyle": ":"},
    {"marker": "D", "linestyle": "-."},
    {"marker": "P", "linestyle": (0, (5, 2))},
    {"marker": "X", "linestyle": (0, (1, 2))},
    {"marker": "*", "linestyle": (0, (3, 1, 1, 1))},
]


def scheme_style(scheme: str) -> Dict[str, object]:
    if scheme in SCHEME_STYLES:
        return SCHEME_STYLES[scheme]
    keys = sorted(SCHEME_DISPLAY_NAMES)
    idx = keys.index(scheme) if scheme in keys else abs(hash(str(scheme)))
    return DEFAULT_SCHEME_STYLES[idx % len(DEFAULT_SCHEME_STYLES)]


PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

TOPOLOGY_DISPLAY_NAMES: Dict[str, str] = {
    "USNET": "USNET",
    "GERMAN": "German",
    "TI": "TI",
}


def format_topology_caption(index: int, topology: str) -> str:
    prefix = PANEL_LABELS[index] if index < len(PANEL_LABELS) else f"({chr(97 + index)})"
    return f"{prefix} {topology}"


def apply_publication_axis_style(ax: plt.Axes, show_ylabel: bool) -> None:
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    if not show_ylabel:
        ax.set_ylabel("")


def add_subplot_legend(ax: plt.Axes, mean_col: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    loc = "lower left" if mean_col == "throughput_mean" else "upper left"
    ax.legend(
        handles,
        labels,
        loc=loc,
        ncol=2,
        fontsize=11,
        frameon=True,
        framealpha=0.88,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=0.7,
    )

LOAD_REGIONS = {
    "Low": {5, 10},
    "Mid": {15},
    "High": {20, 25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot simulator summary CSVs by topology. This script consumes the "
            "precomputed *_mean and *_ci95 columns; it does not recompute 95% CI."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path to simulation_results_summary CSV. If omitted, the newest summary CSV is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save plots. Defaults to "
            "<summary_csv_parent>/plots_by_topology_v31."
        ),
    )
    parser.add_argument(
        "--schemes",
        type=str,
        nargs="*",
        default=None,
        help="Optional scheme names to plot. If omitted, plot all schemes in the CSV.",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        nargs="*",
        default=DEFAULT_TOPOLOGIES,
        help="Topologies to plot. Default: USNET GERMAN TI.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="line",
        choices=["line", "bar"],
        help="Plot style. Use line for load sweeps and bar for compact comparisons.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution. Default: 200 to reduce memory pressure.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "svg", "pdf"],
        help="Figure format. Use svg or pdf for low-memory vector output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display plots interactively.",
    )
    parser.add_argument(
        "--no-improvement-plots",
        action="store_false",
        dest="improvement_plots",
        help="Disable percentage-improvement bar charts.",
    )
    parser.add_argument(
        "--aca-schemes",
        "--ara-schemes",
        dest="ara_schemes",
        type=str,
        nargs="*",
        default=["aca_rsa"],
        help="ACA schemes to compare against baselines in percentage-improvement plots.",
    )
    parser.add_argument(
        "--baseline-schemes",
        type=str,
        nargs="*",
        default=[
            "k1_rsa",
            "k3_rsa",
            "fsa_rsa_ssfm",
            "fsa_k3_rsa",
            "rb_scsp_arsa",
            "mp_ss_rsa",
            "adaptive_mp_ss_rsa",
        ],
        help="Baseline schemes for percentage-improvement plots.",
    )
    parser.add_argument(
        "--table-aca-scheme",
        "--table-ara-scheme",
        dest="table_ara_scheme",
        type=str,
        default="aca_rsa",
        help="Proposed scheme used for the compact Table-5-style improvement table.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["aggregate", "nominal"],
        default="nominal",
        help=(
            "Use the dimensionless nominal load index rho (default), or the "
            "topology-specific aggregate offered load in Erlangs."
        ),
    )
    return parser.parse_args()


def find_latest_summary_csv() -> Path:
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    candidates: List[Path] = []
    for root in roots:
        candidates.extend(root.glob("results*/**/*summary*.csv"))
        candidates.extend(root.glob("**/*summary*.csv"))

    unique_candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    if not unique_candidates:
        raise FileNotFoundError("Could not find a summary CSV. Pass --summary-csv explicitly.")
    return max(unique_candidates, key=lambda p: p.stat().st_mtime)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"scheme", "topology", "load"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["load"] = pd.to_numeric(out["load"], errors="coerce")
    out["topology"] = out["topology"].astype(str).str.upper()
    out["scheme"] = out["scheme"].astype(str)
    return out.dropna(subset=["load"])


def filter_df(
    df: pd.DataFrame,
    schemes: Optional[Iterable[str]],
    topologies: Optional[Iterable[str]],
) -> pd.DataFrame:
    out = df.copy()
    if schemes:
        scheme_set = set(schemes)
        out = out[out["scheme"].isin(scheme_set)]
    if topologies:
        topology_set = {topology.upper() for topology in topologies}
        out = out[out["topology"].isin(topology_set)]
    return out


def make_output_dir(summary_csv: Path, output_dir: Optional[str]) -> Path:
    outdir = (
        Path(output_dir)
        if output_dir
        else summary_csv.parent / "plots_by_topology_v31"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in value)


def metric_filename_stem(mean_col: str) -> str:
    if mean_col == "avg_LFM_mean":
        return "avg_FDM"
    return mean_col.replace("_mean", "")


def format_load(load: float) -> str:
    return str(int(load)) if float(load).is_integer() else str(load)


def x_axis_values(sdf: pd.DataFrame, x_axis: str) -> pd.Series:
    aggregate_col = "actual_aggregate_offered_load_erlangs_mean"
    if x_axis == "aggregate" and aggregate_col in sdf.columns:
        return pd.to_numeric(sdf[aggregate_col], errors="coerce")
    nominal_col = "nominal_load_index_mean"
    if nominal_col in sdf.columns:
        return pd.to_numeric(sdf[nominal_col], errors="coerce")
    return pd.to_numeric(sdf["load"], errors="coerce")


def topology_load_scale(tdf: pd.DataFrame) -> float:
    aggregate_col = "actual_aggregate_offered_load_erlangs_mean"
    valid = tdf[["load", aggregate_col]].dropna()
    valid = valid[valid["load"] != 0]
    if valid.empty:
        raise ValueError("Cannot derive the topology aggregate-load scale.")
    ratios = valid[aggregate_col].astype(float) / valid["load"].astype(float)
    scale = float(ratios.median())
    if not all(math.isclose(float(value), scale, rel_tol=0.0, abs_tol=1e-9) for value in ratios):
        raise ValueError("Inconsistent aggregate offered-load scaling in the summary CSV.")
    return scale


def format_axis_load(value: float) -> str:
    rounded = math.floor(float(value) * 10.0 + 0.5) / 10.0
    return f"{rounded:.0f}" if rounded.is_integer() else f"{rounded:.1f}"


def configure_load_axis(ax: plt.Axes, x_axis: str, tdf: pd.DataFrame) -> None:
    aggregate_col = "actual_aggregate_offered_load_erlangs_mean"
    if x_axis == "aggregate" and aggregate_col in tdf.columns:
        scale = topology_load_scale(tdf)
        ticks = [load * scale for load in LOAD_TICKS]
        ax.set_xlabel("Aggregate offered load (Erlangs)")
        ax.set_xticks(ticks)
        ax.set_xticklabels([format_axis_load(value) for value in ticks])
        ax.set_xlim(ticks[0], ticks[-1])
    else:
        ax.set_xlabel(r"Nominal load index, $\rho$")
        ax.set_xticks(LOAD_TICKS)
        ax.set_xlim(LOAD_TICKS[0], LOAD_TICKS[-1])


def available_metrics(df: pd.DataFrame) -> List[Tuple[str, str, str, str]]:
    metrics = []
    for mean_col, ci_col, title, ylabel in DEFAULT_METRICS:
        if mean_col in df.columns:
            metrics.append((mean_col, ci_col, title, ylabel))
    if not metrics:
        raise ValueError(
            "No *_mean metric columns were found. Use the simulator summary CSV, not the raw CSV."
        )
    return metrics


def plot_line_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    mean_col: str,
    ci_col: str,
    topology: str,
    x_axis: str,
) -> None:
    tdf = df[df["topology"] == topology]
    for scheme in sorted(tdf["scheme"].unique()):
        sdf = tdf[tdf["scheme"] == scheme].sort_values("load")
        if sdf.empty:
            continue
        x = x_axis_values(sdf, x_axis).to_numpy()
        y = sdf[mean_col].to_numpy()
        yerr = sdf[ci_col].to_numpy() if ci_col in sdf.columns else None
        style = scheme_style(scheme)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            markersize=5.0,
            capsize=3,
            label=scheme_label(scheme),
        )

    configure_load_axis(ax, x_axis, tdf)
    ax.grid(True, alpha=0.3)


def plot_bar_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    mean_col: str,
    ci_col: str,
    topology: str,
    x_axis: str,
) -> None:
    tdf = df[df["topology"] == topology]
    loads = list(LOAD_TICKS)
    schemes = sorted(tdf["scheme"].unique().tolist())
    if not loads or not schemes:
        return

    width = 0.8 / max(1, len(schemes))
    centers = list(range(len(loads)))
    for i, scheme in enumerate(schemes):
        sdf = tdf[tdf["scheme"] == scheme].set_index("load")
        heights = []
        errors = []
        for load in loads:
            if load in sdf.index:
                row = sdf.loc[load]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                heights.append(row[mean_col])
                errors.append(row[ci_col] if ci_col in sdf.columns else 0.0)
            else:
                heights.append(math.nan)
                errors.append(0.0)

        offsets = [center - 0.4 + width / 2 + i * width for center in centers]
        ax.bar(
            offsets,
            heights,
            width=width,
            yerr=errors,
            capsize=3,
            label=scheme_label(scheme),
            hatch=["", "//", "\\\\", "..", "xx", "++", "**"][i % 7],
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(centers)
    if x_axis == "aggregate" and "actual_aggregate_offered_load_erlangs_mean" in tdf.columns:
        scale = topology_load_scale(tdf)
        tick_labels = [format_axis_load(load * scale) for load in loads]
        ax.set_xlabel("Aggregate offered load (Erlangs)")
    else:
        tick_labels = [format_load(load) for load in loads]
        ax.set_xlabel(r"Nominal load index, $\rho$")
    ax.set_xticklabels(tick_labels)
    ax.grid(True, axis="y", alpha=0.3)


def plot_metric_by_topology(
    df: pd.DataFrame,
    topologies: Sequence[str],
    mean_col: str,
    ci_col: str,
    title: str,
    ylabel: str,
    style: str,
    outpath: Path,
    show: bool,
    dpi: int,
    x_axis: str,
) -> None:
    present_topologies = [topology for topology in topologies if topology in set(df["topology"])]
    if not present_topologies:
        return

    fig, axes = plt.subplots(
        1,
        len(present_topologies),
        figsize=(max(6.8, 5.4 * len(present_topologies)), 4.4),
        sharey=True,
        squeeze=False,
    )
    axes_list = list(axes[0])

    for idx, (ax, topology) in enumerate(zip(axes_list, present_topologies)):
        if style == "bar":
            plot_bar_panel(ax, df, mean_col, ci_col, topology, x_axis)
        else:
            plot_line_panel(ax, df, mean_col, ci_col, topology, x_axis)
        ax.set_ylabel(ylabel)
        apply_publication_axis_style(ax, show_ylabel=(idx == 0))
        ax.text(
            0.5,
            -0.27,
            format_topology_caption(idx, topology),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
        add_subplot_legend(ax, mean_col)

    # Figure title omitted for manuscript-ready plots.
    # fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_metric_single_topology(
    df: pd.DataFrame,
    topology: str,
    mean_col: str,
    ci_col: str,
    ylabel: str,
    style: str,
    outpath: Path,
    show: bool,
    dpi: int,
    x_axis: str,
) -> None:
    if topology not in set(df["topology"]):
        return

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    if style == "bar":
        plot_bar_panel(ax, df, mean_col, ci_col, topology, x_axis)
    else:
        plot_line_panel(ax, df, mean_col, ci_col, topology, x_axis)
    ax.set_ylabel(ylabel)
    apply_publication_axis_style(ax, show_ylabel=True)
    add_subplot_legend(ax, mean_col)

    # No in-figure title or subfigure label: LaTeX owns the caption.
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def load_region(load: float) -> Optional[str]:
    load_int = int(load)
    for region, loads in LOAD_REGIONS.items():
        if load_int in loads:
            return region
    return None


def percentage_improvement(ara_value: float, baseline_value: float, direction: str) -> Optional[float]:
    if baseline_value == 0 or math.isnan(baseline_value) or math.isnan(ara_value):
        return None
    if direction == "lower":
        return ((baseline_value - ara_value) / baseline_value) * 100.0
    if direction == "higher":
        return ((ara_value - baseline_value) / baseline_value) * 100.0
    raise ValueError(f"Unknown improvement direction: {direction}")


def build_improvement_rows(
    df: pd.DataFrame,
    ara_schemes: Sequence[str],
    baseline_schemes: Sequence[str],
) -> pd.DataFrame:
    rows = []
    baselines = set(baseline_schemes)

    for topology in sorted(df["topology"].unique()):
        topo_df = df[df["topology"] == topology]
        for metric_col, metric_label, direction in IMPROVEMENT_METRICS:
            if metric_col not in topo_df.columns:
                continue

            for load in sorted(topo_df["load"].dropna().unique()):
                region = load_region(load)
                if region is None:
                    continue

                for ara_scheme in ara_schemes:
                    ara_rows = topo_df[
                        (topo_df["scheme"] == ara_scheme) &
                        (topo_df["load"] == load)
                    ]
                    if ara_rows.empty:
                        continue
                    ara_value = float(ara_rows.iloc[0][metric_col])

                    for baseline in sorted(baselines):
                        if baseline == ara_scheme:
                            continue

                        base_rows = topo_df[
                            (topo_df["scheme"] == baseline) &
                            (topo_df["load"] == load)
                        ]
                        if base_rows.empty:
                            continue

                        baseline_value = float(base_rows.iloc[0][metric_col])
                        improvement = percentage_improvement(ara_value, baseline_value, direction)
                        if improvement is None:
                            continue

                        rows.append({
                            "topology": topology,
                            "metric": metric_label,
                            "metric_column": metric_col,
                            "direction": direction,
                            "ara_scheme": ara_scheme,
                            "baseline_scheme": baseline,
                            "load": load,
                            "load_region": region,
                            "ara_value": ara_value,
                            "baseline_value": baseline_value,
                            "percentage_improvement": improvement,
                        })

    return pd.DataFrame(rows)


def summarize_improvements(improvement_df: pd.DataFrame) -> pd.DataFrame:
    if improvement_df.empty:
        return improvement_df

    group_cols = [
        "topology",
        "metric",
        "metric_column",
        "direction",
        "ara_scheme",
        "baseline_scheme",
        "load_region",
    ]
    return (
        improvement_df
        .groupby(group_cols, as_index=False)["percentage_improvement"]
        .mean()
        .sort_values(["topology", "metric", "load_region", "baseline_scheme"])
    )


def plot_improvement_metric(
    summary_df: pd.DataFrame,
    metric: str,
    ara_scheme: str,
    topologies: Sequence[str],
    outpath: Path,
    show: bool,
    dpi: int,
) -> None:
    metric_df = summary_df[summary_df["metric"] == metric]
    present_topologies = [topology for topology in topologies if topology in set(metric_df["topology"])]
    if metric_df.empty or not present_topologies:
        return

    baselines = sorted(metric_df["baseline_scheme"].unique().tolist())
    if not baselines:
        return

    fig, axes = plt.subplots(
        1,
        len(present_topologies),
        figsize=(max(7, 5.2 * len(present_topologies)), 4.2),
        sharey=True,
        squeeze=False,
    )

    for idx, (ax, topology) in enumerate(zip(list(axes[0]), present_topologies)):
        tdf = metric_df[metric_df["topology"] == topology]
        regions = [region for region in ["Low", "Mid", "High"] if region in set(tdf["load_region"])]
        width = 0.8 / max(1, len(baselines))
        centers = list(range(len(regions)))

        for i, baseline in enumerate(baselines):
            heights = []
            for region in regions:
                row = tdf[
                    (tdf["baseline_scheme"] == baseline) &
                    (tdf["load_region"] == region)
                ]
                heights.append(float(row.iloc[0]["percentage_improvement"]) if not row.empty else math.nan)

            offsets = [center - 0.4 + width / 2 + i * width for center in centers]
            ax.bar(
                offsets,
                heights,
                width=width,
                label=SCHEME_DISPLAY_NAMES.get(baseline, baseline),
                hatch=["", "//", "\\\\", "..", "xx", "++", "**"][i % 7],
                edgecolor="black",
                linewidth=0.7,
            )

        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xlabel("Load Region")
        ax.set_xticks(centers)
        ax.set_xticklabels(regions)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylabel("Improvement (%)")
        apply_publication_axis_style(ax, show_ylabel=(idx == 0))
        ax.text(
            0.5,
            -0.27,
            format_topology_caption(idx, topology),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=10)

    # Figure title omitted for manuscript-ready plots.
    # fig.suptitle(f"Average Percentage Improvement of {SCHEME_DISPLAY_NAMES.get(ara_scheme, ara_scheme)} over Baselines ({metric})")
    fig.tight_layout(rect=(0, 0.20 if handles else 0.10, 1, 0.98))
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_improvement_metric_single_topology(
    summary_df: pd.DataFrame,
    metric: str,
    topology: str,
    outpath: Path,
    show: bool,
    dpi: int,
) -> None:
    metric_df = summary_df[
        (summary_df["metric"] == metric) &
        (summary_df["topology"] == topology)
    ]
    if metric_df.empty:
        return

    baselines = sorted(metric_df["baseline_scheme"].unique().tolist())
    regions = [region for region in ["Low", "Mid", "High"] if region in set(metric_df["load_region"])]
    if not baselines or not regions:
        return

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    width = 0.8 / max(1, len(baselines))
    centers = list(range(len(regions)))

    for i, baseline in enumerate(baselines):
        heights = []
        for region in regions:
            row = metric_df[
                (metric_df["baseline_scheme"] == baseline) &
                (metric_df["load_region"] == region)
            ]
            heights.append(float(row.iloc[0]["percentage_improvement"]) if not row.empty else math.nan)

        offsets = [center - 0.4 + width / 2 + i * width for center in centers]
        ax.bar(
            offsets,
            heights,
            width=width,
            label=SCHEME_DISPLAY_NAMES.get(baseline, baseline),
            hatch=["", "//", "\\\\", "..", "xx", "++", "**"][i % 7],
            edgecolor="black",
            linewidth=0.7,
        )

    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Load Region")
    ax.set_xticks(centers)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Improvement (%)")
    ax.grid(True, axis="y", alpha=0.3)
    apply_publication_axis_style(ax, show_ylabel=True)
    ax.legend(
        loc="best",
        ncol=2,
        fontsize=11,
        frameon=True,
        framealpha=0.88,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=0.7,
    )

    # No in-figure title or subfigure label: LaTeX owns the caption.
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_improvement_outputs(
    df: pd.DataFrame,
    outdir: Path,
    ara_schemes: Sequence[str],
    baseline_schemes: Sequence[str],
    topologies: Sequence[str],
    table_ara_schemes: Sequence[str],
    show: bool,
    dpi: int,
    image_format: str,
) -> None:
    improvement_rows = build_improvement_rows(df, ara_schemes, baseline_schemes)
    if improvement_rows.empty:
        print("No percentage-improvement rows could be computed.")
        return

    improvement_summary = summarize_improvements(improvement_rows)
    improvement_rows.to_csv(outdir / "percentage_improvement_by_load.csv", index=False)
    improvement_summary.to_csv(outdir / "percentage_improvement_by_region.csv", index=False)

    for ara_scheme in table_ara_schemes:
        save_table5_style_outputs(
            improvement_rows=improvement_rows,
            outdir=outdir,
            ara_scheme=ara_scheme,
            baseline_schemes=baseline_schemes,
            topologies=topologies,
            show=show,
            dpi=dpi,
            image_format=image_format,
        )

    for ara_scheme in sorted(improvement_summary["ara_scheme"].unique()):
        ara_df = improvement_summary[improvement_summary["ara_scheme"] == ara_scheme]
        for metric in sorted(ara_df["metric"].unique()):
            for topology in topologies:
                filename = safe_filename(
                    f"percentage_improvement_{ara_scheme}_{metric}_{topology}.{image_format}"
                )
                plot_improvement_metric_single_topology(
                    ara_df,
                    metric=metric,
                    topology=topology,
                    outpath=outdir / filename,
                    show=show,
                    dpi=dpi,
                )


def format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1f}%"


def write_simple_latex_table(df: pd.DataFrame, path: Path) -> None:
    """
    Write a compact LaTeX tabular without pandas.to_latex().
    pandas.to_latex() requires jinja2 in recent pandas versions, which may not
    be installed on this simulation machine.
    """
    columns = list(df.columns)
    alignment = "l" + "c" * max(0, len(columns) - 1)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(columns) + r" \\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_table5_style_dataframe(
    improvement_rows: pd.DataFrame,
    ara_scheme: str,
    baseline_schemes: Sequence[str],
    topologies: Sequence[str],
) -> pd.DataFrame:
    table_rows = []
    ara_df = improvement_rows[improvement_rows["ara_scheme"] == ara_scheme]
    if ara_df.empty:
        return pd.DataFrame()

    for baseline in baseline_schemes:
        baseline_df = ara_df[ara_df["baseline_scheme"] == baseline]
        if baseline_df.empty:
            continue

        row = {"Algorithms": SCHEME_DISPLAY_NAMES.get(baseline, baseline)}
        for topology in topologies:
            for metric_col, metric_label, direction in TABLE_METRICS:
                metric_df = baseline_df[
                    (baseline_df["topology"] == topology) &
                    (baseline_df["metric_column"] == metric_col)
                ]
                col_name = f"{topology} {metric_label}"
                if metric_df.empty:
                    row[col_name] = math.nan
                else:
                    # Table-5-style values are computed from averaged metric
                    # values, not by averaging per-load percentages. This
                    # avoids unstable ratios when a baseline blocking value is
                    # close to zero at low load.
                    ara_mean = float(metric_df["ara_value"].mean())
                    baseline_mean = float(metric_df["baseline_value"].mean())
                    improvement = percentage_improvement(ara_mean, baseline_mean, direction)
                    row[col_name] = math.nan if improvement is None else improvement
        table_rows.append(row)

    return pd.DataFrame(table_rows)


def save_table5_style_outputs(
    improvement_rows: pd.DataFrame,
    outdir: Path,
    ara_scheme: str,
    baseline_schemes: Sequence[str],
    topologies: Sequence[str],
    show: bool,
    dpi: int,
    image_format: str,
) -> None:
    table_df = build_table5_style_dataframe(
        improvement_rows=improvement_rows,
        ara_scheme=ara_scheme,
        baseline_schemes=baseline_schemes,
        topologies=topologies,
    )
    if table_df.empty:
        print(f"No Table-5-style rows could be computed for {ara_scheme}.")
        return

    csv_path = outdir / safe_filename(f"table5_style_percentage_improvement_{ara_scheme}.csv")
    latex_path = outdir / safe_filename(f"table5_style_percentage_improvement_{ara_scheme}.tex")
    image_path = outdir / safe_filename(f"table5_style_percentage_improvement_{ara_scheme}.{image_format}")

    table_df.to_csv(csv_path, index=False)

    display_df = table_df.copy()
    for col in display_df.columns:
        if col != "Algorithms":
            display_df[col] = display_df[col].map(format_percent)

    write_simple_latex_table(display_df, latex_path)

    fig_width = max(12, 1.05 * len(display_df.columns))
    fig_height = max(2.6, 0.45 * len(display_df) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
        if col == 0:
            cell.set_text_props(ha="left")
    # Table plot title omitted for manuscript-ready output.
    # ax.set_title(
    #     f"Average Percentage Improvement of {SCHEME_DISPLAY_NAMES.get(ara_scheme, ara_scheme)} over Baselines",
    #     pad=12,
    #     fontweight="bold",
    # )
    fig.tight_layout()
    fig.savefig(image_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def write_plot_summary(
    df: pd.DataFrame,
    outdir: Path,
    summary_csv: Path,
    metrics: Sequence[Tuple[str, str, str, str]],
) -> None:
    lines = [
        "Plot summary",
        "=" * 72,
        f"Summary CSV: {summary_csv}",
        "CI handling: uses precomputed *_ci95 columns from the simulator summary CSV.",
        "",
        f"Topologies: {', '.join(sorted(df['topology'].unique()))}",
        f"Schemes: {', '.join(sorted(df['scheme'].unique()))}",
        "Loads: " + ", ".join(format_load(load) for load in sorted(df["load"].unique())),
        "",
        "Metrics plotted:",
    ]
    for mean_col, ci_col, _, _ in metrics:
        ci_note = ci_col if ci_col in df.columns else "no CI column found; plotted without error bars"
        lines.append(f"  - {mean_col} with {ci_note}")

    (outdir / "plot_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_latex_subfigure_snippets(
    outdir: Path,
    metrics: Sequence[Tuple[str, str, str, str]],
    topologies: Sequence[str],
    image_format: str,
) -> None:
    lines = [
        "% LaTeX subfigure snippets generated for individually exported panels.",
        "% Add this package in the preamble if it is not already present:",
        "% \\usepackage[caption=false,font=footnotesize]{subfig}",
        "",
    ]

    for mean_col, _, title, _ in metrics:
        metric_name = safe_filename(metric_filename_stem(mean_col))
        lines.extend([
            f"% {title}",
            "\\begin{figure*}[t]",
            "\\centering",
        ])
        for topology in topologies:
            filename = safe_filename(f"{metric_name}_{topology}.{image_format}")
            caption = TOPOLOGY_DISPLAY_NAMES.get(topology, topology)
            lines.extend([
                f"\\subfloat[{caption}]{{%",
                f"\\includegraphics[width=0.32\\textwidth]{{{filename}}}%",
                "}",
            ])
        lines.extend([
            f"\\caption{{{title} under the evaluated offered loads.}}",
            f"\\label{{fig:{metric_name.lower()}_by_topology}}",
            "\\end{figure*}",
            "",
        ])

    (outdir / "latex_subfigure_snippets.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    summary_csv = Path(args.summary_csv) if args.summary_csv else find_latest_summary_csv()
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    # The summary file is small; the Python engine avoids rare low-memory
    # failures seen with pandas' C parser on this Windows setup.
    df = pd.read_csv(summary_csv, engine="python")
    df = clean_df(df)
    df = filter_df(df, args.schemes, args.topologies)
    if df.empty:
        raise ValueError("No rows left after filtering. Check --schemes and --topologies.")

    topologies = [topology.upper() for topology in args.topologies]
    metrics = available_metrics(df)
    outdir = make_output_dir(summary_csv, args.output_dir)

    for mean_col, ci_col, title, ylabel in metrics:
        metric_name = safe_filename(metric_filename_stem(mean_col))
        for topology in topologies:
            filename = safe_filename(f"{metric_name}_{topology}.{args.image_format}")
            plot_metric_single_topology(
                df=df,
                topology=topology,
                mean_col=mean_col,
                ci_col=ci_col,
                ylabel=ylabel,
                style=args.style,
                outpath=outdir / filename,
                show=args.show,
                dpi=args.dpi,
                x_axis=args.x_axis,
            )

    if args.improvement_plots:
        save_improvement_outputs(
            df=df,
            outdir=outdir,
            ara_schemes=args.ara_schemes,
            baseline_schemes=args.baseline_schemes,
            topologies=topologies,
            table_ara_schemes=[args.table_ara_scheme] if args.table_ara_scheme else args.ara_schemes,
            show=args.show,
            dpi=args.dpi,
            image_format=args.image_format,
        )

    write_plot_summary(df, outdir, summary_csv, metrics)
    write_latex_subfigure_snippets(outdir, metrics, topologies, args.image_format)
    print(f"Summary CSV: {summary_csv}")
    print(f"Plots saved to: {outdir}")
    print("Topology panels: exported as separate files for LaTeX subfigures.")
    print(f"Load axis: {args.x_axis}.")
    print("CI handling: used existing *_ci95 columns; no CI was recomputed.")


if __name__ == "__main__":
    main()

