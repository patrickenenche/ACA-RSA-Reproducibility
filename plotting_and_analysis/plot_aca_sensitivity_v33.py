"""Publication plots for the ACA-RSA v33 bound-sensitivity experiment.

The script consumes the seed-averaged summary CSV for admission, resource,
and timing metrics. It streams the potentially multi-gigabyte K-debug CSV
once and caches compact seed-level aggregates for fallback diagnostics.

Each output PDF contains one panel only. The generated LaTeX file groups
those panels with ``\subfloat`` so the manuscript does not embed merged
multi-panel graphics.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


TOPOLOGIES = ["USNET", "GERMAN", "TI"]
TOPOLOGY_LABELS = {"USNET": "USNET", "GERMAN": "German", "TI": "TI"}
LOADS = [10, 20, 25]
LOAD_COLORS = {10: "#0072B2", 20: "#D55E00", 25: "#009E73"}

SWEEPS: Mapping[str, Mapping[str, object]] = {
    "K_max": {
        "column": "K_max",
        "values": [1, 2, 3, 4],
        "reference": 3,
        "xlabel": r"Maximum candidate paths, $K_{\max}$",
        "slug": "kmax",
        "caption": r"$K_{\max}$ sensitivity",
    },
    "M_path": {
        "column": "M_path",
        "values": [1, 2, 3, 4],
        "reference": 3,
        "xlabel": r"Maximum segments per path, $M_{\mathrm{path}}$",
        "slug": "mpath",
        "caption": r"$M_{\mathrm{path}}$ sensitivity",
    },
    "M_req": {
        "column": "M_req",
        "values": [2, 4, 6, 8],
        "reference": 6,
        "xlabel": r"Maximum segments per request, $M_{\mathrm{req}}$",
        "slug": "mreq",
        "caption": r"$M_{\mathrm{req}}$ sensitivity",
    },
}

SUMMARY_METRICS: Mapping[str, Mapping[str, object]] = {
    "rbp": {
        "mean": "RBP_mean",
        "ci": "RBP_ci95",
        "ylabel": "Request blocking probability",
        "caption": "request blocking probability",
        "fsa": True,
    },
    "bbp": {
        "mean": "BBP_mean",
        "ci": "BBP_ci95",
        "ylabel": "Bandwidth blocking probability",
        "caption": "bandwidth blocking probability",
        "fsa": True,
    },
    "slot_hops_per_gbps": {
        "mean": "slot_hops_per_accepted_gbps_mean",
        "ci": "slot_hops_per_accepted_gbps_ci95",
        "ylabel": "Slot-hops per accepted Gbps",
        "caption": "slot-hop cost per accepted Gbps",
        "fsa": True,
    },
    "decision_time": {
        "mean": "mean_decision_time_us_mean",
        "ci": "mean_decision_time_us_ci95",
        "ylabel": r"Mean decision time ($\mu$s)",
        "caption": "mean decision time",
        "fsa": True,
    },
}

DEBUG_METRICS: Mapping[str, Mapping[str, object]] = {
    "fallback_success": {
        "mean": "fallback_success_rate_pct_mean",
        "ci": "fallback_success_rate_pct_ci95",
        "ylabel": "Fallback success rate (%)",
        "caption": "fallback success rate",
    },
    "k_eff_success": {
        "mean": "avg_k_eff_success_mean",
        "ci": "avg_k_eff_success_ci95",
        "ylabel": r"Average $K_{\mathrm{eff}}$ among successful fallbacks",
        "caption": r"average $K_{\mathrm{eff}}$ among successful fallbacks",
    },
}


def t_critical_95(num_samples: int) -> float:
    if num_samples <= 1:
        return 0.0
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    degrees = num_samples - 1
    if degrees in table:
        return table[degrees]
    if degrees < 25:
        return table[20]
    if degrees < 30:
        return table[25]
    return 1.96


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def locate_latest_results(search_root: Path) -> Path:
    matches = sorted(
        search_root.glob(
            "results_ACA_RSA_v33_sens_K1-4_MP1-4_MR2-4-6-8_full_*"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No completed v33 sensitivity directory was found in {search_root}."
        )
    return matches[0]


def locate_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one file matching {pattern!r} in {directory}; found {len(matches)}."
        )
    return matches[0]


def validate_summary(summary: pd.DataFrame) -> None:
    required = {
        "scheme",
        "sensitivity_id",
        "sweep_membership",
        "K_max",
        "M_path",
        "M_req",
        "topology",
        "load",
        "num_seeds",
    }
    for spec in SUMMARY_METRICS.values():
        required.add(str(spec["mean"]))
        required.add(str(spec["ci"]))
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary CSV is missing required columns: {missing}")

    aca = summary[summary["scheme"] == "aca_rsa"]
    fsa = summary[summary["scheme"] == "fsa_rsa_ssfm"]
    expected_configs = 10
    if aca["sensitivity_id"].nunique() != expected_configs:
        raise ValueError(
            f"Expected {expected_configs} ACA configurations; "
            f"found {aca['sensitivity_id'].nunique()}."
        )
    if set(aca["load"].astype(int)) != set(LOADS):
        raise ValueError("ACA summary does not contain exactly loads 10, 20, and 25.")
    if fsa.empty or set(fsa["sensitivity_id"]) != {"fsa_control"}:
        raise ValueError("The fixed FSA control is missing or incorrectly identified.")
    if set(summary["num_seeds"].astype(int)) != {10}:
        raise ValueError("The full sensitivity summary must contain ten seeds per row.")


def sweep_rows(
    frame: pd.DataFrame,
    topology: str,
    load: int,
    sweep_name: str,
) -> pd.DataFrame:
    selected = frame[
        (frame["scheme"] == "aca_rsa")
        & (frame["topology"].str.upper() == topology)
        & (frame["load"].astype(int) == load)
        & frame["sweep_membership"].astype(str).str.split("|").apply(
            lambda memberships: sweep_name in memberships
        )
    ].copy()
    column = str(SWEEPS[sweep_name]["column"])
    return selected.sort_values(column)


def fsa_row(frame: pd.DataFrame, topology: str, load: int) -> pd.Series:
    selected = frame[
        (frame["scheme"] == "fsa_rsa_ssfm")
        & (frame["topology"].str.upper() == topology)
        & (frame["load"].astype(int) == load)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one FSA control row for {topology}, load={load}; found {len(selected)}."
        )
    return selected.iloc[0]


def style_axis(ax: plt.Axes, xlabel: str, ylabel: str, ticks: Sequence[int]) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(list(ticks))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def add_semantic_legend(ax: plt.Axes, include_fsa: bool) -> None:
    handles: List[Line2D] = [
        Line2D(
            [0],
            [0],
            color=LOAD_COLORS[load],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=5,
            label=rf"$\rho={load}$",
        )
        for load in LOADS
    ]
    if include_fsa:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="-",
                    linewidth=2.5,
                    label="ACA-RSA",
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    label="FSA-RSA(SSFM)",
                ),
            ]
        )
    ax.legend(
        handles=handles,
        loc="best",
        ncol=2,
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=1.8,
        columnspacing=0.8,
    )


def plot_summary_panel(
    summary: pd.DataFrame,
    metric_slug: str,
    sweep_name: str,
    topology: str,
    output_path: Path,
    image_format: str,
    dpi: int,
) -> None:
    metric = SUMMARY_METRICS[metric_slug]
    sweep = SWEEPS[sweep_name]
    x_values = [int(value) for value in sweep["values"]]
    mean_column = str(metric["mean"])
    ci_column = str(metric["ci"])

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for load in LOADS:
        rows = sweep_rows(summary, topology, load, sweep_name)
        x = rows[str(sweep["column"])].astype(int).tolist()
        if x != x_values:
            raise ValueError(
                f"Unexpected {sweep_name} values for {topology}, load={load}: {x}"
            )
        y = rows[mean_column].astype(float).tolist()
        yerr = rows[ci_column].astype(float).tolist()
        color = LOAD_COLORS[load]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=5,
            capsize=3,
            zorder=3,
        )

        control = fsa_row(summary, topology, load)
        control_mean = float(control[mean_column])
        control_ci = float(control[ci_column])
        ax.plot(
            x_values,
            [control_mean] * len(x_values),
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
            zorder=2,
        )
        ax.fill_between(
            x_values,
            control_mean - control_ci,
            control_mean + control_ci,
            color=color,
            alpha=0.05,
            linewidth=0,
            zorder=1,
        )

    ax.axvline(
        int(sweep["reference"]),
        color="0.35",
        linestyle=":",
        linewidth=1.4,
        zorder=0,
    )
    style_axis(
        ax,
        str(sweep["xlabel"]),
        str(metric["ylabel"]),
        x_values,
    )
    if metric_slug in {"rbp", "bbp"}:
        ax.set_ylim(bottom=0.0)
    add_semantic_legend(ax, include_fsa=True)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(f".{image_format}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@dataclass
class SeedDebugAggregate:
    attempts: int = 0
    successes: int = 0
    successful_k_eff_sum: int = 0

    def add(self, success: int, k_eff: int) -> None:
        self.attempts += 1
        if success == 1:
            self.successes += 1
            self.successful_k_eff_sum += k_eff


def build_debug_seed_cache(debug_csv: Path, cache_csv: Path) -> pd.DataFrame:
    aggregates: Dict[Tuple[str, str, int, int, int, int, int], SeedDebugAggregate]
    aggregates = defaultdict(SeedDebugAggregate)
    processed = 0

    with debug_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "scheme",
            "sensitivity_id",
            "topology",
            "seed",
            "load",
            "K_max",
            "M_path",
            "M_req",
            "success",
            "K_effective",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Debug CSV is missing columns: {sorted(missing)}")

        for row in reader:
            if row.get("scheme", "").lower() != "aca_rsa":
                continue
            key = (
                str(row["sensitivity_id"]),
                str(row["topology"]).upper(),
                as_int(row["load"]),
                as_int(row["seed"]),
                as_int(row["K_max"]),
                as_int(row["M_path"]),
                as_int(row["M_req"]),
            )
            aggregates[key].add(
                success=as_int(row["success"]),
                k_eff=as_int(row["K_effective"]),
            )
            processed += 1
            if processed % 2_000_000 == 0:
                print(f"  streamed {processed:,} fallback rows...", flush=True)

    rows: List[Dict[str, object]] = []
    for key, aggregate in sorted(aggregates.items()):
        sensitivity_id, topology, load, seed, k_max, m_path, m_req = key
        success_rate = (
            100.0 * aggregate.successes / aggregate.attempts
            if aggregate.attempts
            else 0.0
        )
        avg_k_eff_success = (
            aggregate.successful_k_eff_sum / aggregate.successes
            if aggregate.successes
            else 0.0
        )
        rows.append(
            {
                "sensitivity_id": sensitivity_id,
                "topology": topology,
                "load": load,
                "seed": seed,
                "K_max": k_max,
                "M_path": m_path,
                "M_req": m_req,
                "fallback_attempts": aggregate.attempts,
                "fallback_successes": aggregate.successes,
                "fallback_success_rate_pct": success_rate,
                "avg_k_eff_success": avg_k_eff_success,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(cache_csv, index=False)
    print(
        f"Saved {len(frame):,} seed-level debug aggregates from "
        f"{processed:,} streamed rows to {cache_csv}",
        flush=True,
    )
    return frame


def load_or_build_debug_cache(
    debug_csv: Path,
    cache_csv: Path,
    rebuild: bool,
) -> pd.DataFrame:
    if (
        not rebuild
        and cache_csv.exists()
        and cache_csv.stat().st_mtime >= debug_csv.stat().st_mtime
    ):
        print(f"Using cached seed-level fallback statistics: {cache_csv}", flush=True)
        return pd.read_csv(cache_csv)
    print(f"Streaming large fallback debug file: {debug_csv}", flush=True)
    return build_debug_seed_cache(debug_csv, cache_csv)


def summarize_debug_seed_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "sensitivity_id",
        "topology",
        "load",
        "K_max",
        "M_path",
        "M_req",
    ]
    output: List[Dict[str, object]] = []
    for key, group in seed_rows.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, key))
        row["scheme"] = "aca_rsa"
        row["sweep_membership"] = ""
        sensitivity_id = str(row["sensitivity_id"])
        if sensitivity_id == "K3_MP3_MR6":
            row["sweep_membership"] = "K_max|M_path|M_req"
        elif sensitivity_id.startswith(("K1_", "K2_", "K4_")):
            row["sweep_membership"] = "K_max"
        elif "_MP1_" in sensitivity_id or "_MP2_" in sensitivity_id or "_MP4_" in sensitivity_id:
            row["sweep_membership"] = "M_path"
        elif sensitivity_id.endswith(("MR2", "MR4", "MR8")):
            row["sweep_membership"] = "M_req"

        row["num_seeds"] = len(group)
        for metric in ["fallback_success_rate_pct", "avg_k_eff_success"]:
            values = [float(value) for value in group[metric]]
            metric_mean = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0
            metric_ci = (
                t_critical_95(len(values)) * metric_std / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric}_mean"] = metric_mean
            row[f"{metric}_std"] = metric_std
            row[f"{metric}_ci95"] = metric_ci
        output.append(row)
    return pd.DataFrame(output)


def plot_debug_panel(
    debug_summary: pd.DataFrame,
    metric_slug: str,
    sweep_name: str,
    topology: str,
    output_path: Path,
    image_format: str,
    dpi: int,
) -> None:
    metric = DEBUG_METRICS[metric_slug]
    sweep = SWEEPS[sweep_name]
    x_values = [int(value) for value in sweep["values"]]

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for load in LOADS:
        rows = sweep_rows(debug_summary, topology, load, sweep_name)
        x = rows[str(sweep["column"])].astype(int).tolist()
        if x != x_values:
            raise ValueError(
                f"Unexpected debug {sweep_name} values for {topology}, load={load}: {x}"
            )
        ax.errorbar(
            x,
            rows[str(metric["mean"])].astype(float),
            yerr=rows[str(metric["ci"])].astype(float),
            color=LOAD_COLORS[load],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=5,
            capsize=3,
        )

    ax.axvline(
        int(sweep["reference"]),
        color="0.35",
        linestyle=":",
        linewidth=1.4,
        zorder=0,
    )
    style_axis(
        ax,
        str(sweep["xlabel"]),
        str(metric["ylabel"]),
        x_values,
    )
    if metric_slug == "fallback_success":
        ax.set_ylim(0.0, 100.0)
    else:
        ax.set_ylim(bottom=0.0)
    add_semantic_legend(ax, include_fsa=False)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(f".{image_format}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_reporting_csv(
    summary: pd.DataFrame,
    debug_summary: Optional[pd.DataFrame],
    output_path: Path,
) -> None:
    columns = [
        "scheme",
        "sensitivity_id",
        "sweep_membership",
        "K_max",
        "M_path",
        "M_req",
        "topology",
        "load",
        "num_seeds",
    ]
    for metric in SUMMARY_METRICS.values():
        columns.extend([str(metric["mean"]), str(metric["ci"])])
    reporting = summary[columns].copy()

    if debug_summary is not None:
        debug_columns = [
            "sensitivity_id",
            "topology",
            "load",
            "fallback_success_rate_pct_mean",
            "fallback_success_rate_pct_ci95",
            "avg_k_eff_success_mean",
            "avg_k_eff_success_ci95",
        ]
        reporting = reporting.merge(
            debug_summary[debug_columns],
            on=["sensitivity_id", "topology", "load"],
            how="left",
        )
    reporting.to_csv(output_path, index=False)


def write_latex_snippets(
    output_dir: Path,
    image_format: str,
    include_debug: bool,
) -> None:
    metric_specs: Dict[str, Mapping[str, object]] = dict(SUMMARY_METRICS)
    if include_debug:
        metric_specs.update(DEBUG_METRICS)

    lines = [
        "% Generated by plot_aca_sensitivity_v33.py",
        "% Requires: \\usepackage[caption=false,font=footnotesize]{subfig}",
        "% Each included PDF is a separate subfigure file.",
        "",
    ]
    for metric_slug, metric in metric_specs.items():
        for sweep_name, sweep in SWEEPS.items():
            lines.extend(["\\begin{figure*}[t]", "\\centering"])
            for index, topology in enumerate(TOPOLOGIES):
                filename = (
                    f"{metric_slug}_{sweep['slug']}_{topology}.{image_format}"
                )
                suffix = " \\" if index < len(TOPOLOGIES) - 1 else ""
                lines.append(
                    "\\subfloat["
                    + TOPOLOGY_LABELS[topology]
                    + "]{\\includegraphics[width=0.32\\textwidth]{"
                    + filename
                    + "}}"
                    + suffix
                )
            lines.extend(
                [
                    "\\caption{ACA-RSA "
                    + str(sweep["caption"])
                    + " for "
                    + str(metric["caption"])
                    + ". The vertical dotted line marks the reference configuration.}",
                    "\\label{fig:sensitivity_"
                    + metric_slug
                    + "_"
                    + str(sweep["slug"])
                    + "}",
                    "\\end{figure*}",
                    "",
                ]
            )
    (output_dir / "aca_sensitivity_v33_subfigures.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready ACA-RSA v33 sensitivity plots."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Completed v33 results directory. Defaults to the newest beside this script.",
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--debug-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "svg"],
        default="pdf",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--skip-debug",
        action="store_true",
        help="Skip fallback success and K_eff plots.",
    )
    parser.add_argument(
        "--rebuild-debug-cache",
        action="store_true",
        help="Re-read the complete debug CSV even when a current cache exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else locate_latest_results(script_dir)
    )
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else locate_one(results_dir, "simulation_results_summary_*10seeds.csv")
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "sensitivity_plots_v33_pdf"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading summary: {summary_csv}", flush=True)
    summary = pd.read_csv(summary_csv)
    validate_summary(summary)

    generated = 0
    for metric_slug in SUMMARY_METRICS:
        for sweep_name, sweep in SWEEPS.items():
            for topology in TOPOLOGIES:
                output_base = output_dir / (
                    f"{metric_slug}_{sweep['slug']}_{topology}"
                )
                plot_summary_panel(
                    summary,
                    metric_slug,
                    sweep_name,
                    topology,
                    output_base,
                    args.format,
                    args.dpi,
                )
                generated += 1

    debug_summary: Optional[pd.DataFrame] = None
    if not args.skip_debug:
        debug_csv = (
            args.debug_csv.resolve()
            if args.debug_csv is not None
            else locate_one(results_dir, "checkpoint_k_debug_per_seed_*.csv")
        )
        cache_csv = output_dir / "sensitivity_debug_seed_aggregates.csv"
        debug_seed_rows = load_or_build_debug_cache(
            debug_csv,
            cache_csv,
            rebuild=args.rebuild_debug_cache,
        )
        debug_summary = summarize_debug_seed_rows(debug_seed_rows)
        debug_summary.to_csv(
            output_dir / "sensitivity_debug_summary_with_ci95.csv",
            index=False,
        )
        if set(debug_summary["num_seeds"].astype(int)) != {10}:
            raise ValueError("Debug summary does not contain ten seeds per configuration.")

        for metric_slug in DEBUG_METRICS:
            for sweep_name, sweep in SWEEPS.items():
                for topology in TOPOLOGIES:
                    output_base = output_dir / (
                        f"{metric_slug}_{sweep['slug']}_{topology}"
                    )
                    plot_debug_panel(
                        debug_summary,
                        metric_slug,
                        sweep_name,
                        topology,
                        output_base,
                        args.format,
                        args.dpi,
                    )
                    generated += 1

    write_reporting_csv(
        summary,
        debug_summary,
        output_dir / "aca_sensitivity_v33_reporting_values.csv",
    )
    write_latex_snippets(
        output_dir,
        args.format,
        include_debug=not args.skip_debug,
    )

    print(f"Generated {generated} individual plot panels in: {output_dir}")
    print("Generated a reporting CSV and LaTeX subfigure snippets.")


if __name__ == "__main__":
    main()
