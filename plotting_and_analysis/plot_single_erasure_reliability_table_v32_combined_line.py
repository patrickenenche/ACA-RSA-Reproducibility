from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


TOPOLOGY_ORDER = ["USNET", "GERMAN", "TI"]
LOAD_TICKS = [5, 10, 15, 20, 25]
ACA_SCHEME = "aca_rsa"
ADAPTIVE_REPETITION_SCHEME = "adaptive_mp_ss_rsa_repetition"
LEGACY_ADAPTIVE_REPETITION_SCHEME = "adaptive_mp_ss_repetition"

SCHEME_COLORS = {
    ADAPTIVE_REPETITION_SCHEME: "#0072B2",
    ACA_SCHEME: "#E69F00",
}
TOPOLOGY_HATCHES = {
    "USNET": "xx",
    "GERMAN": "///",
    "TI": "\\\\",
}
TOPOLOGY_LABELS = {"USNET": "USNET", "GERMAN": "German", "TI": "TI"}
TOPOLOGY_COLORS = {
    "USNET": "#0072B2",
    "GERMAN": "#D55E00",
    "TI": "#009E73",
}
TOPOLOGY_MARKERS = {"USNET": "o", "GERMAN": "s", "TI": "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build equal-redundancy single-erasure outputs from a v32 summary "
            "CSV containing ACA-RSA r=0/r=1 and "
            "A-MP-SS-RSA-Repetition r=1 rows."
        )
    )
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("scheme", "").lower() == LEGACY_ADAPTIVE_REPETITION_SCHEME:
            row["scheme"] = ADAPTIVE_REPETITION_SCHEME
    return rows


def as_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_col(row: Dict[str, str], base_name: str) -> str:
    mean_col = f"{base_name}_mean"
    return mean_col if mean_col in row else base_name


def row_metric(row: Dict[str, str], base_name: str, default: float = math.nan) -> float:
    return as_float(row.get(metric_col(row, base_name)), default)


def row_metric_any(
    row: Dict[str, str],
    base_names: Iterable[str],
    default: float = math.nan,
) -> float:
    for base_name in base_names:
        value = row_metric(row, base_name, math.nan)
        if not math.isnan(value):
            return value
    return default


def row_ci_any(row: Dict[str, str], base_names: Iterable[str]) -> float:
    for base_name in base_names:
        ci_col = f"{base_name}_ci95"
        if ci_col in row and row.get(ci_col, "") not in (None, ""):
            return as_float(row[ci_col], 0.0)
    return 0.0


def topology_sort_key(topology: str) -> Tuple[int, str]:
    upper = topology.upper()
    if upper in TOPOLOGY_ORDER:
        return (TOPOLOGY_ORDER.index(upper), upper)
    return (len(TOPOLOGY_ORDER), upper)


def filter_rows(
    rows: Iterable[Dict[str, str]],
    topology: str,
    scheme: str,
    reliability_r: int,
) -> List[Dict[str, str]]:
    return [
        row for row in rows
        if row.get("topology", "").upper() == topology.upper()
        and row.get("scheme", "").lower() == scheme.lower()
        and int(as_float(row.get("reliability_r"), -1)) == reliability_r
    ]


def mean_metric(rows: List[Dict[str, str]], base_name: str) -> float:
    values = [row_metric(row, base_name) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else math.nan


def mean_metric_any(rows: List[Dict[str, str]], base_names: Iterable[str]) -> float:
    values = [row_metric_any(row, base_names) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else math.nan


def percent(value: float) -> str:
    return "--" if math.isnan(value) else f"{100.0 * value:.2f}"


def signed_percent(value: float) -> str:
    return "--" if math.isnan(value) else f"{value:.2f}"


def admission_cost(rbp_r0: float, rbp_r1: float) -> float:
    if math.isnan(rbp_r0) or math.isnan(rbp_r1) or rbp_r0 == 0:
        return math.nan
    return ((rbp_r1 - rbp_r0) / rbp_r0) * 100.0


def build_compact_table(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    topologies = sorted({row["topology"].upper() for row in rows}, key=topology_sort_key)
    table = []
    dfp_names = ["mean_post_erasure_DFP_accepted", "post_erasure_DFP"]
    for topology in topologies:
        aca_r0 = filter_rows(rows, topology, ACA_SCHEME, 0)
        aca_r1 = filter_rows(rows, topology, ACA_SCHEME, 1)
        adaptive_r1 = filter_rows(rows, topology, ADAPTIVE_REPETITION_SCHEME, 1)

        rbp_r0 = mean_metric(aca_r0, "RBP")
        rbp_r1 = mean_metric(aca_r1, "RBP")
        adaptive_rbp_r1 = mean_metric(adaptive_r1, "RBP")
        cost = admission_cost(rbp_r0, rbp_r1)

        table.append({
            "Topology": topology,
            "ACA setup RBP r_i=0 (%)": percent(rbp_r0),
            "ACA setup RBP r_i=1 (%)": percent(rbp_r1),
            "A-MP-SS-RSA-Repetition setup RBP r_i=1 (%)": percent(adaptive_rbp_r1),
            "Relative admission cost (%)": signed_percent(cost),
            "Post-erasure DFP A-MP-SS-RSA-Repetition (%)": percent(
                mean_metric_any(adaptive_r1, dfp_names)
            ),
            "Post-erasure DFP ACA-RSA r_i=1 (%)": percent(
                mean_metric_any(aca_r1, dfp_names)
            ),
            "E2EFP A-MP-SS-RSA-Repetition (%)": percent(
                mean_metric(adaptive_r1, "E2EFP")
            ),
            "E2EFP ACA-RSA r_i=1 (%)": percent(mean_metric(aca_r1, "E2EFP")),
        })
    return table


def build_per_load_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    topologies = sorted({row["topology"].upper() for row in rows}, key=topology_sort_key)
    detail_rows = []
    for topology in topologies:
        dfp_names = ["mean_post_erasure_DFP_accepted", "post_erasure_DFP"]
        aca_r0_by_load = {
            int(as_float(row["load"])): row
            for row in filter_rows(rows, topology, ACA_SCHEME, 0)
        }
        aca_r1_by_load = {
            int(as_float(row["load"])): row
            for row in filter_rows(rows, topology, ACA_SCHEME, 1)
        }
        adaptive_r1_by_load = {
            int(as_float(row["load"])): row
            for row in filter_rows(rows, topology, ADAPTIVE_REPETITION_SCHEME, 1)
        }
        for load in sorted(
            set(aca_r0_by_load) | set(aca_r1_by_load) | set(adaptive_r1_by_load)
        ):
            row0 = aca_r0_by_load.get(load, {})
            row1 = aca_r1_by_load.get(load, {})
            adaptive = adaptive_r1_by_load.get(load, {})
            rbp_r0 = row_metric(row0, "RBP")
            rbp_r1 = row_metric(row1, "RBP")
            detail_rows.append({
                "topology": topology,
                "load": load,
                "nominal_load_index": row_metric_any(
                    row0, ["nominal_load_index"], default=float(load)
                ),
                "aggregate_offered_load_erlangs": row_metric_any(
                    row0,
                    ["actual_aggregate_offered_load_erlangs"],
                ),
                "aca_rbp_r0": rbp_r0,
                "aca_rbp_r0_ci95": row_ci_any(row0, ["RBP"]),
                "aca_rbp_r1": rbp_r1,
                "aca_rbp_r1_ci95": row_ci_any(row1, ["RBP"]),
                "adaptive_rbp_r1": row_metric(adaptive, "RBP"),
                "adaptive_rbp_r1_ci95": row_ci_any(adaptive, ["RBP"]),
                "admission_cost_percent": admission_cost(rbp_r0, rbp_r1),
                "adaptive_post_erasure_dfp": row_metric_any(adaptive, dfp_names),
                "adaptive_post_erasure_dfp_ci95": row_ci_any(adaptive, dfp_names),
                "aca_post_erasure_dfp_r1": row_metric_any(row1, dfp_names),
                "aca_post_erasure_dfp_r1_ci95": row_ci_any(row1, dfp_names),
                "adaptive_e2efp_r1": row_metric(adaptive, "E2EFP"),
                "adaptive_e2efp_r1_ci95": row_ci_any(adaptive, ["E2EFP"]),
                "aca_e2efp_r1": row_metric(row1, "E2EFP"),
                "aca_e2efp_r1_ci95": row_ci_any(row1, ["E2EFP"]),
            })
    return detail_rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: object) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def write_latex(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    alignment = "l" + "r" * (len(cols) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Equal-redundancy single-erasure comparison of ACA-RSA "
            r"and A-MP-SS-RSA-Repetition under the common $g_i+1$ allocation target.}"
        ),
        r"\label{tab:equal_redundancy_single_erasure}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{latex_escape(col)}}}" for col in cols) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row[col]) for col in cols) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def configure_axis(
    ax: plt.Axes,
    ylabel: str,
    subset: List[Dict[str, object]],
) -> None:
    ax.set_xlabel(r"Nominal load index, $\rho$")
    ax.set_xticks(LOAD_TICKS)
    ax.set_xlim(LOAD_TICKS[0], LOAD_TICKS[-1])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.grid(True, alpha=0.3)


def topology_rows(
    detail_rows: List[Dict[str, object]], topology: str
) -> List[Dict[str, object]]:
    return sorted(
        [row for row in detail_rows if row["topology"] == topology],
        key=lambda row: int(row["load"]),
    )


def plot_setup_rbp(subset: List[Dict[str, object]], outpath: Path) -> None:
    loads = [float(row["nominal_load_index"]) for row in subset]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.errorbar(
        loads,
        [100.0 * float(row["aca_rbp_r0"]) for row in subset],
        yerr=[100.0 * float(row["aca_rbp_r0_ci95"]) for row in subset],
        marker="o",
        linewidth=2.5,
        capsize=3,
        label=r"ACA-RSA $r_i=0$",
    )
    ax.errorbar(
        loads,
        [100.0 * float(row["aca_rbp_r1"]) for row in subset],
        yerr=[100.0 * float(row["aca_rbp_r1_ci95"]) for row in subset],
        marker="s",
        linewidth=2.5,
        capsize=3,
        label=r"ACA-RSA $r_i=1$",
    )
    # The x markers expose the expected equality of the two r=1 setup curves.
    ax.plot(
        loads,
        [100.0 * float(row["adaptive_rbp_r1"]) for row in subset],
        linestyle="none",
        marker="x",
        markersize=7,
        markeredgewidth=1.8,
        label=r"A-MP-SS-RSA-Repetition $r_i=1$",
        zorder=5,
    )
    configure_axis(ax, "Setup RBP (%)", subset)
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_admission_cost(subset: List[Dict[str, object]], outpath: Path) -> None:
    valid = [
        row for row in subset
        if not math.isnan(float(row["admission_cost_percent"]))
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(
        [float(row["nominal_load_index"]) for row in valid],
        [float(row["admission_cost_percent"]) for row in valid],
        marker="o",
        linewidth=2.5,
        label="ACA-RSA / A-MP-SS-RSA-Repetition",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    configure_axis(ax, "Relative admission cost (%)", valid)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_two_scheme_metric(
    subset: List[Dict[str, object]],
    adaptive_key: str,
    aca_key: str,
    ylabel: str,
    outpath: Path,
) -> None:
    loads = [float(row["nominal_load_index"]) for row in subset]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.errorbar(
        loads,
        [100.0 * float(row[adaptive_key]) for row in subset],
        yerr=[100.0 * float(row[f"{adaptive_key}_ci95"]) for row in subset],
        marker="^",
        linestyle="--",
        linewidth=2.5,
        capsize=3,
        label="A-MP-SS-RSA-Repetition",
    )
    ax.errorbar(
        loads,
        [100.0 * float(row[aca_key]) for row in subset],
        yerr=[100.0 * float(row[f"{aca_key}_ci95"]) for row in subset],
        marker="o",
        linestyle="-",
        linewidth=2.5,
        capsize=3,
        label="ACA-RSA",
    )
    configure_axis(ax, ylabel, subset)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_combined_e2efp_bar(
    detail_rows: List[Dict[str, object]],
    outpath: Path,
) -> None:
    """Plot every topology and scheme in one load-grouped E2EFP panel."""
    topology_data = {
        topology: {
            int(row["load"]): row
            for row in topology_rows(detail_rows, topology)
        }
        for topology in TOPOLOGY_ORDER
    }
    for topology, rows_by_load in topology_data.items():
        missing = set(LOAD_TICKS) - set(rows_by_load)
        if missing:
            raise ValueError(
                f"Combined E2EFP plot is missing loads {sorted(missing)} for {topology}."
            )

    group_centers = np.arange(len(LOAD_TICKS), dtype=float)
    total_group_width = 0.84
    topology_width = total_group_width / len(TOPOLOGY_ORDER)
    bar_width = topology_width / 2.0 * 0.88

    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    for topology_index, topology in enumerate(TOPOLOGY_ORDER):
        topology_offset = (
            topology_index - (len(TOPOLOGY_ORDER) - 1) / 2.0
        ) * topology_width
        rows_by_load = topology_data[topology]

        for scheme_index, (value_key, ci_key, scheme) in enumerate(
            [
                (
                    "adaptive_e2efp_r1",
                    "adaptive_e2efp_r1_ci95",
                    ADAPTIVE_REPETITION_SCHEME,
                ),
                (
                    "aca_e2efp_r1",
                    "aca_e2efp_r1_ci95",
                    ACA_SCHEME,
                ),
            ]
        ):
            scheme_offset = (scheme_index - 0.5) * bar_width
            positions = group_centers + topology_offset + scheme_offset
            values = [
                100.0 * float(rows_by_load[load][value_key])
                for load in LOAD_TICKS
            ]
            errors = [
                100.0 * float(rows_by_load[load][ci_key])
                for load in LOAD_TICKS
            ]
            ax.bar(
                positions,
                values,
                width=bar_width,
                yerr=errors,
                color=SCHEME_COLORS[scheme],
                edgecolor="black",
                linewidth=0.65,
                hatch=TOPOLOGY_HATCHES[topology],
                error_kw={
                    "ecolor": "black",
                    "elinewidth": 0.9,
                    "capsize": 2.0,
                    "capthick": 0.9,
                },
                zorder=3,
            )

    legend_handles = [
        Patch(
            facecolor=SCHEME_COLORS[ADAPTIVE_REPETITION_SCHEME],
            edgecolor="black",
            label="A-MP-SS-RSA-Repetition",
        ),
        Patch(
            facecolor=SCHEME_COLORS[ACA_SCHEME],
            edgecolor="black",
            label="ACA-RSA",
        ),
    ]
    legend_handles.extend(
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=TOPOLOGY_HATCHES[topology],
            label=TOPOLOGY_LABELS[topology],
        )
        for topology in TOPOLOGY_ORDER
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        fontsize=9.5,
        frameon=True,
        framealpha=0.94,
        borderpad=0.35,
        handlelength=1.6,
        columnspacing=0.9,
    )
    ax.set_xlabel(r"Nominal load index, $\rho$", fontsize=11)
    ax.set_ylabel("E2EFP (%)", fontsize=11)
    ax.set_xticks(group_centers, [str(load) for load in LOAD_TICKS])
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(group_centers[0] - 0.55, group_centers[-1] + 0.55)
    ax.set_ylim(0.0, 100.0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_combined_e2efp_line(
    detail_rows: List[Dict[str, object]],
    outpath: Path,
) -> None:
    """Plot six compact, publication-distinguishable E2EFP curves."""
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    handles: List[Line2D] = []

    for topology in TOPOLOGY_ORDER:
        subset = topology_rows(detail_rows, topology)
        if [int(row["load"]) for row in subset] != LOAD_TICKS:
            raise ValueError(
                f"Combined E2EFP line plot has incomplete loads for {topology}."
            )
        loads = [float(row["nominal_load_index"]) for row in subset]
        color = TOPOLOGY_COLORS[topology]
        marker = TOPOLOGY_MARKERS[topology]

        for value_key, ci_key, scheme_label, linestyle, filled in [
            (
                "aca_e2efp_r1",
                "aca_e2efp_r1_ci95",
                "ACA-RSA",
                "-",
                True,
            ),
            (
                "adaptive_e2efp_r1",
                "adaptive_e2efp_r1_ci95",
                "A-MP-SS-RSA-Repetition",
                "--",
                False,
            ),
        ]:
            marker_face = color if filled else "white"
            ax.errorbar(
                loads,
                [100.0 * float(row[value_key]) for row in subset],
                yerr=[100.0 * float(row[ci_key]) for row in subset],
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                marker=marker,
                markersize=6.5,
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markeredgewidth=1.3,
                capsize=3,
                elinewidth=1.0,
                zorder=3,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    marker=marker,
                    markersize=6.5,
                    markerfacecolor=marker_face,
                    markeredgecolor=color,
                    markeredgewidth=1.3,
                    label=f"{scheme_label}, {TOPOLOGY_LABELS[topology]}",
                )
            )

    ax.set_xlabel(r"Nominal load index, $\rho$", fontsize=11)
    ax.set_ylabel("E2EFP (%)", fontsize=11)
    ax.set_xticks(LOAD_TICKS)
    ax.set_xlim(LOAD_TICKS[0], LOAD_TICKS[-1])
    ax.set_ylim(0.0, 100.0)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        handles=handles,
        loc="center",
        ncol=1,
        fontsize=8.0,
        frameon=True,
        framealpha=0.94,
        borderpad=0.35,
        handlelength=2.2,
        labelspacing=0.25,
    )
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def write_latex_figure_snippets(outdir: Path, image_format: str) -> None:
    specs = [
        ("setup_rbp", "Setup request blocking probability"),
        ("admission_cost", "Relative admission cost"),
        ("post_erasure_dfp", "Post-erasure decoding-failure probability"),
        ("e2efp", "End-to-end failure probability"),
    ]
    lines = [
        "% LaTeX snippets for separately exported reliability panels.",
        "% Requires: \\usepackage[caption=false,font=footnotesize]{subfig}",
        "",
    ]
    for stub, caption in specs:
        lines.extend([r"\begin{figure*}[t]", r"\centering"])
        for topology in TOPOLOGY_ORDER:
            display = "German" if topology == "GERMAN" else topology
            lines.extend([
                f"\\subfloat[{display}]{{%",
                f"\\includegraphics[width=0.32\\textwidth]{{single_erasure_{stub}_{topology}.{image_format}}}%",
                "}",
            ])
        lines.extend([
            f"\\caption{{{caption} under the equal-redundancy comparison.}}",
            f"\\label{{fig:single_erasure_{stub}}}",
            r"\end{figure*}",
            "",
        ])
    lines.extend([
        r"\begin{figure*}[t]",
        r"\centering",
        (
            f"\\includegraphics[width=0.98\\textwidth]"
            f"{{single_erasure_e2efp_combined_line.{image_format}}}"
        ),
        (
            r"\caption{End-to-end failure probability under the equal-redundancy "
            r"$g_i+1$ single-erasure experiment across the three topologies and "
            r"five nominal offered-load levels. Error bars show 95\% Student-$t$ "
            r"confidence intervals over ten independent seeds.}"
        ),
        r"\label{fig:single_erasure_e2efp_combined_line}",
        r"\end{figure*}",
        "",
    ])
    (outdir / "latex_single_erasure_subfigure_snippets.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    outdir = args.outdir or (
        args.summary_csv.parent / "single_erasure_reliability_v31"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.summary_csv)
    available = {
        (row.get("scheme", "").lower(), int(as_float(row.get("reliability_r"), -1)))
        for row in rows
    }
    required = {
        (ACA_SCHEME, 0),
        (ACA_SCHEME, 1),
        (ADAPTIVE_REPETITION_SCHEME, 1),
    }
    missing = required - available
    if missing:
        raise ValueError(f"Summary CSV is missing required configurations: {sorted(missing)}")

    compact_table = build_compact_table(rows)
    detail_rows = build_per_load_rows(rows)

    write_csv(compact_table, outdir / "single_erasure_reliability_table_equal_redundancy.csv")
    write_latex(compact_table, outdir / "single_erasure_reliability_table_equal_redundancy.tex")
    write_csv(detail_rows, outdir / "single_erasure_reliability_per_load_equal_redundancy.csv")

    topologies = sorted({str(row["topology"]) for row in detail_rows}, key=topology_sort_key)
    for topology in topologies:
        subset = topology_rows(detail_rows, topology)
        plot_setup_rbp(
            subset, outdir / f"single_erasure_setup_rbp_{topology}.{args.format}"
        )
        plot_admission_cost(
            subset, outdir / f"single_erasure_admission_cost_{topology}.{args.format}"
        )
        plot_two_scheme_metric(
            subset,
            "adaptive_post_erasure_dfp",
            "aca_post_erasure_dfp_r1",
            "Post-erasure DFP (%)",
            outdir / f"single_erasure_post_erasure_dfp_{topology}.{args.format}",
        )
        plot_two_scheme_metric(
            subset,
            "adaptive_e2efp_r1",
            "aca_e2efp_r1",
            "E2EFP (%)",
            outdir / f"single_erasure_e2efp_{topology}.{args.format}",
        )

    plot_combined_e2efp_line(
        detail_rows,
        outdir / f"single_erasure_e2efp_combined_line.{args.format}",
    )

    write_latex_figure_snippets(outdir, args.format)
    print(f"Wrote equal-redundancy reliability outputs to: {outdir}")


if __name__ == "__main__":
    main()
