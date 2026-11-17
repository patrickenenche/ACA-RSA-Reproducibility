r"""Grouped-bar sensitivity plots for the ACA-RSA v33 experiment.

The script reads the compact reporting CSV produced by
``plot_aca_sensitivity_v33.py``. It creates six separate publication panels
at the demanding load rho=25: request blocking probability and fallback
success rate for the K_max, M_path, and M_req sweeps. Each parameter value is
represented by grouped German, TI, and USNET bars with seed-level 95% Student-t
confidence intervals. The selected (3, 3, 6) setting is highlighted.

Separate PDF files are generated so that the manuscript can compose them with
LaTeX subfigure commands rather than embedding a merged raster figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


TOPOLOGIES = ["GERMAN", "TI", "USNET"]
TOPOLOGY_LABELS = {"GERMAN": "German", "TI": "TI", "USNET": "USNET"}
TOPOLOGY_COLORS = {
    "GERMAN": "#0072B2",
    "TI": "#D55E00",
    "USNET": "#009E73",
}
TOPOLOGY_HATCHES = {"GERMAN": "///", "TI": "\\\\", "USNET": "xx"}

SWEEPS: Mapping[str, Mapping[str, object]] = {
    "K_max": {
        "column": "K_max",
        "values": [1, 2, 3, 4],
        "reference": 3,
        "strict_id": "K1_MP3_MR6",
        "reference_id": "K3_MP3_MR6",
        "relaxed_id": "K4_MP3_MR6",
        "xlabel": r"Maximum candidate paths, $K_{\max}$",
        "slug": "kmax",
        "caption": r"$K_{\max}$ sensitivity",
        "comparison": r"$1\rightarrow3\rightarrow4$",
    },
    "M_path": {
        "column": "M_path",
        "values": [1, 2, 3, 4],
        "reference": 3,
        "strict_id": "K3_MP1_MR6",
        "reference_id": "K3_MP3_MR6",
        "relaxed_id": "K3_MP4_MR6",
        "xlabel": r"Maximum segments per path, $M_{\mathrm{path}}$",
        "slug": "mpath",
        "caption": r"$M_{\mathrm{path}}$ sensitivity",
        "comparison": r"$1\rightarrow3\rightarrow4$",
    },
    "M_req": {
        "column": "M_req",
        "values": [2, 4, 6, 8],
        "reference": 6,
        "strict_id": "K3_MP3_MR2",
        "reference_id": "K3_MP3_MR6",
        "relaxed_id": "K3_MP3_MR8",
        "xlabel": r"Maximum segments per request, $M_{\mathrm{req}}$",
        "slug": "mreq",
        "caption": r"$M_{\mathrm{req}}$ sensitivity",
        "comparison": r"$2\rightarrow6\rightarrow8$",
    },
}

METRICS: Mapping[str, Mapping[str, str]] = {
    "rbp": {
        "mean": "RBP_mean",
        "ci": "RBP_ci95",
        "ylabel": "Request blocking probability",
        "caption": "request blocking probability",
    },
    "fallback_success": {
        "mean": "fallback_success_rate_pct_mean",
        "ci": "fallback_success_rate_pct_ci95",
        "ylabel": "Fallback success rate (%)",
        "caption": "fallback success rate",
    },
}


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


def locate_reporting_csv(results_dir: Path) -> Path:
    direct = results_dir / "sensitivity_plots_v33_pdf" / (
        "aca_sensitivity_v33_reporting_values.csv"
    )
    if direct.exists():
        return direct
    matches = sorted(results_dir.rglob("aca_sensitivity_v33_reporting_values.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one aca_sensitivity_v33_reporting_values.csv below "
            f"{results_dir}; found {len(matches)}. Run the v33 plotter first."
        )
    return matches[0]


def validate_reporting(frame: pd.DataFrame, load: int) -> None:
    required = {
        "scheme",
        "sensitivity_id",
        "sweep_membership",
        "K_max",
        "M_path",
        "M_req",
        "topology",
        "load",
        "RBP_mean",
        "RBP_ci95",
        "fallback_success_rate_pct_mean",
        "fallback_success_rate_pct_ci95",
        "slot_hops_per_accepted_gbps_mean",
        "mean_decision_time_us_mean",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Reporting CSV is missing required columns: {missing}")

    aca = frame[frame["scheme"] == "aca_rsa"]
    if aca["sensitivity_id"].nunique() != 10:
        raise ValueError("Expected exactly ten ACA-RSA sensitivity configurations.")
    if load not in set(aca["load"].astype(int)):
        raise ValueError(f"Requested load rho={load} is absent from the reporting CSV.")
    if not required - {
        "fallback_success_rate_pct_mean",
        "fallback_success_rate_pct_ci95",
    } <= set(frame.columns):
        raise ValueError("The v33 reporting CSV is incomplete.")
    if aca["fallback_success_rate_pct_mean"].isna().any():
        raise ValueError(
            "Fallback diagnostics are missing. Re-run the v33 plotter without "
            "--skip-debug before generating bar plots."
        )


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


def plot_bar_panel(
    frame: pd.DataFrame,
    metric_slug: str,
    sweep_name: str,
    load: int,
    output_path: Path,
    image_format: str,
    dpi: int,
) -> None:
    metric = METRICS[metric_slug]
    sweep = SWEEPS[sweep_name]
    values = [int(value) for value in sweep["values"]]
    positions = np.arange(len(values), dtype=float)
    group_width = 0.78
    bar_width = group_width / len(TOPOLOGIES)

    fig, ax = plt.subplots(figsize=(5.4, 4.05))
    reference_index = values.index(int(sweep["reference"]))
    ax.axvspan(
        reference_index - 0.48,
        reference_index + 0.48,
        color="0.93",
        zorder=0,
    )

    for topology_index, topology in enumerate(TOPOLOGIES):
        rows = sweep_rows(frame, topology, load, sweep_name)
        observed = rows[str(sweep["column"])].astype(int).tolist()
        if observed != values:
            raise ValueError(
                f"Unexpected {sweep_name} values for {topology}, rho={load}: "
                f"{observed}"
            )
        offset = (topology_index - (len(TOPOLOGIES) - 1) / 2) * bar_width
        ax.bar(
            positions + offset,
            rows[metric["mean"]].astype(float),
            width=bar_width * 0.92,
            yerr=rows[metric["ci"]].astype(float),
            color=TOPOLOGY_COLORS[topology],
            edgecolor="black",
            linewidth=0.7,
            hatch=TOPOLOGY_HATCHES[topology],
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.0,
                "capsize": 2.5,
                "capthick": 1.0,
            },
            zorder=3,
        )

    legend_handles = [
        Patch(
            facecolor=TOPOLOGY_COLORS[topology],
            edgecolor="black",
            linewidth=0.7,
            hatch=TOPOLOGY_HATCHES[topology],
            label=TOPOLOGY_LABELS[topology],
        )
        for topology in TOPOLOGIES
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=9.5,
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        handlelength=1.6,
        columnspacing=0.9,
    )
    ax.set_xlabel(str(sweep["xlabel"]), fontsize=11)
    ax.set_ylabel(metric["ylabel"], fontsize=11)
    ax.set_xticks(positions, [str(value) for value in values])
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.025)
    if metric_slug == "rbp":
        ax.set_ylim(bottom=0.0)
    else:
        ax.set_ylim(0.0, 100.0)

    fig.tight_layout()
    fig.savefig(
        output_path.with_suffix(f".{image_format}"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def average_metric(
    frame: pd.DataFrame,
    sensitivity_id: str,
    load: int,
    column: str,
) -> float:
    rows = frame[
        (frame["scheme"] == "aca_rsa")
        & (frame["sensitivity_id"] == sensitivity_id)
        & (frame["load"].astype(int) == load)
    ]
    if set(rows["topology"].str.upper()) != set(TOPOLOGIES):
        raise ValueError(
            f"Configuration {sensitivity_id}, rho={load} lacks a topology row."
        )
    return float(rows[column].astype(float).mean())


def percent_reduction(before: float, after: float) -> float:
    return 100.0 * (before - after) / before


def percent_change(before: float, after: float) -> float:
    return 100.0 * (after - before) / before


def build_tradeoff_table(frame: pd.DataFrame, load: int) -> pd.DataFrame:
    rows = []
    for sweep_name, sweep in SWEEPS.items():
        strict_id = str(sweep["strict_id"])
        reference_id = str(sweep["reference_id"])
        relaxed_id = str(sweep["relaxed_id"])

        rbp_strict = average_metric(frame, strict_id, load, "RBP_mean")
        rbp_reference = average_metric(frame, reference_id, load, "RBP_mean")
        rbp_relaxed = average_metric(frame, relaxed_id, load, "RBP_mean")
        fallback_strict = average_metric(
            frame, strict_id, load, "fallback_success_rate_pct_mean"
        )
        fallback_reference = average_metric(
            frame, reference_id, load, "fallback_success_rate_pct_mean"
        )
        fallback_relaxed = average_metric(
            frame, relaxed_id, load, "fallback_success_rate_pct_mean"
        )
        slot_strict = average_metric(
            frame, strict_id, load, "slot_hops_per_accepted_gbps_mean"
        )
        slot_reference = average_metric(
            frame, reference_id, load, "slot_hops_per_accepted_gbps_mean"
        )
        slot_relaxed = average_metric(
            frame, relaxed_id, load, "slot_hops_per_accepted_gbps_mean"
        )
        time_strict = average_metric(
            frame, strict_id, load, "mean_decision_time_us_mean"
        )
        time_reference = average_metric(
            frame, reference_id, load, "mean_decision_time_us_mean"
        )
        time_relaxed = average_metric(
            frame, relaxed_id, load, "mean_decision_time_us_mean"
        )

        rows.extend(
            [
                {
                    "sweep": sweep_name,
                    "comparison": "strict_to_reference",
                    "parameter_transition": (
                        f"{int(sweep['values'][0])}->{int(sweep['reference'])}"
                    ),
                    "RBP_reduction_pct": percent_reduction(
                        rbp_strict, rbp_reference
                    ),
                    "fallback_gain_percentage_points": (
                        fallback_reference - fallback_strict
                    ),
                    "slot_hop_change_pct": percent_change(
                        slot_strict, slot_reference
                    ),
                    "decision_time_change_pct": percent_change(
                        time_strict, time_reference
                    ),
                },
                {
                    "sweep": sweep_name,
                    "comparison": "reference_to_relaxed",
                    "parameter_transition": (
                        f"{int(sweep['reference'])}->{int(sweep['values'][-1])}"
                    ),
                    "RBP_reduction_pct": percent_reduction(
                        rbp_reference, rbp_relaxed
                    ),
                    "fallback_gain_percentage_points": (
                        fallback_relaxed - fallback_reference
                    ),
                    "slot_hop_change_pct": percent_change(
                        slot_reference, slot_relaxed
                    ),
                    "decision_time_change_pct": percent_change(
                        time_reference, time_relaxed
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


def write_tradeoff_latex(table: pd.DataFrame, output_path: Path, load: int) -> None:
    sweep_labels: Dict[str, str] = {
        "K_max": r"$K_{\max}$",
        "M_path": r"$M_{\mathrm{path}}$",
        "M_req": r"$M_{\mathrm{req}}$",
    }
    transitions = {
        ("K_max", "strict_to_reference"): r"$1\rightarrow3$",
        ("K_max", "reference_to_relaxed"): r"$3\rightarrow4$",
        ("M_path", "strict_to_reference"): r"$1\rightarrow3$",
        ("M_path", "reference_to_relaxed"): r"$3\rightarrow4$",
        ("M_req", "strict_to_reference"): r"$2\rightarrow6$",
        ("M_req", "reference_to_relaxed"): r"$6\rightarrow8$",
    }
    lines = [
        "% Generated by plot_aca_sensitivity_v34_bar.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Topology-averaged ACA-RSA sensitivity trade-offs at "
        + rf"$\rho={load}$"
        + r". Fallback changes are percentage points.}",
        r"\label{tab:aca_sensitivity_tradeoff}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Bound & Change & RBP red. (\%) & Fallback gain & Slot-hop (\%) & Time (\%) \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        key = (str(row["sweep"]), str(row["comparison"]))
        lines.append(
            f"{sweep_labels[key[0]]} & {transitions[key]} & "
            f"{float(row['RBP_reduction_pct']):.2f} & "
            f"{float(row['fallback_gain_percentage_points']):+.2f} & "
            f"{float(row['slot_hop_change_pct']):+.2f} & "
            f"{float(row['decision_time_change_pct']):+.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_subfigure_latex(output_dir: Path, image_format: str, load: int) -> None:
    panel_order = [
        ("rbp", "K_max"),
        ("rbp", "M_path"),
        ("rbp", "M_req"),
        ("fallback_success", "K_max"),
        ("fallback_success", "M_path"),
        ("fallback_success", "M_req"),
    ]
    lines = [
        "% Generated by plot_aca_sensitivity_v34_bar.py",
        "% Requires: \\usepackage{subcaption}",
        "% Panel references:",
        "% Figs.~\\ref{fig:aca_sens_rbp_kmax}--\\ref{fig:aca_sens_rbp_mreq}",
        "% Figs.~\\ref{fig:aca_sens_fallback_kmax}--\\ref{fig:aca_sens_fallback_mreq}",
        "% Overall figure: Fig.~\\ref{fig:aca_bound_sensitivity}",
        "\\begin{figure*}[t]",
        "\\centering",
    ]
    for index, (metric_slug, sweep_name) in enumerate(panel_order):
        sweep = SWEEPS[sweep_name]
        metric = METRICS[metric_slug]
        filename = (
            f"bar_{metric_slug}_{sweep['slug']}_rho{load}.{image_format}"
        )
        label_metric = "fallback" if metric_slug == "fallback_success" else metric_slug
        panel_label = f"fig:aca_sens_{label_metric}_{sweep['slug']}"
        lines.extend(
            [
                "\\begin{subfigure}[t]{0.32\\textwidth}",
                "\\centering",
                "\\includegraphics[width=\\linewidth]{"
                + "sensitivity_bar_plots_v34_pdf/"
                + filename
                + "}",
                "\\caption{" + str(sweep["caption"]) + ": "
                + str(metric["caption"]) + ".}",
                "\\label{" + panel_label + "}",
                "\\end{subfigure}",
            ]
        )
        if index in {0, 1, 3, 4}:
            lines.append("\\hfill")
        elif index == 2:
            lines.extend(["", "\\vspace{1mm}", ""])
    lines.extend(
        [
            "\\caption{ACA-RSA bound sensitivity at "
            + rf"$\rho={load}$"
            + ". The shaded groups mark the selected reference values "
            + r"$K_{\max}=3$, $M_{\mathrm{path}}=3$, and "
            + r"$M_{\mathrm{req}}=6$. Error bars show 95\% Student-$t$ "
            + "confidence intervals over ten seeds.}",
            "\\label{fig:aca_bound_sensitivity}",
            "\\end{figure*}",
            "",
        ]
    )
    (output_dir / "aca_sensitivity_v34_bar_subfigures.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_combined_include(output_dir: Path) -> None:
    figure_path = output_dir / "aca_sensitivity_v34_bar_subfigures.tex"
    table_path = output_dir / "aca_sensitivity_v34_tradeoff_table.tex"
    combined = [
        "% Generated by plot_aca_sensitivity_v34_bar.py",
        "% Include from the manuscript root with:",
        "% \\input{sensitivity_bar_plots_v34_pdf/aca_sensitivity_v34_figure_and_table.tex}",
        "% The included figure panels reference PDFs in sensitivity_bar_plots_v34_pdf/.",
        "",
        figure_path.read_text(encoding="utf-8"),
        "",
        table_path.read_text(encoding="utf-8"),
    ]
    (output_dir / "aca_sensitivity_v34_figure_and_table.tex").write_text(
        "\n".join(combined), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grouped-bar ACA-RSA sensitivity plots."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Completed v33 sensitivity directory.",
    )
    parser.add_argument("--reporting-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--load", type=int, default=25)
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else locate_latest_results(script_dir)
    )
    reporting_csv = (
        args.reporting_csv.resolve()
        if args.reporting_csv is not None
        else locate_reporting_csv(results_dir)
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "sensitivity_bar_plots_v34_pdf"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reporting values: {reporting_csv}", flush=True)
    reporting = pd.read_csv(reporting_csv)
    validate_reporting(reporting, args.load)

    generated = 0
    for metric_slug in METRICS:
        for sweep_name, sweep in SWEEPS.items():
            output_base = output_dir / (
                f"bar_{metric_slug}_{sweep['slug']}_rho{args.load}"
            )
            plot_bar_panel(
                reporting,
                metric_slug,
                sweep_name,
                args.load,
                output_base,
                args.format,
                args.dpi,
            )
            generated += 1

    tradeoff = build_tradeoff_table(reporting, args.load)
    tradeoff.to_csv(
        output_dir / "aca_sensitivity_v34_tradeoff_table.csv", index=False
    )
    write_tradeoff_latex(
        tradeoff,
        output_dir / "aca_sensitivity_v34_tradeoff_table.tex",
        args.load,
    )
    write_subfigure_latex(output_dir, args.format, args.load)
    write_combined_include(output_dir)

    print(f"Generated {generated} grouped-bar panels in: {output_dir}")
    print("Generated LaTeX subfigure and trade-off table snippets.")


if __name__ == "__main__":
    main()
