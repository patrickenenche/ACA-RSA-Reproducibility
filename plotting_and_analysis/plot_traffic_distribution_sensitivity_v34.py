"""Analyze and plot the v34 mean-matched traffic-distribution experiment.

The script validates the completed run matrix and traffic-trace pairing,
summarizes the rho=25 sensitivity results, performs seed-paired t-tests with
Holm correction, and creates compact publication outputs.  The large fallback
debug CSV is processed in chunks so it is never loaded fully into memory.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats


PROFILE_ORDER = ["constant_10", "uniform_4_16", "bimodal_4_16"]
PROFILE_LABELS = {
    "constant_10": "Constant 10",
    "uniform_4_16": "Uniform 4--16",
    "bimodal_4_16": "Bimodal 4/16",
}
PROFILE_VARIANCES = {
    "constant_10": 0.0,
    "uniform_4_16": 14.0,
    "bimodal_4_16": 36.0,
}

TOPOLOGY_ORDER = ["USNET", "GERMAN", "TI"]
TOPOLOGY_LABELS = {"USNET": "USNET", "GERMAN": "German", "TI": "TI"}
TOPOLOGY_COLORS = {
    "USNET": "#0072B2",
    "GERMAN": "#D55E00",
    "TI": "#009E73",
}
TOPOLOGY_HATCHES = {"USNET": "///", "GERMAN": "\\\\", "TI": "xx"}

SCHEMES = ["aca_rsa", "fsa_rsa_ssfm", "mp_ss_rsa"]
BASELINES = ["fsa_rsa_ssfm", "mp_ss_rsa"]
SCHEME_LABELS = {
    "aca_rsa": "ACA-RSA",
    "fsa_rsa_ssfm": "FSA-RSA(SSFM)",
    "mp_ss_rsa": "MP-SS-RSA",
}

LOADS = [10, 20, 25]
EXPECTED_SEEDS = list(range(1, 11))
STRESS_LOAD = 25


def long_path(path: Path) -> str:
    """Return a Windows extended path when the ordinary path is long."""
    value = str(path.resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def locate_latest_results(search_root: Path) -> Path:
    matches = sorted(
        search_root.glob(
            "results_ACA_RSA_v34_traffic_distribution_sensitivity_full_*"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No completed v34 traffic-sensitivity directory found in {search_root}."
        )
    return matches[0]


def locate_one(results_dir: Path, pattern: str) -> Path:
    # Path.is_file() can return False for valid paths near Windows MAX_PATH.
    # The directory contains only result files for these specific patterns.
    matches = list(results_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {results_dir}; "
            f"found {len(matches)}."
        )
    return matches[0]


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(long_path(path), **kwargs)


def validate_matrix(raw: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    expected_raw = len(SCHEMES) * len(PROFILE_ORDER) * len(TOPOLOGY_ORDER) * len(LOADS) * len(EXPECTED_SEEDS)
    expected_summary = len(SCHEMES) * len(PROFILE_ORDER) * len(TOPOLOGY_ORDER) * len(LOADS)
    messages: list[str] = []

    if len(raw) != expected_raw:
        raise ValueError(f"Expected {expected_raw} raw rows; found {len(raw)}.")
    if len(summary) != expected_summary:
        raise ValueError(
            f"Expected {expected_summary} summary rows; found {len(summary)}."
        )

    dimensions = {
        "scheme": (set(raw["scheme"]), set(SCHEMES)),
        "traffic_profile_id": (
            set(raw["traffic_profile_id"]),
            set(PROFILE_ORDER),
        ),
        "topology": (set(raw["topology"]), set(TOPOLOGY_ORDER)),
        "load": (set(raw["load"].astype(int)), set(LOADS)),
        "seed": (set(raw["seed"].astype(int)), set(EXPECTED_SEEDS)),
    }
    for name, (observed, expected) in dimensions.items():
        if observed != expected:
            raise ValueError(
                f"Unexpected {name} values: observed={sorted(observed)}, "
                f"expected={sorted(expected)}."
            )

    duplicated = raw.duplicated(
        ["scheme", "traffic_profile_id", "topology", "load", "seed"]
    ).sum()
    if duplicated:
        raise ValueError(f"Found {duplicated} duplicate raw run keys.")

    trace_groups = raw.groupby(
        ["traffic_profile_id", "topology", "load", "seed"]
    )["traffic_trace_sha256"].nunique()
    if not (trace_groups == 1).all():
        raise ValueError(
            "Complete traffic-trace hashes do not match across schemes."
        )

    non_demand_groups = raw.groupby(["topology", "load", "seed"])[
        "non_demand_trace_sha256"
    ].nunique()
    if not (non_demand_groups == 1).all():
        raise ValueError(
            "Non-demand trace hashes do not match across traffic profiles."
        )

    messages.extend(
        [
            f"Raw run matrix: {len(raw)}/{expected_raw} rows complete.",
            f"Summary matrix: {len(summary)}/{expected_summary} rows complete.",
            f"Complete traffic-trace pairing: {len(trace_groups)} groups verified.",
            f"Cross-profile non-demand pairing: {len(non_demand_groups)} groups verified.",
        ]
    )
    return messages


def t_ci95(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array))
    if len(array) <= 1:
        return mean, 0.0, 0.0
    sem = float(stats.sem(array))
    half_width = float(stats.t.ppf(0.975, len(array) - 1) * sem)
    return mean, float(np.std(array, ddof=1)), half_width


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [math.nan] * len(values)
    running_max = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def paired_tests(raw: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for baseline in BASELINES:
        for metric in ["RBP", "BBP"]:
            family_indices: list[int] = []
            family_p_values: list[float] = []
            for profile in PROFILE_ORDER:
                for topology in TOPOLOGY_ORDER:
                    for load in LOADS:
                        selected = raw[
                            (raw["traffic_profile_id"] == profile)
                            & (raw["topology"] == topology)
                            & (raw["load"].astype(int) == load)
                            & raw["scheme"].isin(["aca_rsa", baseline])
                        ]
                        paired = selected.pivot(
                            index="seed", columns="scheme", values=metric
                        ).sort_index()
                        if list(paired.index.astype(int)) != EXPECTED_SEEDS:
                            raise ValueError(
                                f"Incomplete seed pairing for {profile}, {topology}, "
                                f"rho={load}, {baseline}, {metric}."
                            )
                        differences = paired[baseline] - paired["aca_rsa"]
                        test = stats.ttest_rel(
                            paired[baseline], paired["aca_rsa"], nan_policy="raise"
                        )
                        mean_difference, difference_std, difference_ci95 = t_ci95(
                            differences
                        )
                        baseline_mean = float(paired[baseline].mean())
                        aca_mean = float(paired["aca_rsa"].mean())
                        record = {
                            "baseline": baseline,
                            "baseline_label": SCHEME_LABELS[baseline],
                            "metric": metric,
                            "traffic_profile_id": profile,
                            "topology": topology,
                            "load": load,
                            "num_seed_pairs": len(paired),
                            "baseline_mean": baseline_mean,
                            "aca_mean": aca_mean,
                            "mean_paired_reduction": mean_difference,
                            "paired_reduction_std": difference_std,
                            "paired_reduction_ci95": difference_ci95,
                            "relative_reduction_pct": (
                                100.0 * mean_difference / baseline_mean
                                if baseline_mean > 0
                                else math.nan
                            ),
                            "paired_t_statistic": float(test.statistic),
                            "paired_p_value": float(test.pvalue),
                            "holm_adjusted_p_value": math.nan,
                        }
                        records.append(record)
                        family_indices.append(len(records) - 1)
                        family_p_values.append(float(test.pvalue))

            adjusted = holm_adjust(family_p_values)
            for index, adjusted_p in zip(family_indices, adjusted):
                records[index]["holm_adjusted_p_value"] = adjusted_p

    return pd.DataFrame.from_records(records)


def aggregate_debug(debug_path: Path) -> pd.DataFrame:
    use_columns = [
        "traffic_profile_id",
        "topology",
        "seed",
        "load",
        "success",
        "K_selected",
        "K_effective",
        "NPR",
    ]
    partials: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        long_path(debug_path), usecols=use_columns, chunksize=500_000
    ):
        partial = chunk.groupby(
            ["traffic_profile_id", "topology", "seed", "load"]
        ).agg(
            fallback_attempts=("success", "size"),
            fallback_successes=("success", "sum"),
            k_selected_sum=("K_selected", "sum"),
            k_effective_sum=("K_effective", "sum"),
            npr_sum=("NPR", "sum"),
        )
        partials.append(partial)

    combined = pd.concat(partials).groupby(level=[0, 1, 2, 3]).sum()
    combined["fallback_success_rate"] = (
        combined["fallback_successes"] / combined["fallback_attempts"]
    )
    combined["avg_K_selected_per_attempt"] = (
        combined["k_selected_sum"] / combined["fallback_attempts"]
    )
    combined["avg_K_effective_per_success"] = (
        combined["k_effective_sum"]
        / combined["fallback_successes"].replace(0, np.nan)
    )
    combined["avg_NPR_per_attempt"] = (
        combined["npr_sum"] / combined["fallback_attempts"]
    )
    return combined.reset_index()


def compact_reporting_table(
    summary: pd.DataFrame, debug: pd.DataFrame
) -> pd.DataFrame:
    selected = summary[summary["load"].astype(int) == STRESS_LOAD].copy()
    records: list[dict[str, object]] = []
    for profile in PROFILE_ORDER:
        profile_rows = selected[selected["traffic_profile_id"] == profile]
        scheme_means = profile_rows.groupby("scheme")[["RBP_mean", "BBP_mean"]].mean()
        aca_rbp = float(scheme_means.loc["aca_rsa", "RBP_mean"])
        aca_bbp = float(scheme_means.loc["aca_rsa", "BBP_mean"])
        fsa_rbp = float(scheme_means.loc["fsa_rsa_ssfm", "RBP_mean"])
        fsa_bbp = float(scheme_means.loc["fsa_rsa_ssfm", "BBP_mean"])
        mp_rbp = float(scheme_means.loc["mp_ss_rsa", "RBP_mean"])
        mp_bbp = float(scheme_means.loc["mp_ss_rsa", "BBP_mean"])

        debug_rows = debug[
            (debug["traffic_profile_id"] == profile)
            & (debug["load"].astype(int) == STRESS_LOAD)
        ]
        fallback_mean, fallback_std, fallback_ci95 = t_ci95(
            debug_rows["fallback_success_rate"]
        )

        records.append(
            {
                "traffic_profile_id": profile,
                "traffic_profile": PROFILE_LABELS[profile],
                "mean_slots": 10.0,
                "variance_slots2": PROFILE_VARIANCES[profile],
                "aca_rbp_mean_pct": 100.0 * aca_rbp,
                "aca_bbp_mean_pct": 100.0 * aca_bbp,
                "fallback_success_mean_pct": 100.0 * fallback_mean,
                "fallback_success_std_pct": 100.0 * fallback_std,
                "fallback_success_ci95_pct": 100.0 * fallback_ci95,
                "rbp_reduction_vs_fsa_pct": 100.0 * (fsa_rbp - aca_rbp) / fsa_rbp,
                "bbp_reduction_vs_fsa_pct": 100.0 * (fsa_bbp - aca_bbp) / fsa_bbp,
                "rbp_reduction_vs_mp_ss_pct": 100.0 * (mp_rbp - aca_rbp) / mp_rbp,
                "bbp_reduction_vs_mp_ss_pct": 100.0 * (mp_bbp - aca_bbp) / mp_bbp,
                "aca_slot_hops_per_accepted_gbps": float(
                    profile_rows[profile_rows["scheme"] == "aca_rsa"][
                        "slot_hops_per_accepted_gbps_mean"
                    ].mean()
                ),
                "aca_mean_decision_time_ms": float(
                    profile_rows[profile_rows["scheme"] == "aca_rsa"][
                        "mean_decision_time_us_mean"
                    ].mean()
                    / 1000.0
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def write_latex_table(reporting: pd.DataFrame, output_path: Path) -> None:
    rows: list[str] = []
    for _, row in reporting.iterrows():
        rows.append(
            "{} & {:.0f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.4f} & {:.2f} \\\\".format(
                row["traffic_profile"],
                row["variance_slots2"],
                row["aca_rbp_mean_pct"],
                row["aca_bbp_mean_pct"],
                row["fallback_success_mean_pct"],
                row["rbp_reduction_vs_fsa_pct"],
                row["bbp_reduction_vs_fsa_pct"],
                row["rbp_reduction_vs_mp_ss_pct"],
                row["aca_slot_hops_per_accepted_gbps"],
                row["aca_mean_decision_time_ms"],
            )
        )

    content = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Mean-matched traffic-distribution sensitivity at "
            r"$\rho=25$, averaged across the German, USNET, and TI "
            r"topologies. All profiles have a mean demand of ten slots. "
            r"Positive reductions indicate lower blocking under \acs{aca}; "
            r"slot-hop cost and decision time are likewise topology-averaged.}",
            r"\label{tab:traffic_distribution_sensitivity}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrrrr}",
            r"\toprule",
            r"\textbf{Traffic profile} & \textbf{Variance} & "
            r"\textbf{\acs{aca} \acs{rbp} (\%)} & "
            r"\textbf{\acs{aca} \acs{bbp} (\%)} & "
            r"\textbf{Fallback success (\%)} & "
            r"\textbf{\acs{rbp} red. vs. \acs{fsa}(\acs{ssfm}) (\%)} & "
            r"\textbf{\acs{bbp} red. vs. \acs{fsa}(\acs{ssfm}) (\%)} & "
            r"\textbf{\acs{rbp} red. vs. \acs{mpss} (\%)} & "
            r"\textbf{Slot-hops/accepted Gbps} & "
            r"\textbf{Decision time (ms)} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    with open(long_path(output_path), "w", encoding="ascii", newline="") as handle:
        handle.write(content)


def plot_rbp(summary: pd.DataFrame, output_path: Path, dpi: int) -> None:
    selected = summary[
        (summary["scheme"] == "aca_rsa")
        & (summary["load"].astype(int) == STRESS_LOAD)
    ].copy()
    positions = np.arange(len(PROFILE_ORDER), dtype=float)
    group_width = 0.78
    bar_width = group_width / len(TOPOLOGY_ORDER)

    fig, ax = plt.subplots(figsize=(5.4, 3.75))
    for topology_index, topology in enumerate(TOPOLOGY_ORDER):
        rows = (
            selected[selected["topology"] == topology]
            .set_index("traffic_profile_id")
            .loc[PROFILE_ORDER]
        )
        offset = (
            topology_index - (len(TOPOLOGY_ORDER) - 1) / 2
        ) * bar_width
        ax.bar(
            positions + offset,
            rows["RBP_mean"].astype(float),
            width=bar_width * 0.92,
            yerr=rows["RBP_ci95"].astype(float),
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

    handles = [
        Patch(
            facecolor=TOPOLOGY_COLORS[topology],
            edgecolor="black",
            linewidth=0.7,
            hatch=TOPOLOGY_HATCHES[topology],
            label=TOPOLOGY_LABELS[topology],
        )
        for topology in TOPOLOGY_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        fontsize=10,
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        handlelength=1.5,
        columnspacing=0.8,
    )
    ax.set_xlabel("Mean-matched traffic profile", fontsize=11)
    ax.set_ylabel("Request blocking probability", fontsize=11)
    ax.set_xticks(positions, [PROFILE_LABELS[profile] for profile in PROFILE_ORDER])
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout(pad=0.65)
    fig.savefig(long_path(output_path), format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_validation_report(
    output_path: Path,
    messages: list[str],
    tests: pd.DataFrame,
    reporting: pd.DataFrame,
) -> None:
    lines = ["ACA-RSA v34 traffic-distribution sensitivity validation", ""]
    lines.extend(messages)
    lines.append("")
    for baseline in BASELINES:
        for metric in ["RBP", "BBP"]:
            family = tests[
                (tests["baseline"] == baseline) & (tests["metric"] == metric)
            ]
            lines.append(
                f"{metric}: ACA-RSA versus {SCHEME_LABELS[baseline]} -- "
                f"{int((family['mean_paired_reduction'] > 0).sum())}/{len(family)} "
                f"positive paired reductions; "
                f"{int((family['holm_adjusted_p_value'] < 0.05).sum())}/{len(family)} "
                f"Holm-significant; maximum adjusted p="
                f"{family['holm_adjusted_p_value'].max():.6g}."
            )
    lines.append("")
    lines.append("Compact rho=25 reporting values (topology-averaged):")
    lines.append(reporting.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    with open(long_path(output_path), "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the ACA-RSA v34 traffic-distribution sensitivity run."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Completed v34 results directory. Defaults to the latest below --search-root.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory searched when --results-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to a new folder inside the results directory.",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        help="Explicit seed-level raw CSV (useful for a compact package).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Explicit summary CSV (useful for a compact package).",
    )
    parser.add_argument(
        "--debug-summary-csv",
        type=Path,
        help=(
            "Pre-aggregated seed-level fallback table. When supplied, the "
            "multi-gigabyte event-level debug CSV is not required."
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else locate_latest_results(args.search_root.resolve())
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else results_dir / "traffic_distribution_sensitivity_v34_analysis"
    )
    os.makedirs(long_path(output_dir), exist_ok=True)

    raw_path = (
        args.raw_csv.resolve()
        if args.raw_csv
        else locate_one(results_dir, "simulation_results_raw_*req_10seeds.csv")
    )
    summary_path = (
        args.summary_csv.resolve()
        if args.summary_csv
        else locate_one(results_dir, "simulation_results_summary_*req_10seeds.csv")
    )

    raw = read_csv(raw_path)
    summary = read_csv(summary_path)
    messages = validate_matrix(raw, summary)
    tests = paired_tests(raw)
    if args.debug_summary_csv:
        debug = read_csv(args.debug_summary_csv.resolve())
    else:
        debug_path = locate_one(
            results_dir, "checkpoint_k_debug_per_seed_*req_10seeds.csv"
        )
        debug = aggregate_debug(debug_path)
    reporting = compact_reporting_table(summary, debug)

    reporting.to_csv(
        long_path(
            output_dir / "traffic_distribution_sensitivity_v34_compact_table.csv"
        ),
        index=False,
    )
    tests.to_csv(
        long_path(output_dir / "traffic_distribution_sensitivity_v34_paired_tests.csv"),
        index=False,
    )
    debug.to_csv(
        long_path(
            output_dir / "traffic_distribution_sensitivity_v34_fallback_by_seed.csv"
        ),
        index=False,
    )
    write_latex_table(
        reporting,
        output_dir / "traffic_distribution_sensitivity_v34_table.tex",
    )
    plot_rbp(
        summary,
        output_dir / "traffic_distribution_sensitivity_v34_rbp_rho25.pdf",
        args.dpi,
    )
    write_validation_report(
        output_dir / "traffic_distribution_sensitivity_v34_validation.txt",
        messages,
        tests,
        reporting,
    )

    print(f"Validated results: {results_dir}")
    for message in messages:
        print(f"  {message}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
