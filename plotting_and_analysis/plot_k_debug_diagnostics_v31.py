"""
Streaming fallback diagnostics for large reviewer-revision debug CSVs.

This script avoids loading the full debug CSV into memory. It aggregates rows
by scheme/topology/load and writes separate plots for ACA-RSA, A-MP-SS-RSA,
and P-ACA-RSA by default.

Example:
    python plot_k_debug_diagnostics_v31.py --csv path/to/k_debug.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


DEFAULT_SCHEMES = ["aca_rsa", "adaptive_mp_ss_rsa", "p_aca_rsa"]
LOAD_TICKS = [5, 10, 15, 20, 25]

SCHEME_DISPLAY_NAMES = {
    "aca_rsa": "ACA-RSA",
    "adaptive_mp_ss_rsa": "A-MP-SS-RSA",
    "adaptive_mp_ss_rsa_repetition": "A-MP-SS-RSA-Repetition",
    "ara_rsa": "ACA-RSA",
    "p_aca_rsa": "P-ACA-RSA",
}

TOPOLOGY_STYLES = {
    "USNET": {"marker": "o", "linestyle": "-"},
    "GERMAN": {"marker": "s", "linestyle": "--"},
    "TI": {"marker": "^", "linestyle": "-."},
}

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
HATCHES = ["", "//", "\\\\", "..", "xx", "++"]


def distribution_ylabel(field: str) -> str:
    if field == "K_effective_success":
        return "Share of successful fallback allocations (%)"
    return "Share of fallback attempts (%)"


def display_scheme(scheme: str) -> str:
    return SCHEME_DISPLAY_NAMES.get(str(scheme), str(scheme))


def load_checkpoint_scales(csv_path: Path, config_path: Path | None) -> Tuple[Path, Dict[str, float]]:
    if config_path is None:
        matches = sorted(csv_path.parent.glob("checkpoint_configuration_*.json"))
        if len(matches) != 1:
            raise ValueError(
                "Could not identify one checkpoint configuration JSON beside the debug CSV. "
                "Pass --config-json explicitly."
            )
        config_path = matches[0]

    with config_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    configuration = manifest.get("configuration", manifest)
    raw_scales = configuration.get("load_scale_by_topology")
    if not isinstance(raw_scales, dict) or not raw_scales:
        raise ValueError(f"No load_scale_by_topology mapping found in {config_path}")
    scales = {str(topology).upper(): float(scale) for topology, scale in raw_scales.items()}
    return config_path, scales


def aggregate_load(load: int, topology: str, load_scales: Dict[str, float]) -> float:
    key = topology.upper()
    if key not in load_scales:
        raise ValueError(f"No aggregate-load scale found for topology={topology}")
    return float(load) * load_scales[key]


def format_aggregate_load(value: float) -> str:
    rounded = math.floor(float(value) * 10.0 + 0.5) / 10.0
    return f"{rounded:.0f}" if rounded.is_integer() else f"{rounded:.1f}"


def panel_caption(index: int, text: str) -> str:
    prefix = PANEL_LABELS[index] if index < len(PANEL_LABELS) else f"({chr(97 + index)})"
    return f"{prefix} {text}"


def apply_publication_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


@dataclass
class DebugAggregate:
    attempts: int = 0
    successes: int = 0
    sum_k_selected: float = 0.0
    sum_k_effective: float = 0.0
    sum_wasted_path_ratio: float = 0.0
    wasted_path_ratio_count: int = 0
    sum_r: float = 0.0
    k_selected_counts: Counter = field(default_factory=Counter)
    k_effective_counts: Counter = field(default_factory=Counter)
    k_effective_success_counts: Counter = field(default_factory=Counter)

    def add(self, row: Dict[str, str]) -> None:
        success = as_int(row.get("success", "0"))
        k_selected = as_int(row.get("K_selected", "0"))
        k_effective = as_int(row.get("K_effective", "0"))

        self.attempts += 1
        self.successes += success
        self.sum_k_selected += as_float(row.get("K_selected", "0"))
        self.sum_k_effective += as_float(row.get("K_effective", "0"))
        raw_wasted_path_ratio = row.get("wasted_path_ratio", "")
        if k_selected > 0 and raw_wasted_path_ratio not in (None, ""):
            self.sum_wasted_path_ratio += as_float(raw_wasted_path_ratio)
            self.wasted_path_ratio_count += 1
        self.sum_r += as_float(row.get("r", "0"))
        self.k_selected_counts[k_selected] += 1
        self.k_effective_counts[k_effective] += 1
        if success == 1:
            self.k_effective_success_counts[k_effective] += 1

    def mean(self, value: float) -> float:
        return value / self.attempts if self.attempts else 0.0

    def success_rate_pct(self) -> float:
        return 100.0 * self.successes / self.attempts if self.attempts else 0.0

    def mean_wasted_path_ratio(self) -> float:
        return (
            self.sum_wasted_path_ratio / self.wasted_path_ratio_count
            if self.wasted_path_ratio_count
            else 0.0
        )


def pct_distribution(counts: Counter) -> Dict[int, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {int(k): 100.0 * v / total for k, v in sorted(counts.items())}


def aggregate_csv(
    csv_path: Path,
    schemes: Sequence[str],
) -> Dict[Tuple[str, str, int], DebugAggregate]:
    wanted = {scheme.lower() for scheme in schemes}
    aggregates: Dict[Tuple[str, str, int], DebugAggregate] = defaultdict(DebugAggregate)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scheme = row.get("scheme", "").lower()
            if wanted and scheme not in wanted:
                continue

            topology = row.get("topology", "UNKNOWN").upper()
            load = as_int(row.get("load", "0"))
            aggregates[(scheme, topology, load)].add(row)

    return aggregates


def values_for(
    aggregates: Dict[Tuple[str, str, int], DebugAggregate],
    scheme: str,
    topology: str,
    loads: Iterable[int],
    reducer,
) -> List[float]:
    return [
        reducer(aggregates.get((scheme, topology, load), DebugAggregate()))
        for load in loads
    ]


def line_metric_specs():
    return [
        ("Fallback Success Rate", "Fallback Success Rate", "%", lambda agg: agg.success_rate_pct()),
        (r"Average $K_{\mathrm{sel}}$", "Average selected path count", "", lambda agg: agg.mean(agg.sum_k_selected)),
        (r"Average $K_{\mathrm{eff}}$", "Average effective path count", "", lambda agg: agg.mean(agg.sum_k_effective)),
        (
            "Average NPR",
            "Average NPR",
            "",
            lambda agg: agg.mean_wasted_path_ratio(),
        ),
        ("Average Redundancy r", "Average Redundancy r", "slots", lambda agg: agg.mean(agg.sum_r)),
    ]


def plot_line_metrics_for_scheme(
    aggregates: Dict[Tuple[str, str, int], DebugAggregate],
    scheme: str,
    outpath: Path,
    load_scales: Dict[str, float],
) -> None:
    topologies = sorted({topology for s, topology, _ in aggregates if s == scheme})
    loads = sorted({load for s, _, load in aggregates if s == scheme})
    if not topologies or not loads:
        return

    metric_specs = line_metric_specs()

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
    axes_flat = axes.ravel()

    for idx, (ax, (ylabel, caption, suffix, reducer)) in enumerate(zip(axes_flat, metric_specs)):
        for topology in topologies:
            y = values_for(aggregates, scheme, topology, loads, reducer)
            x = loads
            style = TOPOLOGY_STYLES.get(topology, {"marker": "o", "linestyle": "-"})
            ax.plot(
                x,
                y,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                markersize=5,
                label=topology,
            )
        ax.set_xlabel(r"Nominal load index, $\rho$")
        ax.set_xticks(LOAD_TICKS)
        ax.set_xlim(LOAD_TICKS[0], LOAD_TICKS[-1])
        ax.set_ylabel(ylabel if not suffix else f"{ylabel} ({suffix})")
        apply_publication_axis_style(ax)
        ax.text(
            0.5,
            -0.31,
            panel_caption(idx, caption),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        ax.grid(alpha=0.25)

    axes_flat[-1].axis("off")

    # Figure title omitted for manuscript-ready plots.
    # fig.suptitle(f"K-Debug Line Diagnostics: {scheme}")
    fig.tight_layout(rect=(0, 0.10, 1, 0.98))
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_combined_line_metrics_for_scheme(
    aggregates: Dict[Tuple[str, str, int], DebugAggregate],
    scheme: str,
    outpath: Path,
) -> None:
    """Write the four-panel fallback-diagnostics figure used in the manuscript."""
    topologies = sorted({topology for s, topology, _ in aggregates if s == scheme})
    loads = sorted({load for s, _, load in aggregates if s == scheme})
    if not topologies or not loads:
        return

    metric_specs = line_metric_specs()[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))

    for idx, (ax, (ylabel, caption, suffix, reducer)) in enumerate(
        zip(axes.ravel(), metric_specs)
    ):
        for topology in topologies:
            y = values_for(aggregates, scheme, topology, loads, reducer)
            style = TOPOLOGY_STYLES.get(
                topology, {"marker": "o", "linestyle": "-"}
            )
            ax.plot(
                loads,
                y,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                markersize=5,
                label=topology,
            )

        ax.set_xlabel(r"Nominal load index, $\rho$")
        ax.set_xticks(LOAD_TICKS)
        ax.set_xlim(LOAD_TICKS[0], LOAD_TICKS[-1])
        ax.set_ylabel(ylabel if not suffix else f"{ylabel} ({suffix})")
        apply_publication_axis_style(ax)
        ax.text(
            0.5,
            -0.25,
            panel_caption(idx, caption),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_line_metric_all_topologies_for_scheme(
    aggregates: Dict[Tuple[str, str, int], DebugAggregate],
    scheme: str,
    metric_index: int,
    outpath: Path,
) -> None:
    """Write one manuscript subfigure containing all topology curves."""
    topologies = sorted({topology for s, topology, _ in aggregates if s == scheme})
    loads = sorted({load for s, _, load in aggregates if s == scheme})
    specs = line_metric_specs()
    if not topologies or not loads or metric_index >= len(specs):
        return

    ylabel, _caption, suffix, reducer = specs[metric_index]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))

    for topology in topologies:
        y = values_for(aggregates, scheme, topology, loads, reducer)
        style = TOPOLOGY_STYLES.get(
            topology, {"marker": "o", "linestyle": "-"}
        )
        ax.plot(
            loads,
            y,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            markersize=5,
            label=topology,
        )

    ax.set_xlabel(r"Nominal load index, $\rho$")
    ax.set_ylabel(ylabel if not suffix else f"{ylabel} ({suffix})")
    ax.set_xticks(LOAD_TICKS)
    ax.set_xlim(LOAD_TICKS[0], LOAD_TICKS[-1])
    apply_publication_axis_style(ax)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_line_metric_single_panel_for_scheme(
    aggregates: Dict[Tuple[str, str, int], DebugAggregate],
    scheme: str,
    topology: str,
    metric_index: int,
    outpath: Path,
    load_scales: Dict[str, float],
) -> None:
    loads = sorted({
        load
        for s, candidate_topology, load in aggregates
        if s == scheme and candidate_topology == topology
    })
    specs = line_metric_specs()
    if not loads or metric_index >= len(specs):
        return

    ylabel, _caption, suffix, reducer = specs[metric_index]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    y = values_for(aggregates, scheme, topology, loads, reducer)
    x = loads
    style = TOPOLOGY_STYLES.get(topology, {"marker": "o", "linestyle": "-"})
    ax.plot(
        x,
        y,
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2.5,
        markersize=5,
    )

    ax.set_xlabel(r"Nominal load index, $\rho$")
    ax.set_ylabel(ylabel if not suffix else f"{ylabel} ({suffix})")
    ax.set_xticks(LOAD_TICKS)
    ax.set_xlim(LOAD_TICKS[0], LOAD_TICKS[-1])
    apply_publication_axis_style(ax)
    ax.grid(alpha=0.25)

    # No in-figure title or subfigure label: LaTeX owns the caption.
    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_for_scheme(
    aggregates: Dict[Tuple[str, str, int], DebugAggregate],
    scheme: str,
    field: str,
    title: str,
    outpath: Path,
    load_scales: Dict[str, float],
) -> None:
    topologies = sorted({topology for s, topology, _ in aggregates if s == scheme})
    loads = sorted({load for s, _, load in aggregates if s == scheme})
    if not topologies or not loads:
        return

    if field == "K_selected":
        count_getter = lambda agg: agg.k_selected_counts
    elif field == "K_effective":
        count_getter = lambda agg: agg.k_effective_counts
    elif field == "K_effective_success":
        count_getter = lambda agg: agg.k_effective_success_counts
    else:
        raise ValueError(f"Unsupported field: {field}")

    categories = sorted({
        category
        for key, agg in aggregates.items()
        if key[0] == scheme
        for category in count_getter(agg)
    })
    if not categories:
        return

    fig, axes = plt.subplots(
        1,
        len(topologies),
        figsize=(max(6, 5.5 * len(topologies)), 4.2),
        sharey=True,
        squeeze=False,
    )

    colors = plt.cm.Set2.colors
    for panel_idx, (ax, topology) in enumerate(zip(list(axes[0]), topologies)):
        plot_loads = list(LOAD_TICKS)
        x_positions = list(range(len(plot_loads)))
        bottoms = [0.0] * len(plot_loads)
        for idx, category in enumerate(categories):
            vals = []
            for load in plot_loads:
                agg = aggregates.get((scheme, topology, load), DebugAggregate())
                dist = pct_distribution(count_getter(agg))
                vals.append(dist.get(category, 0.0))
            ax.bar(
                x_positions,
                vals,
                bottom=bottoms,
                label=f"K_eff={category}" if field == "K_effective_success" else f"{field}={category}",
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.7,
                hatch=HATCHES[idx % len(HATCHES)],
            )
            bottoms = [a + b for a, b in zip(bottoms, vals)]

        ax.set_xlabel(r"Nominal load index, $\rho$")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(plot_loads)
        apply_publication_axis_style(ax)
        ax.text(
            0.5,
            -0.27,
            panel_caption(panel_idx, topology),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
        ax.grid(axis="y", alpha=0.25)

    axes[0][0].set_ylabel(distribution_ylabel(field))
    handles, labels = axes[0][-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 4),
            fontsize=11,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    # Figure title omitted for manuscript-ready plots.
    # fig.suptitle(f"{title}: {scheme}")
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_single_topology_for_scheme(
    aggregates: Dict[Tuple[str, str, int], DebugAggregate],
    scheme: str,
    field: str,
    topology: str,
    outpath: Path,
    load_scales: Dict[str, float],
) -> None:
    loads = sorted({load for s, t, load in aggregates if s == scheme and t == topology})
    if not loads:
        return

    if field == "K_selected":
        count_getter = lambda agg: agg.k_selected_counts
    elif field == "K_effective":
        count_getter = lambda agg: agg.k_effective_counts
    elif field == "K_effective_success":
        count_getter = lambda agg: agg.k_effective_success_counts
    else:
        raise ValueError(f"Unsupported field: {field}")

    categories = sorted({
        category
        for key, agg in aggregates.items()
        if key[0] == scheme and key[1] == topology
        for category in count_getter(agg)
    })
    if not categories:
        return

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    colors = plt.cm.Set2.colors
    plot_loads = list(LOAD_TICKS)
    x_positions = list(range(len(plot_loads)))
    bottoms = [0.0] * len(plot_loads)

    for idx, category in enumerate(categories):
        vals = []
        for load in plot_loads:
            agg = aggregates.get((scheme, topology, load), DebugAggregate())
            dist = pct_distribution(count_getter(agg))
            vals.append(dist.get(category, 0.0))
        ax.bar(
            x_positions,
            vals,
            bottom=bottoms,
            label=f"K_eff={category}" if field == "K_effective_success" else f"{field}={category}",
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.7,
            hatch=HATCHES[idx % len(HATCHES)],
        )
        bottoms = [a + b for a, b in zip(bottoms, vals)]

    ax.set_xlabel(r"Nominal load index, $\rho$")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_loads)
    ax.set_ylabel(distribution_ylabel(field))
    apply_publication_axis_style(ax)
    ax.legend(
        loc="upper center",
        ncol=min(len(categories), 3),
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
    )
    ax.grid(axis="y", alpha=0.25)

    # No in-figure title or subfigure label: LaTeX owns the caption.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    aggregates: Dict[Tuple[str, str, int], DebugAggregate],
    output_path: Path,
    load_scales: Dict[str, float],
) -> None:
    fieldnames = [
        "scheme",
        "topology",
        "load",
        "aggregate_offered_load_erlangs",
        "attempts",
        "success_rate_pct",
        "avg_K_selected",
        "avg_K_effective",
        "avg_wasted_path_ratio",
        "avg_r",
        "K_selected_dist_pct",
        "K_effective_dist_pct",
        "K_effective_success_dist_pct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (scheme, topology, load), agg in sorted(aggregates.items()):
            writer.writerow({
                "scheme": scheme,
                "topology": topology,
                "load": load,
                "aggregate_offered_load_erlangs": aggregate_load(
                    load, topology, load_scales
                ),
                "attempts": agg.attempts,
                "success_rate_pct": agg.success_rate_pct(),
                "avg_K_selected": agg.mean(agg.sum_k_selected),
                "avg_K_effective": agg.mean(agg.sum_k_effective),
                "avg_wasted_path_ratio": agg.mean_wasted_path_ratio(),
                "avg_r": agg.mean(agg.sum_r),
                "K_selected_dist_pct": pct_distribution(agg.k_selected_counts),
                "K_effective_dist_pct": pct_distribution(agg.k_effective_counts),
                "K_effective_success_dist_pct": pct_distribution(agg.k_effective_success_counts),
            })


def safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in value)


def write_latex_subfigure_snippets(
    outdir: Path,
    schemes: Sequence[str],
    image_format: str,
) -> None:
    distribution_captions = {
        "k_selected_distribution": "Selected path count distribution",
        "k_effective_distribution": "Effective path count distribution",
        "k_effective_success_distribution": "Successful effective path count distribution",
    }
    topology_captions = {"GERMAN": "German", "TI": "TI", "USNET": "USNET"}

    lines = [
        "% LaTeX snippets generated for individually exported K-debug panels.",
        "% Add this package in the preamble if it is not already present:",
        "% \\usepackage[caption=false,font=footnotesize]{subfig}",
        "",
    ]

    for scheme in schemes:
        scheme_name = display_scheme(scheme)
        metric_files = [
            ("fallback_success_rate", "Fallback success rate"),
            ("average_k_selected", "Average selected path count"),
            ("average_k_effective", "Average effective path count"),
            ("average_non_contributing_path_ratio", "Average non-contributing path ratio"),
            ("average_redundancy_r", "Average redundancy r"),
        ]
        for file_stub, caption in metric_files:
            lines.extend([
                f"% {scheme_name}: {caption}",
                "\\begin{figure*}[t]",
                "\\centering",
            ])
            for topology in ["GERMAN", "TI", "USNET"]:
                filename = f"{scheme}_{file_stub}_{topology}.{image_format}"
                lines.extend([
                    f"\\subfloat[{topology_captions[topology]}]{{%",
                    f"\\includegraphics[width=0.32\\textwidth]{{{filename}}}%",
                    "}",
                ])
            lines.extend([
                f"\\caption{{{caption} for {scheme_name}.}}",
                f"\\label{{fig:{safe_filename(scheme)}_{file_stub}}}",
                "\\end{figure*}",
                "",
            ])

        for stub, caption in distribution_captions.items():
            lines.extend([
                f"% {scheme_name}: {caption}",
                "\\begin{figure*}[t]",
                "\\centering",
            ])
            for topology in ["GERMAN", "TI", "USNET"]:
                filename = f"{scheme}_{stub}_{topology}.{image_format}"
                lines.extend([
                    f"\\subfloat[{topology_captions[topology]}]{{%",
                    f"\\includegraphics[width=0.32\\textwidth]{{{filename}}}%",
                    "}",
                ])
            lines.extend([
                f"\\caption{{{caption} for {scheme_name}.}}",
                f"\\label{{fig:{safe_filename(scheme)}_{stub}}}",
                "\\end{figure*}",
                "",
            ])

    (outdir / "latex_k_debug_subfigure_snippets.tex").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="Path to k_debug CSV")
    parser.add_argument("--outdir", type=Path, default=None, help="Output plot directory")
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help=(
            "Checkpoint configuration JSON containing load_scale_by_topology. "
            "Defaults to the unique checkpoint_configuration_*.json beside the CSV."
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Figure format. Default: pdf.",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=DEFAULT_SCHEMES,
        help="Schemes to plot separately. Default: aca_rsa adaptive_mp_ss_rsa p_aca_rsa",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, load_scales = load_checkpoint_scales(args.csv, args.config_json)
    outdir = args.outdir or (args.csv.parent / "k_debug_plots_v31")
    outdir.mkdir(parents=True, exist_ok=True)

    schemes = [scheme.lower() for scheme in args.schemes]
    aggregates = aggregate_csv(args.csv, schemes)
    if not aggregates:
        raise SystemExit(f"No matching rows found in {args.csv} for schemes: {', '.join(schemes)}")

    write_summary_csv(
        aggregates,
        outdir / "k_debug_summary_by_scheme_v31.csv",
        load_scales,
    )

    for scheme in schemes:
        if not any(key[0] == scheme for key in aggregates):
            print(f"No rows found for scheme={scheme}; skipping.")
            continue

        metric_files = [
            "fallback_success_rate",
            "average_k_selected",
            "average_k_effective",
            "average_non_contributing_path_ratio",
            "average_redundancy_r",
        ]
        for metric_index, file_stub in enumerate(metric_files[:4]):
            plot_line_metric_all_topologies_for_scheme(
                aggregates,
                scheme,
                metric_index,
                outdir / f"{scheme}_{file_stub}.{args.image_format}",
            )

        plot_combined_line_metrics_for_scheme(
            aggregates,
            scheme,
            outdir / f"{scheme}_k_debug_line_metrics.{args.image_format}",
        )

        for metric_index, file_stub in enumerate(metric_files):
            for topology in ["GERMAN", "TI", "USNET"]:
                plot_line_metric_single_panel_for_scheme(
                    aggregates,
                    scheme,
                    topology,
                    metric_index,
                    outdir / f"{scheme}_{file_stub}_{topology}.{args.image_format}",
                    load_scales,
                )

        for field, file_stub in [
            ("K_selected", "k_selected_distribution"),
            ("K_effective", "k_effective_distribution"),
            ("K_effective_success", "k_effective_success_distribution"),
        ]:
            for topology in ["GERMAN", "TI", "USNET"]:
                plot_distribution_single_topology_for_scheme(
                    aggregates,
                    scheme,
                    field,
                    topology,
                    outdir / f"{scheme}_{file_stub}_{topology}.{args.image_format}",
                    load_scales,
                )

    write_latex_subfigure_snippets(outdir, schemes, args.image_format)
    print(f"Checkpoint configuration read from: {config_path}")
    print(f"Wrote v31 fallback diagnostics to: {outdir}")


if __name__ == "__main__":
    main()






