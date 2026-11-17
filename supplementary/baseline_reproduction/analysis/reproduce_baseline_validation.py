#!/usr/bin/env python3
"""Regenerate the ACA-RSA baseline-reproduction comparison tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple


LOADS = (5, 10, 15, 20, 25)
TOPOLOGIES = ("GERMAN", "USNET", "TI")
SCHEMES = ("k1_rsa", "k3_rsa", "fsa_rsa_ssfm")
METRICS = ("RBP_mean", "FSU_mean", "FDM_mean", "throughput_mean")

DISPLAY_SCHEME = {
    "k1_rsa": "K1-RSA",
    "k3_rsa": "K3-RSA",
    "fsa_rsa_ssfm": "FSA-RSA(SSFM)",
}

PUBLISHED_LOW_LOAD = {
    ("k1_rsa", "GERMAN"): 0.00042,
    ("k1_rsa", "USNET"): 0.00077,
    ("k1_rsa", "TI"): 0.050,
    ("k3_rsa", "GERMAN"): 0.00015,
    ("k3_rsa", "USNET"): 0.00019,
    ("k3_rsa", "TI"): 0.006,
    ("fsa_rsa_ssfm", "GERMAN"): 0.0,
    ("fsa_rsa_ssfm", "USNET"): 0.0,
    ("fsa_rsa_ssfm", "TI"): 0.00065,
}

PUBLISHED_TABLE5 = {
    ("GERMAN", "k1_rsa"): (-48.3, 3.0, 5.1, 4.3),
    ("GERMAN", "k3_rsa"): (-31.7, -3.9, -1.2, 0.3),
    ("USNET", "k1_rsa"): (-52.7, 6.1, 8.2, 5.4),
    ("USNET", "k3_rsa"): (-33.1, -2.0, 0.47, 0.4),
    ("TI", "k1_rsa"): (-50.0, 14.0, 11.7, 17.2),
    ("TI", "k3_rsa"): (-25.0, -5.0, -4.7, 2.2),
}


def read_rows(path: Path) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    rows: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            key = (raw["scheme"], raw["topology"], int(float(raw["load"])))
            rows[key] = {metric: float(raw[metric]) for metric in METRICS}

    expected = {
        (scheme, topology, load)
        for scheme in SCHEMES
        for topology in TOPOLOGIES
        for load in LOADS
    }
    missing = sorted(expected - set(rows))
    if missing:
        raise ValueError(f"Missing required rows: {missing}")
    return rows


def source_style_change(
    rows: Dict[Tuple[str, str, int], Dict[str, float]],
    topology: str,
    baseline: str,
    metric: str,
) -> float:
    changes = []
    for load in LOADS:
        proposed = rows[("fsa_rsa_ssfm", topology, load)][metric]
        reference = rows[(baseline, topology, load)][metric]
        if reference == 0:
            raise ZeroDivisionError(
                f"Zero denominator for {topology}/{baseline}/{load}/{metric}"
            )
        changes.append(100.0 * (proposed - reference) / reference)
    return sum(changes) / len(changes)


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=package_root / "data" / "v31_baseline_summary_subset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_root / "generated",
    )
    args = parser.parse_args()

    rows = read_rows(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    low_load_rows = []
    for scheme in SCHEMES:
        for topology in TOPOLOGIES:
            published = PUBLISHED_LOW_LOAD[(scheme, topology)]
            reproduced = rows[(scheme, topology, 5)]["RBP_mean"]
            low_load_rows.append(
                {
                    "scheme": DISPLAY_SCHEME[scheme],
                    "topology": topology,
                    "published_RBP": f"{published:.10g}",
                    "reproduced_RBP": f"{reproduced:.10g}",
                    "absolute_difference": f"{abs(reproduced - published):.10g}",
                }
            )

    write_csv(
        output_dir / "baseline_low_load_validation.csv",
        (
            "scheme",
            "topology",
            "published_RBP",
            "reproduced_RBP",
            "absolute_difference",
        ),
        low_load_rows,
    )

    metric_labels = ("RBP", "FSU", "LFM_FDM", "Throughput")
    comparison_rows = []
    k1_differences = []
    for topology in TOPOLOGIES:
        for baseline in ("k1_rsa", "k3_rsa"):
            published_values = PUBLISHED_TABLE5[(topology, baseline)]
            reproduced_values = tuple(
                source_style_change(rows, topology, baseline, metric)
                for metric in METRICS
            )
            row = {
                "topology": topology,
                "baseline": DISPLAY_SCHEME[baseline],
            }
            for label, published, reproduced in zip(
                metric_labels, published_values, reproduced_values
            ):
                difference = reproduced - published
                row[f"published_{label}"] = f"{published:.2f}"
                row[f"reproduced_{label}"] = f"{reproduced:.2f}"
                row[f"difference_pp_{label}"] = f"{difference:.2f}"
                if baseline == "k1_rsa":
                    k1_differences.append(abs(difference))
            comparison_rows.append(row)

    table5_fields = ["topology", "baseline"]
    for label in metric_labels:
        table5_fields.extend(
            (
                f"published_{label}",
                f"reproduced_{label}",
                f"difference_pp_{label}",
            )
        )
    write_csv(
        output_dir / "baseline_table5_comparison.csv",
        table5_fields,
        comparison_rows,
    )

    max_k1_difference = max(k1_differences)
    summary = (
        "Baseline reproduction summary\n"
        "=============================\n"
        f"Input: {args.input}\n"
        f"Rows validated: {len(rows)}\n"
        "Source-style calculation: mean of five load-specific signed changes\n"
        f"Maximum K1 published/reproduced difference: "
        f"{max_k1_difference:.2f} percentage points\n"
        "Expected manuscript statement: at most 1.33 percentage points\n"
    )
    (output_dir / "validation_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()

