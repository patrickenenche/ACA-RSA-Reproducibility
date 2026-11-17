import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PACKAGE_ROOT / "processed_results" / "v31" / "simulation_results_raw.csv"

ACA = "aca_rsa"
BASELINES = ("fsa_rsa_ssfm", "mp_ss_rsa")
METRICS = ("RBP", "BBP")
T_CRIT = 2.2621571628540993  # two-sided 95%, df=9


def holm_adjust(pvalues):
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [math.nan] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, (index, pvalue) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * pvalue)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


values = defaultdict(dict)
with CSV_PATH.open(newline="", encoding="utf-8-sig") as stream:
    for row in csv.DictReader(stream):
        scheme = row["scheme"]
        if scheme not in (ACA, *BASELINES):
            continue
        key = (row["topology"], int(row["load"]), scheme, int(row["seed"]))
        values[key] = {metric: float(row[metric]) for metric in METRICS}

tests = []
for topology in ("USNET", "GERMAN", "TI"):
    for load in (5, 10, 15, 20, 25):
        for metric in METRICS:
            for baseline in BASELINES:
                seeds = sorted(
                    seed
                    for t, rho, scheme, seed in values
                    if t == topology and rho == load and scheme == ACA
                )
                aca = [values[(topology, load, ACA, seed)][metric] for seed in seeds]
                base = [values[(topology, load, baseline, seed)][metric] for seed in seeds]
                differences = [b - a for a, b in zip(aca, base)]
                identical = all(abs(value) < 1e-15 for value in differences)
                if identical:
                    pvalue = math.nan
                    t_stat = math.nan
                else:
                    result = stats.ttest_rel(base, aca, alternative="two-sided")
                    pvalue = float(result.pvalue)
                    t_stat = float(result.statistic)
                tests.append(
                    {
                        "topology": topology,
                        "load": load,
                        "metric": metric,
                        "baseline": baseline,
                        "aca_mean": statistics.fmean(aca),
                        "aca_ci": T_CRIT * statistics.stdev(aca) / math.sqrt(len(aca)),
                        "baseline_mean": statistics.fmean(base),
                        "baseline_ci": T_CRIT * statistics.stdev(base) / math.sqrt(len(base)),
                        "mean_difference": statistics.fmean(differences),
                        "t_stat": t_stat,
                        "p_raw": pvalue,
                        "identical": identical,
                    }
                )

# One conservative Holm family containing all 60 main RBP/BBP tests.
# Exact equality is represented by p=1 because no difference is present.
nonidentical = [test for test in tests if not test["identical"]]
adjusted = holm_adjust([
    1.0 if test["identical"] else test["p_raw"]
    for test in tests
])
for test, p_adj in zip(tests, adjusted):
    test["p_adj_all"] = p_adj

print(f"total_tests={len(tests)}")
print(f"identical_tests={sum(test['identical'] for test in tests)}")
print(f"nonidentical_tests={len(nonidentical)}")
print(f"significant_holm_0.05={sum(test['p_adj_all'] < 0.05 for test in nonidentical)}")
print(f"significant_holm_0.001={sum(test['p_adj_all'] < 0.001 for test in nonidentical)}")
print(f"max_adjusted_p={max(test['p_adj_all'] for test in nonidentical):.12g}")
print("\nNONSIGNIFICANT_OR_IDENTICAL")
for test in tests:
    if test["identical"] or test["p_adj_all"] >= 0.05:
        print(
            test["topology"], test["load"], test["metric"], test["baseline"],
            "identical" if test["identical"] else f"p_adj={test['p_adj_all']:.6g}",
        )

print("\nADJUSTED_P_AT_LEAST_0.001")
for test in nonidentical:
    if test["p_adj_all"] >= 0.001:
        print(
            test["topology"], test["load"], test["metric"], test["baseline"],
            f"p_adj={test['p_adj_all']:.12g}",
        )

print("\nRHO25")
print(
    "topology,metric,baseline,aca_mean,aca_ci95,baseline_mean,baseline_ci95,"
    "mean_difference,t_stat,p_raw,p_adj_all"
)
for test in tests:
    if test["load"] != 25:
        continue
    print(
        f"{test['topology']},{test['metric']},{test['baseline']},"
        f"{test['aca_mean']:.10f},{test['aca_ci']:.10f},"
        f"{test['baseline_mean']:.10f},{test['baseline_ci']:.10f},"
        f"{test['mean_difference']:.10f},{test['t_stat']:.6f},"
        f"{test['p_raw']:.12g},{test['p_adj_all']:.12g}"
    )
