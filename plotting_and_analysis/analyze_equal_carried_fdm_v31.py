import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PACKAGE_ROOT / "processed_results" / "v31" / "simulation_results_raw.csv"

SCHEMES = ("aca_rsa", "fsa_rsa_ssfm", "mp_ss_rsa")
TARGET_GBPS = {"GERMAN": 20000.0, "USNET": 28000.0, "TI": 29000.0}
T_CRITICAL_95_DF9 = 2.262


def interpolate(points, target):
    points = sorted(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target <= x1:
            if x1 == x0:
                return y0
            weight = (target - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    raise ValueError(f"Target {target} is outside [{points[0][0]}, {points[-1][0]}]")


curves = defaultdict(list)
with CSV_PATH.open(newline="", encoding="utf-8-sig") as stream:
    for row in csv.DictReader(stream):
        if row["scheme"] not in SCHEMES:
            continue
        key = (row["topology"], row["scheme"], int(row["seed"]))
        curves[key].append((float(row["mean_active_carried_gbps"]), float(row["FDM"])))

print("topology,scheme,target_carried_gbps,n,fdm_mean,fdm_ci95")
for topology in ("USNET", "GERMAN", "TI"):
    target = TARGET_GBPS[topology]
    for scheme in SCHEMES:
        values = []
        seeds = sorted(seed for t, s, seed in curves if t == topology and s == scheme)
        for seed in seeds:
            values.append(interpolate(curves[(topology, scheme, seed)], target))
        mean = statistics.fmean(values)
        ci = T_CRITICAL_95_DF9 * statistics.stdev(values) / math.sqrt(len(values))
        print(f"{topology},{scheme},{target:.0f},{len(values)},{mean:.8f},{ci:.8f}")
