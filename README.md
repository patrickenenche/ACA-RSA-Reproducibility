# ACA-RSA Reproducibility Package

This package accompanies the ACA-RSA study on bounded fragmented-spectrum
aggregation in elastic optical networks. It contains the four final simulator
versions, plotting and statistical-analysis scripts, exact topology data,
configuration manifests, compact seed-level results, and manuscript-ready
tables, figures, and supplementary baseline-reproduction materials.

Repository: https://github.com/patrickenenche/ACA-RSA-Reproducibility

## Package layout

```text
simulators/
  v31_main.py                 Principal zero-redundancy experiment
  v32_reliability.py          Equal-redundancy single-erasure experiment
  v33_bound_sensitivity.py    Path/segment-bound sensitivity
  v34_traffic_sensitivity.py  Traffic-distribution sensitivity
plotting_and_analysis/        Figure, table, and topology scripts
topologies/                   Machine-readable topologies and rendered figures
configurations/               Run manifests and sensitivity definitions
processed_results/            Compact raw/summary CSVs and final outputs
supplementary/                Baseline-reproduction supplement and evidence
requirements.txt              Pinned analysis environment
MANIFEST_SHA256.txt           Package integrity checksums
```

The four simulator files were renamed for clarity only. Their contents are
unchanged from the executed v31-v34 sources, so their hashes can be compared
with `source_sha256` in the corresponding configuration JSON files.

## Reference execution environments

All paper-scale experiments used 100,000 requests and ten seeds.

- Principal, bound-sensitivity, and traffic-distribution experiments (v31,
  v33, and v34): Windows 11 Education, Python 3.13.14, Intel Core i7-13700
  (16 cores / 24 logical processors), 16 GB RAM, 24 workers, and an external
  SSD. No other compute-intensive workload was active.
- Equal-redundancy reliability experiment (v32): Windows 11 Pro, Python
  3.11.0, Intel Core i5-11400 (6 cores / 12 logical processors), 8 GB RAM,
  10 workers, and an internal SSD. No other compute-intensive workload was
  active.

Create an isolated environment from the package root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The simulators themselves use only the Python standard library. The pinned
packages are required for plotting, statistical analysis, and topology
rendering.

## Experiments

### v31: principal zero-redundancy experiment

Runs ACA-RSA, A-MP-SS-RSA, fixed-partition MP-SS-RSA, FSA-RSA(SSFM), and the
other reported baselines over German, USNET, and TI at nominal load indices
5, 10, 15, 20, and 25.

```powershell
python .\simulators\v31_main.py --workers 24
```

### v32: equal-redundancy single-erasure experiment

Compares ACA-RSA and A-MP-SS-RSA-Repetition under the same redundancy target
and evaluates every possible one-unit erasure analytically.

```powershell
python .\simulators\v32_reliability.py --workers 10
```

### v33: path/segment-bound sensitivity

The built-in validation mode uses 3,000 requests and seeds 1-3:

```powershell
python .\simulators\v33_bound_sensitivity.py --validation --workers 3
```

The full paper-scale run is:

```powershell
python .\simulators\v33_bound_sensitivity.py --workers 24
```

### v34: traffic-distribution sensitivity

The validation and full commands are:

```powershell
python .\simulators\v34_traffic_sensitivity.py --validation --workers 3
python .\simulators\v34_traffic_sensitivity.py --workers 24
```

Worker count affects execution speed only. Each topology/scheme/load/seed run
retains the original sequential event ordering.

## Checkpoint resume

Each simulator writes transactional checkpoints and validates the stored
configuration fingerprint before resuming. In PowerShell, point the resume
environment variable to the interrupted results directory and rerun the same
simulator and scientific configuration. The worker count may be changed
because it affects execution speed, not event ordering:

```powershell
$env:ACA_RSA_RESUME_RESULTS_DIR = "C:\path\to\existing_results_directory"
python .\simulators\v31_main.py --workers 24
```

Clear the variable before starting an unrelated fresh run:

```powershell
Remove-Item Env:ACA_RSA_RESUME_RESULTS_DIR
```

## Reproduce the principal figures and tables

The commands below use the included processed results and write new output to
`reproduced_figures/`.

```powershell
python .\plotting_and_analysis\plot_result_comprehensive_summary_by_topology_v31.py `
  --summary-csv .\processed_results\v31\simulation_results_summary.csv `
  --output-dir .\reproduced_figures\v31_main `
  --image-format pdf --x-axis nominal

python .\plotting_and_analysis\plot_single_erasure_reliability_table_v32_combined_line.py `
  --summary-csv .\processed_results\v32\simulation_results_summary.csv `
  --outdir .\reproduced_figures\v32_reliability --format pdf

python .\plotting_and_analysis\plot_aca_sensitivity_v34_bar.py `
  --results-dir .\processed_results\v33 `
  --reporting-csv .\processed_results\v33\aca_sensitivity_v33_reporting_values.csv `
  --output-dir .\reproduced_figures\v33_bound_sensitivity `
  --load 25 --format pdf

python .\plotting_and_analysis\plot_traffic_distribution_sensitivity_v34.py `
  --results-dir .\processed_results\v34 `
  --raw-csv .\processed_results\v34\simulation_results_raw.csv `
  --summary-csv .\processed_results\v34\simulation_results_summary.csv `
  --debug-summary-csv .\processed_results\v34\figures\traffic_distribution_sensitivity_v34_fallback_by_seed.csv `
  --output-dir .\reproduced_figures\v34_traffic_sensitivity
```

The equal-carried-traffic FDM interpolation and the 60 paired v31 blocking
tests can be rerun with:

```powershell
python .\plotting_and_analysis\analyze_equal_carried_fdm_v31.py
python .\plotting_and_analysis\analyze_v31_paired_statistics.py
```

Regenerate the topology definitions and topology figures with:

```powershell
python .\plotting_and_analysis\export_topology_definitions.py
python .\plotting_and_analysis\draw_author_topologies.py
```

## Processed results

For each experiment, `simulation_results_raw.csv` contains seed-level output
and `simulation_results_summary.csv` contains means and 95% Student-t
confidence intervals. Configuration JSON files record the scientific
parameters, source hash, seed set, load levels, schemes, and topology setup.
The v31/v34 raw files also retain the traffic-trace hashes used to verify
matched request realizations.

The final manuscript-ready PDFs, LaTeX snippets, percentage-change table,
paired statistical tests, sensitivity summaries, and validation reports are
stored below the corresponding `processed_results/vXX/figures/` directory.

## Supplementary baseline reproduction

The `supplementary/baseline_reproduction/` directory contains the compiled
IEEE-style supplement, its LaTeX source, the processed v31 baseline subset,
the regeneration script, and the published/reproduced Table IV and Table V
checks. The source-style Table V calculation is kept separate from the
load-averaged percentage calculation used for the principal ACA-RSA results.

## Large debug logs

The full per-event K-debug checkpoint CSVs are deliberately not included in
the ZIP because they total more than 7 GB. Their compact aggregate tables and
all figures used in the manuscript are included. A fresh simulator run
regenerates the complete debug logs when exact event-level auditing is needed.

## Integrity check

From PowerShell, verify a file against the package manifest with:

```powershell
Get-FileHash .\simulators\v31_main.py -Algorithm SHA256
```

`MANIFEST_SHA256.txt` lists the expected SHA-256 value for every packaged
research artifact except the manifest itself.
