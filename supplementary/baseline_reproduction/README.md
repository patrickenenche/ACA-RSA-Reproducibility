# ACA-RSA Supplementary Baseline-Reproduction Package

This package supports the baseline-reproduction statements in the ACA-RSA
manuscript. It separates the source-study reproduction calculation from the
load-averaged percentage calculation used for the principal ACA-RSA results.

## Contents

- `supplementary/ACA_RSA_Supplementary_Baseline_Reproduction.tex`
  - Standalone IEEE-style supplementary document.
  - Contains the low-load Table IV check, the complete Table V comparison,
    and implementation-faithful pseudocode for K1-RSA, K3-RSA, and
    FSA-RSA(SSFM).
- `supplementary/ACA_RSA_Supplementary_Baseline_Reproduction.pdf`
  - Compiled, publication-ready version of the supplementary document.
- `manuscript_insertion/baseline_reproduction_main_text.tex`
  - Concise text intended for the main manuscript.
- `manuscript_insertion/optional_main_low_load_table.tex`
  - Optional low-load table if the authors prefer to retain numerical
    reproduction evidence in the main manuscript rather than only in the
    supplement.
- `data/v31_baseline_summary_subset.csv`
  - Seed-averaged v31 rows needed to regenerate the checks.
- `analysis/reproduce_baseline_validation.py`
  - Standard-library script that regenerates the validation CSV files.
- `generated/`
  - Recomputed low-load and multi-load comparison files.
- `SOURCE_NOTES.md`
  - Provenance, interpretation, and limitations.
- `CHECKSUMS.sha256`
  - SHA-256 checksums for package-integrity verification.

Only the concise text in `manuscript_insertion/` is intended for the main
paper. The detailed comparison tables, source-style calculation, and baseline
procedures remain in the supplementary document.

## Reproduce the tables

From the package root, run:

```powershell
python .\analysis\reproduce_baseline_validation.py
```

The script uses only the Python standard library and writes:

- `generated/baseline_low_load_validation.csv`
- `generated/baseline_table5_comparison.csv`
- `generated/validation_summary.txt`

The expected maximum absolute published/reproduced difference for the twelve
K1-RSA topology-metric entries is `1.33` percentage points.

## Important distinction between calculations

For the source-study Table V check, the script computes a signed relative
change separately at each nominal load index and then averages the five
changes. This is done only to reproduce the presentation of the source table.

The principal ACA-RSA percentage table in the manuscript uses a different and
more robust procedure: it first averages each metric over the five load indices
and then computes the relative change between those load-averaged values. The
latter avoids undefined load-specific ratios when a blocking probability is
zero. The two calculations must not be mixed.

## Source-study limitation

The source study does not specify how its three fixed K3-RSA routes are
generated or ordered. The implementation evaluated here precomputes the three
shortest physical-distance paths and attempts them sequentially using
First-Fit. The resulting K3-RSA values, particularly for TI, are therefore
reported as a documented common-simulator implementation rather than an exact
reproduction.

