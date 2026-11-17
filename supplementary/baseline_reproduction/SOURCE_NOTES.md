# Source Notes

## Published reference

R. Srivastava and Y. N. Singh, "A novel fragmentation metric and
fragmentation-aware adaptive routing and spectrum allocation algorithm in
elastic optical network," *Optical Fiber Technology*, vol. 94, Art. no.
104318, 2025. DOI: 10.1016/j.yofte.2025.104318.

- Published low-load RBP values were transcribed from Table IV.
- Published signed percentage changes were transcribed from Table V.
- The source PDF is not redistributed in this package.

## Reproduced data

The reproduced values were extracted from:

`simulation_results_summary_ACA_RSA_v31_adaptive_mp_ss_metrics.csv`

Only the K1-RSA, K3-RSA, and FSA-RSA(SSFM) rows and the metrics required for
the reproduction checks are included in `data/v31_baseline_summary_subset.csv`.

## Metric naming

The source study calls the free-fragment-density quantity LFM. The current
manuscript calls the same fragments-per-slot quantity FDM to avoid implying
that a lower value is universally preferable. The supplementary comparison
therefore labels the column `LFM/FDM`.

## Interpretation

Table IV served as the low-load check used while fixing the topology-specific
load mapping. Table V provides the broader multi-load comparison. K1-RSA
closely reproduces all twelve topology-metric changes. K3-RSA has larger TI
differences because the source paper does not define its fixed route-table
construction or route ordering.

