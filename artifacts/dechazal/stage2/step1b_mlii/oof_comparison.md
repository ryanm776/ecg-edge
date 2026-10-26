# Pooled OOF per-class AUC

Runs: `step1` = `artifacts/dechazal/stage2/step1_beat_rr/cv`, `step1b` = `artifacts/dechazal/stage2/step1b_mlii/cv`. Reference: `step1`.
Seeds 42 43 44; mean ± sample std (ddof=1) over seeds. Scores are
`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.

## Decision: gain over the reference

`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate
seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.

| run | model | metric | reference | candidate | gain | spread | exceeds_spread | separated |
|---|---|---|---|---|---|---|---|---|
| step1b | int8 | S_roc_auc | 0.790 ± 0.007 | 0.801 ± 0.037 | 0.011 | 0.037 | False | False |
| step1b | int8 | S_pr_auc | 0.122 ± 0.048 | 0.160 ± 0.068 | 0.038 | 0.068 | False | False |
| step1b | int8 | V_roc_auc | 0.973 ± 0.002 | 0.977 ± 0.006 | 0.004 | 0.006 | False | False |
| step1b | int8 | V_pr_auc | 0.894 ± 0.016 | 0.914 ± 0.025 | 0.020 | 0.025 | False | False |
| step1b | float | S_roc_auc | 0.795 ± 0.002 | 0.803 ± 0.040 | 0.008 | 0.040 | False | False |
| step1b | float | S_pr_auc | 0.127 ± 0.040 | 0.168 ± 0.069 | 0.041 | 0.069 | False | False |
| step1b | float | V_roc_auc | 0.973 ± 0.002 | 0.977 ± 0.007 | 0.004 | 0.007 | False | False |
| step1b | float | V_pr_auc | 0.897 ± 0.014 | 0.914 ± 0.025 | 0.018 | 0.025 | False | False |

## Mean ± std over seeds

Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.

| run | model | S_roc_auc | S_pr_auc | V_roc_auc | V_pr_auc | threshold | S_recall | V_recall | N_fp_rate | S_median_score |
|---|---|---|---|---|---|---|---|---|---|---|
| step1 | int8 | 0.790 ± 0.007 | 0.122 ± 0.048 | 0.973 ± 0.002 | 0.894 ± 0.016 | 0.483 ± 0.076 | 0.237 ± 0.134 | 0.853 ± 0.024 | 0.016 ± 0.006 | 0.279 ± 0.052 |
| step1 | float | 0.795 ± 0.002 | 0.127 ± 0.040 | 0.973 ± 0.002 | 0.897 ± 0.014 | 0.483 ± 0.076 | 0.251 ± 0.116 | 0.852 ± 0.027 | 0.016 ± 0.006 | 0.289 ± 0.031 |
| step1b | int8 | 0.801 ± 0.037 | 0.160 ± 0.068 | 0.977 ± 0.006 | 0.914 ± 0.025 | 0.450 ± 0.000 | 0.240 ± 0.128 | 0.852 ± 0.034 | 0.010 ± 0.004 | 0.310 ± 0.067 |
| step1b | float | 0.803 ± 0.040 | 0.168 ± 0.069 | 0.977 ± 0.007 | 0.914 ± 0.025 | 0.450 ± 0.000 | 0.246 ± 0.122 | 0.852 ± 0.033 | 0.010 ± 0.004 | 0.313 ± 0.069 |

## Per seed

| run | seed | model | n_beats | threshold | S_roc_auc | S_pr_auc | S_recall | V_roc_auc | V_pr_auc | V_recall | N_fp_rate | S_median_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step1 | 42 | int8 | 48596 | 0.5000 | 0.7950 | 0.1036 | 0.2380 | 0.9757 | 0.8791 | 0.8652 | 0.0223 | 0.3047 |
| step1 | 42 | float | 48596 | 0.5000 | 0.7960 | 0.1049 | 0.2405 | 0.9748 | 0.8839 | 0.8671 | 0.0223 | 0.3027 |
| step1 | 43 | int8 | 48596 | 0.5500 | 0.7819 | 0.0864 | 0.1031 | 0.9717 | 0.8922 | 0.8251 | 0.0101 | 0.2188 |
| step1 | 43 | float | 48596 | 0.5500 | 0.7964 | 0.1038 | 0.1399 | 0.9707 | 0.8943 | 0.8212 | 0.0098 | 0.2526 |
| step1 | 44 | int8 | 48596 | 0.4000 | 0.7930 | 0.1762 | 0.3706 | 0.9723 | 0.9118 | 0.8688 | 0.0155 | 0.3125 |
| step1 | 44 | float | 48596 | 0.4000 | 0.7928 | 0.1738 | 0.3718 | 0.9725 | 0.9118 | 0.8688 | 0.0161 | 0.3105 |
| step1b | 42 | int8 | 48596 | 0.4500 | 0.7719 | 0.0962 | 0.1767 | 0.9747 | 0.8893 | 0.8240 | 0.0136 | 0.2344 |
| step1b | 42 | float | 48596 | 0.4500 | 0.7722 | 0.0983 | 0.1791 | 0.9740 | 0.8894 | 0.8245 | 0.0138 | 0.2358 |
| step1b | 43 | int8 | 48596 | 0.4500 | 0.8419 | 0.1509 | 0.1558 | 0.9725 | 0.9141 | 0.8426 | 0.0053 | 0.3633 |
| step1b | 43 | float | 48596 | 0.4500 | 0.8485 | 0.1711 | 0.1730 | 0.9717 | 0.9138 | 0.8429 | 0.0055 | 0.3674 |
| step1b | 44 | int8 | 48596 | 0.4500 | 0.7883 | 0.2319 | 0.3865 | 0.9841 | 0.9393 | 0.8894 | 0.0120 | 0.3320 |
| step1b | 44 | float | 48596 | 0.4500 | 0.7871 | 0.2353 | 0.3865 | 0.9845 | 0.9401 | 0.8894 | 0.0115 | 0.3368 |

## Per record: S vs. N ROC-AUC, INT8 (records with >= 20 S beats)

| record_id | step1 s42 | step1 s43 | step1 s44 | step1b s42 | step1b s43 | step1b s44 |
|---|---|---|---|---|---|---|
| 118 | 0.992 | 0.978 | 0.999 | 0.989 | 0.928 | 0.999 |
| 124 | 0.630 | 0.510 | 0.664 | 0.599 | 0.555 | 0.697 |
| 207 | 0.995 | 0.987 | 0.992 | 0.990 | 0.986 | 0.967 |
| 209 | 0.853 | 0.859 | 0.577 | 0.809 | 0.814 | 0.708 |
| 220 | 0.846 | 0.914 | 0.983 | 0.965 | 0.892 | 0.983 |
| 223 | 0.978 | 0.992 | 0.990 | 0.967 | 0.977 | 0.991 |
