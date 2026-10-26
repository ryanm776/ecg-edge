# Pooled OOF per-class AUC

Runs: `pre_fix` = `artifacts/dechazal/cv`, `step1` = `artifacts/dechazal/stage2/step1_beat_rr/cv`. Reference: `pre_fix`.
Seeds 42 43 44; mean ± sample std (ddof=1) over seeds. Scores are
`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.

## Decision: gain over the reference

`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate
seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.

| run | model | metric | reference | candidate | gain | spread | exceeds_spread | separated |
|---|---|---|---|---|---|---|---|---|
| step1 | int8 | S_roc_auc | 0.771 ± 0.060 | 0.790 ± 0.007 | 0.019 | 0.060 | False | False |
| step1 | int8 | S_pr_auc | 0.075 ± 0.028 | 0.122 ± 0.048 | 0.047 | 0.048 | False | False |
| step1 | int8 | V_roc_auc | 0.966 ± 0.003 | 0.973 ± 0.002 | 0.008 | 0.003 | True | True |
| step1 | int8 | V_pr_auc | 0.860 ± 0.010 | 0.894 ± 0.016 | 0.034 | 0.016 | True | True |
| step1 | float | S_roc_auc | 0.772 ± 0.060 | 0.795 ± 0.002 | 0.023 | 0.060 | False | False |
| step1 | float | S_pr_auc | 0.074 ± 0.028 | 0.127 ± 0.040 | 0.053 | 0.040 | True | True |
| step1 | float | V_roc_auc | 0.966 ± 0.003 | 0.973 ± 0.002 | 0.007 | 0.003 | True | True |
| step1 | float | V_pr_auc | 0.861 ± 0.010 | 0.897 ± 0.014 | 0.036 | 0.014 | True | True |

## Mean ± std over seeds

Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.

| run | model | S_roc_auc | S_pr_auc | V_roc_auc | V_pr_auc | threshold | S_recall | V_recall | N_fp_rate | S_median_score |
|---|---|---|---|---|---|---|---|---|---|---|
| pre_fix | int8 | 0.771 ± 0.060 | 0.075 ± 0.028 | 0.966 ± 0.003 | 0.860 ± 0.010 | 0.500 ± 0.000 | 0.094 ± 0.014 | 0.772 ± 0.007 | 0.012 ± 0.002 | 0.233 ± 0.033 |
| pre_fix | float | 0.772 ± 0.060 | 0.074 ± 0.028 | 0.966 ± 0.003 | 0.861 ± 0.010 | 0.500 ± 0.000 | 0.095 ± 0.013 | 0.776 ± 0.007 | 0.013 ± 0.002 | 0.235 ± 0.033 |
| step1 | int8 | 0.790 ± 0.007 | 0.122 ± 0.048 | 0.973 ± 0.002 | 0.894 ± 0.016 | 0.483 ± 0.076 | 0.237 ± 0.134 | 0.853 ± 0.024 | 0.016 ± 0.006 | 0.279 ± 0.052 |
| step1 | float | 0.795 ± 0.002 | 0.127 ± 0.040 | 0.973 ± 0.002 | 0.897 ± 0.014 | 0.483 ± 0.076 | 0.251 ± 0.116 | 0.852 ± 0.027 | 0.016 ± 0.006 | 0.289 ± 0.031 |

## Per seed

| run | seed | model | n_beats | threshold | S_roc_auc | S_pr_auc | S_recall | V_roc_auc | V_pr_auc | V_recall | N_fp_rate | S_median_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre_fix | 42 | int8 | 48605 | 0.5000 | 0.7812 | 0.0783 | 0.1080 | 0.9683 | 0.8648 | 0.7783 | 0.0121 | 0.2344 |
| pre_fix | 42 | float | 48605 | 0.5000 | 0.7819 | 0.0784 | 0.1104 | 0.9683 | 0.8656 | 0.7816 | 0.0128 | 0.2347 |
| pre_fix | 43 | int8 | 48605 | 0.5000 | 0.7068 | 0.0446 | 0.0798 | 0.9654 | 0.8488 | 0.7649 | 0.0144 | 0.1992 |
| pre_fix | 43 | float | 48605 | 0.5000 | 0.7072 | 0.0447 | 0.0847 | 0.9653 | 0.8492 | 0.7682 | 0.0145 | 0.2020 |
| pre_fix | 44 | int8 | 48605 | 0.5000 | 0.8258 | 0.1008 | 0.0933 | 0.9630 | 0.8675 | 0.7738 | 0.0100 | 0.2656 |
| pre_fix | 44 | float | 48605 | 0.5000 | 0.8266 | 0.0998 | 0.0908 | 0.9630 | 0.8679 | 0.7786 | 0.0105 | 0.2682 |
| step1 | 42 | int8 | 48596 | 0.5000 | 0.7950 | 0.1036 | 0.2380 | 0.9757 | 0.8791 | 0.8652 | 0.0223 | 0.3047 |
| step1 | 42 | float | 48596 | 0.5000 | 0.7960 | 0.1049 | 0.2405 | 0.9748 | 0.8839 | 0.8671 | 0.0223 | 0.3027 |
| step1 | 43 | int8 | 48596 | 0.5500 | 0.7819 | 0.0864 | 0.1031 | 0.9717 | 0.8922 | 0.8251 | 0.0101 | 0.2188 |
| step1 | 43 | float | 48596 | 0.5500 | 0.7964 | 0.1038 | 0.1399 | 0.9707 | 0.8943 | 0.8212 | 0.0098 | 0.2526 |
| step1 | 44 | int8 | 48596 | 0.4000 | 0.7930 | 0.1762 | 0.3706 | 0.9723 | 0.9118 | 0.8688 | 0.0155 | 0.3125 |
| step1 | 44 | float | 48596 | 0.4000 | 0.7928 | 0.1738 | 0.3718 | 0.9725 | 0.9118 | 0.8688 | 0.0161 | 0.3105 |

## Per record: S vs. N ROC-AUC, INT8 (records with >= 20 S beats)

| record_id | pre_fix s42 | pre_fix s43 | pre_fix s44 | step1 s42 | step1 s43 | step1 s44 |
|---|---|---|---|---|---|---|
| 118 | 0.843 | 0.964 | 0.960 | 0.992 | 0.978 | 0.999 |
| 124 | 0.569 | 0.588 | 0.557 | 0.630 | 0.510 | 0.664 |
| 207 | 0.975 | 0.983 | 0.988 | 0.995 | 0.987 | 0.992 |
| 209 | 0.583 | 0.869 | 0.964 | 0.853 | 0.859 | 0.577 |
| 220 | 0.994 | 0.974 | 0.977 | 0.846 | 0.914 | 0.983 |
| 223 | 0.958 | 0.951 | 0.987 | 0.978 | 0.992 | 0.990 |
