# Pooled OOF per-class AUC

Runs: `step1b` = `artifacts/dechazal/stage2/step1b_mlii/cv`, `nsv_matched` = `artifacts/dechazal/stage2/step2a_nsv_matched/cv`. Reference: `step1b`.
Seeds 42 43 44; mean ± sample std (ddof=1) over seeds. Scores are
`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.

## Decision: gain over the reference

`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate
seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.

| run | model | metric | reference | candidate | gain | spread | exceeds_spread | separated |
|---|---|---|---|---|---|---|---|---|
| nsv_matched | int8 | S_roc_auc | 0.801 ± 0.037 | 0.862 ± 0.030 | 0.061 | 0.037 | True | False |
| nsv_matched | int8 | S_pr_auc | 0.160 ± 0.068 | 0.124 ± 0.041 | -0.036 | 0.068 | False | False |
| nsv_matched | int8 | V_roc_auc | 0.977 ± 0.006 | 0.977 ± 0.005 | -0.001 | 0.006 | False | False |
| nsv_matched | int8 | V_pr_auc | 0.914 ± 0.025 | 0.912 ± 0.019 | -0.002 | 0.025 | False | False |
| nsv_matched | float | S_roc_auc | 0.803 ± 0.040 | 0.863 ± 0.030 | 0.060 | 0.040 | True | False |
| nsv_matched | float | S_pr_auc | 0.168 ± 0.069 | 0.127 ± 0.043 | -0.041 | 0.069 | False | False |
| nsv_matched | float | V_roc_auc | 0.977 ± 0.007 | 0.976 ± 0.005 | -0.001 | 0.007 | False | False |
| nsv_matched | float | V_pr_auc | 0.914 ± 0.025 | 0.912 ± 0.019 | -0.002 | 0.025 | False | False |

## Mean ± std over seeds

Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.
`*_pS` / `*_pV`: S vs. N AUC on P(S), V vs. N AUC on P(V); `S_as_V` / `V_as_S`: S beats called V and
V beats called S (N unless 1 - P(N) > threshold, else the likelier of S and V). N/S/V runs only (`—`
for binary); descriptive, never decisive.

| run | model | S_roc_auc | S_pr_auc | V_roc_auc | V_pr_auc | threshold | S_recall | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step1b | int8 | 0.801 ± 0.037 | 0.160 ± 0.068 | 0.977 ± 0.006 | 0.914 ± 0.025 | 0.450 ± 0.000 | 0.240 ± 0.128 | 0.852 ± 0.034 | 0.010 ± 0.004 | 0.310 ± 0.067 | — | — | — | — | — | — |
| step1b | float | 0.803 ± 0.040 | 0.168 ± 0.069 | 0.977 ± 0.007 | 0.914 ± 0.025 | 0.450 ± 0.000 | 0.246 ± 0.122 | 0.852 ± 0.033 | 0.010 ± 0.004 | 0.313 ± 0.069 | — | — | — | — | — | — |
| nsv_matched | int8 | 0.862 ± 0.030 | 0.124 ± 0.041 | 0.977 ± 0.005 | 0.912 ± 0.019 | 0.583 ± 0.076 | 0.188 ± 0.105 | 0.862 ± 0.043 | 0.013 ± 0.004 | 0.337 ± 0.060 | 0.805 ± 0.074 | 0.251 ± 0.165 | 0.979 ± 0.004 | 0.895 ± 0.008 | 52.667 ± 44.636 | 281.000 ± 83.899 |
| nsv_matched | float | 0.863 ± 0.030 | 0.127 ± 0.043 | 0.976 ± 0.005 | 0.912 ± 0.019 | 0.583 ± 0.076 | 0.191 ± 0.108 | 0.863 ± 0.043 | 0.013 ± 0.003 | 0.338 ± 0.056 | 0.806 ± 0.074 | 0.251 ± 0.161 | 0.979 ± 0.005 | 0.897 ± 0.007 | 52.333 ± 45.081 | 283.333 ± 86.818 |

## INT8 minus float, decision metrics

Per run, over seeds: mean ± std of (INT8 − float) and the largest |gap| of any seed.

| run | metric | int8_minus_float | max_abs_gap |
|---|---|---|---|
| step1b | S_roc_auc | -0.002 ± 0.004 | 0.007 |
| step1b | S_pr_auc | -0.009 ± 0.010 | 0.020 |
| step1b | V_roc_auc | 0.000 ± 0.001 | 0.001 |
| step1b | V_pr_auc | -0.000 ± 0.001 | 0.001 |
| nsv_matched | S_roc_auc | -0.001 ± 0.001 | 0.002 |
| nsv_matched | S_pr_auc | -0.003 ± 0.003 | 0.007 |
| nsv_matched | V_roc_auc | 0.000 ± 0.000 | 0.001 |
| nsv_matched | V_pr_auc | -0.000 ± 0.000 | 0.001 |

## Operating points at fixed abnormal recall, INT8 (descriptive)

Per run and target recall (`target_recall`: 0.90, 0.95): the highest threshold at which pooled-OOF
abnormal recall (S and V beats with abnormal_score > threshold) reaches the target recall. `abnormal_recall`
is the recall achieved; INT8 score ties can push it above the target recall.
`precision`, `TP` and `FP` are for abnormal vs. N over every OOF beat (uncapped).
`Se_*_flagged` (every run): the class's beats flagged abnormal. `Se_*_call` / `pP_*_call` (N/S/V runs
only, `—` for binary): Se and +P under the call rule, so an S beat called V is an S miss and counts
against +P_V. Float values are in `oof_comparison_operating_points.csv`.

| run | target_recall | threshold | abnormal_recall | precision | TP | FP | Se_S_flagged | Se_V_flagged | Se_S_call | Se_V_call | pP_S_call | pP_V_call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step1b | 0.900 | 0.242 ± 0.072 | 0.901 ± 0.001 | 0.358 ± 0.077 | 3970.333 ± 2.887 | 7448.000 ± 2194.792 | 0.653 ± 0.080 | 0.958 ± 0.018 | — | — | — | — |
| step1b | 0.950 | 0.128 ± 0.049 | 0.951 ± 0.001 | 0.203 ± 0.046 | 4189.333 ± 3.786 | 17123.000 ± 4316.077 | 0.821 ± 0.052 | 0.981 ± 0.013 | — | — | — | — |
| nsv_matched | 0.900 | 0.237 ± 0.057 | 0.901 ± 0.000 | 0.468 ± 0.137 | 3968.667 ± 2.082 | 5020.000 ± 2693.058 | 0.665 ± 0.044 | 0.955 ± 0.010 | 0.463 ± 0.043 | 0.844 ± 0.006 | 0.141 ± 0.097 | 0.592 ± 0.029 |
| nsv_matched | 0.950 | 0.138 ± 0.028 | 0.952 ± 0.001 | 0.270 ± 0.050 | 4193.333 ± 5.033 | 11718.000 ± 3295.131 | 0.853 ± 0.023 | 0.974 ± 0.005 | 0.640 ± 0.034 | 0.855 ± 0.003 | 0.067 ± 0.023 | 0.419 ± 0.024 |

## Per seed

| run | seed | model | n_beats | threshold | S_roc_auc | S_pr_auc | S_recall | V_roc_auc | V_pr_auc | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step1b | 42 | int8 | 48596 | 0.4500 | 0.7719 | 0.0962 | 0.1767 | 0.9747 | 0.8893 | 0.8240 | 0.0136 | 0.2344 | — | — | — | — | — | — |
| step1b | 42 | float | 48596 | 0.4500 | 0.7722 | 0.0983 | 0.1791 | 0.9740 | 0.8894 | 0.8245 | 0.0138 | 0.2358 | — | — | — | — | — | — |
| step1b | 43 | int8 | 48596 | 0.4500 | 0.8419 | 0.1509 | 0.1558 | 0.9725 | 0.9141 | 0.8426 | 0.0053 | 0.3633 | — | — | — | — | — | — |
| step1b | 43 | float | 48596 | 0.4500 | 0.8485 | 0.1711 | 0.1730 | 0.9717 | 0.9138 | 0.8429 | 0.0055 | 0.3674 | — | — | — | — | — | — |
| step1b | 44 | int8 | 48596 | 0.4500 | 0.7883 | 0.2319 | 0.3865 | 0.9841 | 0.9393 | 0.8894 | 0.0120 | 0.3320 | — | — | — | — | — | — |
| step1b | 44 | float | 48596 | 0.4500 | 0.7871 | 0.2353 | 0.3865 | 0.9845 | 0.9401 | 0.8894 | 0.0115 | 0.3368 | — | — | — | — | — | — |
| nsv_matched | 42 | int8 | 48596 | 0.6500 | 0.8740 | 0.1175 | 0.1141 | 0.9726 | 0.9037 | 0.8337 | 0.0095 | 0.3828 | 0.8428 | 0.1757 | 0.9753 | 0.8893 | 44 | 258 |
| nsv_matched | 42 | float | 48596 | 0.6500 | 0.8751 | 0.1175 | 0.1166 | 0.9719 | 0.9039 | 0.8340 | 0.0095 | 0.3834 | 0.8463 | 0.1813 | 0.9748 | 0.8898 | 44 | 258 |
| nsv_matched | 43 | int8 | 48596 | 0.6000 | 0.8280 | 0.0866 | 0.1423 | 0.9756 | 0.8988 | 0.8418 | 0.0141 | 0.2695 | 0.7198 | 0.1370 | 0.9781 | 0.9040 | 13 | 211 |
| nsv_matched | 43 | float | 48596 | 0.6000 | 0.8286 | 0.0890 | 0.1423 | 0.9751 | 0.8989 | 0.8421 | 0.0143 | 0.2754 | 0.7206 | 0.1369 | 0.9779 | 0.9041 | 12 | 212 |
| nsv_matched | 44 | int8 | 48596 | 0.5000 | 0.8831 | 0.1675 | 0.3080 | 0.9817 | 0.9339 | 0.9120 | 0.0164 | 0.3594 | 0.8530 | 0.4397 | 0.9839 | 0.8929 | 101 | 374 |
| nsv_matched | 44 | float | 48596 | 0.5000 | 0.8850 | 0.1742 | 0.3153 | 0.9816 | 0.9344 | 0.9128 | 0.0159 | 0.3551 | 0.8521 | 0.4358 | 0.9843 | 0.8966 | 101 | 380 |

## Per record: S vs. N ROC-AUC, INT8 (records with >= 20 S beats)

| record_id | nsv_matched s42 | nsv_matched s43 | nsv_matched s44 | step1b s42 | step1b s43 | step1b s44 |
|---|---|---|---|---|---|---|
| 118 | 0.384 | 0.898 | 0.984 | 0.989 | 0.928 | 0.999 |
| 124 | 0.754 | 0.512 | 0.724 | 0.599 | 0.555 | 0.697 |
| 207 | 0.989 | 0.975 | 0.988 | 0.990 | 0.986 | 0.967 |
| 209 | 0.975 | 0.931 | 0.917 | 0.809 | 0.814 | 0.708 |
| 220 | 0.963 | 0.961 | 0.982 | 0.965 | 0.892 | 0.983 |
| 223 | 0.997 | 0.963 | 0.964 | 0.967 | 0.977 | 0.991 |
