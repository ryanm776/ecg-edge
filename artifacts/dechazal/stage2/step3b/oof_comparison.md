# Pooled OOF per-class AUC

Runs: `nsv_matched` = `artifacts/dechazal/stage2/step3_nsv_matched/cv`, `rr_norm` = `artifacts/dechazal/stage2/step3b_rr_norm/cv`. Reference: `nsv_matched`.
Seeds 42 43 44 45 46; mean ± sample std (ddof=1) over seeds. Scores are
`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.

## Decision: gain over the reference

`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate
seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.

| run | model | metric | reference | candidate | gain | spread | exceeds_spread | separated |
|---|---|---|---|---|---|---|---|---|
| rr_norm | int8 | S_roc_auc | 0.854 ± 0.028 | 0.850 ± 0.042 | -0.004 | 0.042 | False | False |
| rr_norm | int8 | S_pr_auc | 0.146 ± 0.043 | 0.230 ± 0.038 | 0.084 | 0.043 | True | False |
| rr_norm | int8 | V_roc_auc | 0.973 ± 0.006 | 0.983 ± 0.006 | 0.010 | 0.006 | True | False |
| rr_norm | int8 | V_pr_auc | 0.905 ± 0.018 | 0.933 ± 0.014 | 0.029 | 0.018 | True | False |
| rr_norm | float | S_roc_auc | 0.855 ± 0.028 | 0.850 ± 0.044 | -0.005 | 0.044 | False | False |
| rr_norm | float | S_pr_auc | 0.147 ± 0.042 | 0.232 ± 0.039 | 0.085 | 0.042 | True | False |
| rr_norm | float | V_roc_auc | 0.973 ± 0.006 | 0.983 ± 0.006 | 0.010 | 0.006 | True | False |
| rr_norm | float | V_pr_auc | 0.905 ± 0.018 | 0.933 ± 0.014 | 0.028 | 0.018 | True | False |

## Mean ± std over seeds

Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.
`*_pS` / `*_pV`: S vs. N AUC on P(S), V vs. N AUC on P(V); `S_as_V` / `V_as_S`: S beats called V and
V beats called S (N unless 1 - P(N) > threshold, else the likelier of S and V). N/S/V runs only (`—`
for binary); descriptive, never decisive.

| run | model | S_roc_auc | S_pr_auc | V_roc_auc | V_pr_auc | threshold | S_recall | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nsv_matched | int8 | 0.854 ± 0.028 | 0.146 ± 0.043 | 0.973 ± 0.006 | 0.905 ± 0.018 | 0.570 ± 0.057 | 0.256 ± 0.133 | 0.865 ± 0.032 | 0.014 ± 0.005 | 0.373 ± 0.080 | 0.814 ± 0.062 | 0.273 ± 0.132 | 0.975 ± 0.007 | 0.882 ± 0.020 | 83.800 ± 53.186 | 260.400 ± 74.433 |
| nsv_matched | float | 0.855 ± 0.028 | 0.147 ± 0.042 | 0.973 ± 0.006 | 0.905 ± 0.018 | 0.570 ± 0.057 | 0.258 ± 0.133 | 0.866 ± 0.032 | 0.014 ± 0.005 | 0.373 ± 0.077 | 0.814 ± 0.062 | 0.274 ± 0.130 | 0.975 ± 0.007 | 0.883 ± 0.020 | 83.600 ± 53.472 | 259.800 ± 77.667 |
| rr_norm | int8 | 0.850 ± 0.042 | 0.230 ± 0.038 | 0.983 ± 0.006 | 0.933 ± 0.014 | 0.520 ± 0.057 | 0.362 ± 0.092 | 0.891 ± 0.032 | 0.015 ± 0.006 | 0.360 ± 0.074 | 0.845 ± 0.043 | 0.297 ± 0.094 | 0.984 ± 0.005 | 0.914 ± 0.016 | 136.000 ± 50.498 | 268.200 ± 110.936 |
| rr_norm | float | 0.850 ± 0.044 | 0.232 ± 0.039 | 0.983 ± 0.006 | 0.933 ± 0.014 | 0.520 ± 0.057 | 0.364 ± 0.094 | 0.890 ± 0.033 | 0.016 ± 0.006 | 0.362 ± 0.073 | 0.845 ± 0.045 | 0.299 ± 0.095 | 0.984 ± 0.004 | 0.915 ± 0.016 | 138.200 ± 50.465 | 262.200 ± 110.638 |

## INT8 minus float, decision metrics

Per run, over seeds: mean ± std of (INT8 − float) and the largest |gap| of any seed.

| run | metric | int8_minus_float | max_abs_gap |
|---|---|---|---|
| nsv_matched | S_roc_auc | -0.001 ± 0.001 | 0.002 |
| nsv_matched | S_pr_auc | -0.001 ± 0.003 | 0.007 |
| nsv_matched | V_roc_auc | 0.000 ± 0.000 | 0.001 |
| nsv_matched | V_pr_auc | -0.000 ± 0.000 | 0.001 |
| rr_norm | S_roc_auc | 0.000 ± 0.003 | 0.004 |
| rr_norm | S_pr_auc | -0.001 ± 0.005 | 0.009 |
| rr_norm | V_roc_auc | -0.000 ± 0.000 | 0.000 |
| rr_norm | V_pr_auc | -0.000 ± 0.001 | 0.001 |

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
| nsv_matched | 0.900 | 0.239 ± 0.046 | 0.901 ± 0.000 | 0.438 ± 0.129 | 3968.200 ± 1.924 | 5776.600 ± 2987.707 | 0.680 ± 0.039 | 0.951 ± 0.009 | 0.462 ± 0.036 | 0.846 ± 0.006 | 0.141 ± 0.093 | 0.538 ± 0.079 |
| nsv_matched | 0.950 | 0.128 ± 0.024 | 0.952 ± 0.001 | 0.237 ± 0.060 | 4191.800 ± 4.494 | 14413.400 ± 4759.385 | 0.859 ± 0.018 | 0.973 ± 0.004 | 0.630 ± 0.028 | 0.856 ± 0.004 | 0.063 ± 0.026 | 0.363 ± 0.089 |
| rr_norm | 0.900 | 0.220 ± 0.039 | 0.901 ± 0.001 | 0.535 ± 0.092 | 3968.400 ± 2.510 | 3610.400 ± 1233.168 | 0.637 ± 0.027 | 0.961 ± 0.007 | 0.435 ± 0.044 | 0.865 ± 0.027 | 0.192 ± 0.052 | 0.568 ± 0.103 |
| rr_norm | 0.950 | 0.094 ± 0.020 | 0.952 ± 0.001 | 0.238 ± 0.057 | 4192.400 ± 3.209 | 14333.200 ± 4700.168 | 0.824 ± 0.043 | 0.981 ± 0.010 | 0.618 ± 0.057 | 0.877 ± 0.026 | 0.061 ± 0.023 | 0.352 ± 0.053 |

## Per seed

| run | seed | model | n_beats | threshold | S_roc_auc | S_pr_auc | S_recall | V_roc_auc | V_pr_auc | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nsv_matched | 42 | int8 | 48596 | 0.6500 | 0.8740 | 0.1175 | 0.1141 | 0.9726 | 0.9037 | 0.8337 | 0.0095 | 0.3828 | 0.8428 | 0.1757 | 0.9753 | 0.8893 | 44 | 258 |
| nsv_matched | 42 | float | 48596 | 0.6500 | 0.8751 | 0.1175 | 0.1166 | 0.9719 | 0.9039 | 0.8340 | 0.0095 | 0.3834 | 0.8463 | 0.1813 | 0.9748 | 0.8898 | 44 | 258 |
| nsv_matched | 43 | int8 | 48596 | 0.6000 | 0.8280 | 0.0866 | 0.1423 | 0.9756 | 0.8988 | 0.8418 | 0.0141 | 0.2695 | 0.7198 | 0.1370 | 0.9781 | 0.9040 | 13 | 211 |
| nsv_matched | 43 | float | 48596 | 0.6000 | 0.8286 | 0.0890 | 0.1423 | 0.9751 | 0.8989 | 0.8421 | 0.0143 | 0.2754 | 0.7206 | 0.1369 | 0.9779 | 0.9041 | 12 | 212 |
| nsv_matched | 44 | int8 | 48596 | 0.5000 | 0.8831 | 0.1675 | 0.3080 | 0.9817 | 0.9339 | 0.9120 | 0.0164 | 0.3594 | 0.8530 | 0.4397 | 0.9839 | 0.8929 | 101 | 374 |
| nsv_matched | 44 | float | 48596 | 0.5000 | 0.8850 | 0.1742 | 0.3153 | 0.9816 | 0.9344 | 0.9128 | 0.0159 | 0.3551 | 0.8521 | 0.4358 | 0.9843 | 0.8966 | 101 | 380 |
| nsv_matched | 45 | int8 | 48596 | 0.5500 | 0.8214 | 0.1681 | 0.2724 | 0.9654 | 0.9026 | 0.8579 | 0.0100 | 0.3594 | 0.7813 | 0.2284 | 0.9686 | 0.8715 | 125 | 180 |
| nsv_matched | 45 | float | 48596 | 0.5500 | 0.8208 | 0.1664 | 0.2724 | 0.9658 | 0.9029 | 0.8591 | 0.0102 | 0.3635 | 0.7801 | 0.2310 | 0.9689 | 0.8721 | 126 | 175 |
| nsv_matched | 46 | int8 | 48596 | 0.5500 | 0.8629 | 0.1910 | 0.4429 | 0.9717 | 0.8835 | 0.8813 | 0.0209 | 0.4922 | 0.8708 | 0.3821 | 0.9666 | 0.8543 | 136 | 279 |
| nsv_matched | 46 | float | 48596 | 0.5500 | 0.8634 | 0.1897 | 0.4429 | 0.9718 | 0.8844 | 0.8813 | 0.0208 | 0.4895 | 0.8711 | 0.3837 | 0.9670 | 0.8549 | 135 | 274 |
| rr_norm | 42 | int8 | 48596 | 0.5000 | 0.8762 | 0.2436 | 0.4491 | 0.9749 | 0.9107 | 0.9081 | 0.0234 | 0.4648 | 0.8832 | 0.3239 | 0.9809 | 0.8908 | 145 | 389 |
| rr_norm | 42 | float | 48596 | 0.5000 | 0.8762 | 0.2432 | 0.4552 | 0.9745 | 0.9103 | 0.9084 | 0.0236 | 0.4667 | 0.8839 | 0.3268 | 0.9809 | 0.8921 | 145 | 381 |
| rr_norm | 43 | int8 | 48596 | 0.5000 | 0.8777 | 0.2110 | 0.3387 | 0.9851 | 0.9319 | 0.8727 | 0.0124 | 0.3477 | 0.8593 | 0.1587 | 0.9780 | 0.9091 | 169 | 298 |
| rr_norm | 43 | float | 48596 | 0.5000 | 0.8807 | 0.2199 | 0.3460 | 0.9855 | 0.9328 | 0.8713 | 0.0124 | 0.3456 | 0.8635 | 0.1611 | 0.9781 | 0.9105 | 175 | 290 |
| rr_norm | 44 | int8 | 48596 | 0.6000 | 0.7773 | 0.1861 | 0.2638 | 0.9874 | 0.9419 | 0.8702 | 0.0110 | 0.2969 | 0.7888 | 0.3260 | 0.9860 | 0.9233 | 113 | 157 |
| rr_norm | 44 | float | 48596 | 0.6000 | 0.7734 | 0.1844 | 0.2613 | 0.9874 | 0.9424 | 0.8694 | 0.0110 | 0.3001 | 0.7850 | 0.3218 | 0.9860 | 0.9238 | 116 | 154 |
| rr_norm | 45 | int8 | 48596 | 0.4500 | 0.8674 | 0.2882 | 0.4663 | 0.9891 | 0.9466 | 0.9401 | 0.0194 | 0.4023 | 0.8100 | 0.2624 | 0.9898 | 0.9335 | 191 | 350 |
| rr_norm | 45 | float | 48596 | 0.4500 | 0.8679 | 0.2906 | 0.4663 | 0.9891 | 0.9469 | 0.9398 | 0.0198 | 0.4036 | 0.8107 | 0.2647 | 0.9897 | 0.9340 | 191 | 347 |
| rr_norm | 46 | int8 | 48596 | 0.5500 | 0.8518 | 0.2231 | 0.2920 | 0.9798 | 0.9339 | 0.8643 | 0.0106 | 0.2891 | 0.8830 | 0.4146 | 0.9830 | 0.9157 | 62 | 147 |
| rr_norm | 46 | float | 48596 | 0.5500 | 0.8515 | 0.2212 | 0.2908 | 0.9797 | 0.9337 | 0.8618 | 0.0109 | 0.2929 | 0.8838 | 0.4188 | 0.9829 | 0.9155 | 64 | 139 |

## Per record: S vs. N ROC-AUC, INT8 (records with >= 20 S beats)

| record_id | nsv_matched s42 | nsv_matched s43 | nsv_matched s44 | nsv_matched s45 | nsv_matched s46 | rr_norm s42 | rr_norm s43 | rr_norm s44 | rr_norm s45 | rr_norm s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 118 | 0.384 | 0.898 | 0.984 | 0.690 | 0.958 | 0.979 | 0.996 | 0.998 | 0.997 | 0.996 |
| 124 | 0.754 | 0.512 | 0.724 | 0.400 | 0.435 | 0.386 | 0.616 | 0.456 | 0.414 | 0.533 |
| 207 | 0.989 | 0.975 | 0.988 | 0.992 | 0.958 | 0.991 | 0.991 | 0.996 | 0.995 | 0.969 |
| 209 | 0.975 | 0.931 | 0.917 | 0.732 | 0.947 | 0.940 | 0.976 | 0.851 | 0.850 | 0.830 |
| 220 | 0.963 | 0.961 | 0.982 | 0.948 | 0.965 | 1.000 | 0.999 | 0.999 | 0.990 | 0.986 |
| 223 | 0.997 | 0.963 | 0.964 | 0.950 | 0.934 | 0.980 | 0.980 | 0.984 | 0.997 | 0.995 |
