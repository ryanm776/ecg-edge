# Pooled OOF per-class AUC

Runs: `nsv_matched` = `artifacts/dechazal/stage2/step3_nsv_matched/cv`, `nsv_rr30` = `artifacts/dechazal/stage2/step3a_rr30/cv`. Reference: `nsv_matched`.
Seeds 42 43 44 45 46; mean ± sample std (ddof=1) over seeds. Scores are
`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.

## Decision: gain over the reference

`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate
seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.

| run | model | metric | reference | candidate | gain | spread | exceeds_spread | separated |
|---|---|---|---|---|---|---|---|---|
| nsv_rr30 | int8 | S_roc_auc | 0.854 ± 0.028 | 0.817 ± 0.040 | -0.036 | 0.040 | False | False |
| nsv_rr30 | int8 | S_pr_auc | 0.146 ± 0.043 | 0.144 ± 0.047 | -0.002 | 0.047 | False | False |
| nsv_rr30 | int8 | V_roc_auc | 0.973 ± 0.006 | 0.972 ± 0.006 | -0.002 | 0.006 | False | False |
| nsv_rr30 | int8 | V_pr_auc | 0.905 ± 0.018 | 0.899 ± 0.018 | -0.005 | 0.018 | False | False |
| nsv_rr30 | float | S_roc_auc | 0.855 ± 0.028 | 0.818 ± 0.039 | -0.036 | 0.039 | False | False |
| nsv_rr30 | float | S_pr_auc | 0.147 ± 0.042 | 0.143 ± 0.045 | -0.004 | 0.045 | False | False |
| nsv_rr30 | float | V_roc_auc | 0.973 ± 0.006 | 0.972 ± 0.006 | -0.002 | 0.006 | False | False |
| nsv_rr30 | float | V_pr_auc | 0.905 ± 0.018 | 0.900 ± 0.018 | -0.005 | 0.018 | False | False |

## Mean ± std over seeds

Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.
`*_pS` / `*_pV`: S vs. N AUC on P(S), V vs. N AUC on P(V); `S_as_V` / `V_as_S`: S beats called V and
V beats called S (N unless 1 - P(N) > threshold, else the likelier of S and V). N/S/V runs only (`—`
for binary); descriptive, never decisive.

| run | model | S_roc_auc | S_pr_auc | V_roc_auc | V_pr_auc | threshold | S_recall | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nsv_matched | int8 | 0.854 ± 0.028 | 0.146 ± 0.043 | 0.973 ± 0.006 | 0.905 ± 0.018 | 0.570 ± 0.057 | 0.256 ± 0.133 | 0.865 ± 0.032 | 0.014 ± 0.005 | 0.373 ± 0.080 | 0.814 ± 0.062 | 0.273 ± 0.132 | 0.975 ± 0.007 | 0.882 ± 0.020 | 83.800 ± 53.186 | 260.400 ± 74.433 |
| nsv_matched | float | 0.855 ± 0.028 | 0.147 ± 0.042 | 0.973 ± 0.006 | 0.905 ± 0.018 | 0.570 ± 0.057 | 0.258 ± 0.133 | 0.866 ± 0.032 | 0.014 ± 0.005 | 0.373 ± 0.077 | 0.814 ± 0.062 | 0.274 ± 0.130 | 0.975 ± 0.007 | 0.883 ± 0.020 | 83.600 ± 53.472 | 259.800 ± 77.667 |
| nsv_rr30 | int8 | 0.817 ± 0.040 | 0.144 ± 0.047 | 0.972 ± 0.006 | 0.899 ± 0.018 | 0.540 ± 0.082 | 0.270 ± 0.158 | 0.864 ± 0.021 | 0.015 ± 0.006 | 0.342 ± 0.091 | 0.779 ± 0.049 | 0.270 ± 0.082 | 0.975 ± 0.010 | 0.889 ± 0.035 | 70.800 ± 49.414 | 168.400 ± 85.102 |
| nsv_rr30 | float | 0.818 ± 0.039 | 0.143 ± 0.045 | 0.972 ± 0.006 | 0.900 ± 0.018 | 0.540 ± 0.082 | 0.269 ± 0.158 | 0.864 ± 0.021 | 0.015 ± 0.007 | 0.345 ± 0.088 | 0.779 ± 0.050 | 0.272 ± 0.082 | 0.975 ± 0.009 | 0.890 ± 0.034 | 70.600 ± 50.387 | 165.800 ± 84.813 |

## INT8 minus float, decision metrics

Per run, over seeds: mean ± std of (INT8 − float) and the largest |gap| of any seed.

| run | metric | int8_minus_float | max_abs_gap |
|---|---|---|---|
| nsv_matched | S_roc_auc | -0.001 ± 0.001 | 0.002 |
| nsv_matched | S_pr_auc | -0.001 ± 0.003 | 0.007 |
| nsv_matched | V_roc_auc | 0.000 ± 0.000 | 0.001 |
| nsv_matched | V_pr_auc | -0.000 ± 0.000 | 0.001 |
| nsv_rr30 | S_roc_auc | -0.001 ± 0.001 | 0.002 |
| nsv_rr30 | S_pr_auc | 0.000 ± 0.003 | 0.005 |
| nsv_rr30 | V_roc_auc | 0.000 ± 0.000 | 0.001 |
| nsv_rr30 | V_pr_auc | -0.001 ± 0.001 | 0.002 |

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
| nsv_rr30 | 0.900 | 0.215 ± 0.045 | 0.901 ± 0.000 | 0.340 ± 0.075 | 3967.600 ± 1.949 | 8144.400 ± 2633.183 | 0.658 ± 0.032 | 0.956 ± 0.008 | 0.445 ± 0.028 | 0.878 ± 0.015 | 0.101 ± 0.051 | 0.457 ± 0.135 |
| nsv_rr30 | 0.950 | 0.113 ± 0.019 | 0.952 ± 0.001 | 0.187 ± 0.029 | 4191.800 ± 3.899 | 18682.000 ± 3508.754 | 0.846 ± 0.027 | 0.975 ± 0.007 | 0.620 ± 0.044 | 0.890 ± 0.017 | 0.045 ± 0.012 | 0.310 ± 0.082 |

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
| nsv_rr30 | 42 | int8 | 48596 | 0.4500 | 0.8518 | 0.1947 | 0.5117 | 0.9685 | 0.8757 | 0.8811 | 0.0242 | 0.4648 | 0.7922 | 0.3267 | 0.9746 | 0.8726 | 154 | 250 |
| nsv_rr30 | 42 | float | 48596 | 0.4500 | 0.8508 | 0.1902 | 0.5104 | 0.9684 | 0.8752 | 0.8811 | 0.0245 | 0.4648 | 0.7918 | 0.3254 | 0.9747 | 0.8733 | 155 | 252 |
| nsv_rr30 | 43 | int8 | 48596 | 0.5000 | 0.7861 | 0.1514 | 0.1718 | 0.9649 | 0.9119 | 0.8540 | 0.0079 | 0.2227 | 0.7138 | 0.1616 | 0.9774 | 0.9195 | 49 | 46 |
| nsv_rr30 | 43 | float | 48596 | 0.5000 | 0.7879 | 0.1540 | 0.1718 | 0.9640 | 0.9123 | 0.8554 | 0.0080 | 0.2307 | 0.7112 | 0.1625 | 0.9770 | 0.9197 | 49 | 43 |
| nsv_rr30 | 44 | int8 | 48596 | 0.6000 | 0.8216 | 0.1138 | 0.1472 | 0.9772 | 0.9032 | 0.8379 | 0.0103 | 0.3320 | 0.7983 | 0.2121 | 0.9801 | 0.8942 | 28 | 211 |
| nsv_rr30 | 44 | float | 48596 | 0.6000 | 0.8227 | 0.1152 | 0.1411 | 0.9772 | 0.9048 | 0.8357 | 0.0099 | 0.3300 | 0.8000 | 0.2158 | 0.9803 | 0.8962 | 25 | 204 |
| nsv_rr30 | 45 | int8 | 48596 | 0.5000 | 0.8597 | 0.1794 | 0.3497 | 0.9778 | 0.9190 | 0.8889 | 0.0144 | 0.3867 | 0.7501 | 0.2892 | 0.9845 | 0.9222 | 75 | 115 |
| nsv_rr30 | 45 | float | 48596 | 0.5000 | 0.8594 | 0.1766 | 0.3497 | 0.9778 | 0.9195 | 0.8883 | 0.0146 | 0.3874 | 0.7502 | 0.2906 | 0.9845 | 0.9227 | 75 | 116 |
| nsv_rr30 | 46 | int8 | 48596 | 0.6500 | 0.7678 | 0.0794 | 0.1706 | 0.9708 | 0.8870 | 0.8585 | 0.0176 | 0.3047 | 0.8418 | 0.3594 | 0.9595 | 0.8387 | 48 | 220 |
| nsv_rr30 | 46 | float | 48596 | 0.6500 | 0.7698 | 0.0804 | 0.1718 | 0.9707 | 0.8887 | 0.8585 | 0.0177 | 0.3098 | 0.8431 | 0.3641 | 0.9597 | 0.8401 | 49 | 214 |

## Per record: S vs. N ROC-AUC, INT8 (records with >= 20 S beats)

| record_id | nsv_matched s42 | nsv_matched s43 | nsv_matched s44 | nsv_matched s45 | nsv_matched s46 | nsv_rr30 s42 | nsv_rr30 s43 | nsv_rr30 s44 | nsv_rr30 s45 | nsv_rr30 s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 118 | 0.384 | 0.898 | 0.984 | 0.690 | 0.958 | 0.854 | 0.790 | 0.901 | 0.910 | 0.991 |
| 124 | 0.754 | 0.512 | 0.724 | 0.400 | 0.435 | 0.532 | 0.629 | 0.713 | 0.624 | 0.624 |
| 207 | 0.989 | 0.975 | 0.988 | 0.992 | 0.958 | 0.982 | 0.989 | 0.984 | 0.947 | 0.975 |
| 209 | 0.975 | 0.931 | 0.917 | 0.732 | 0.947 | 0.876 | 0.698 | 0.598 | 0.780 | 0.891 |
| 220 | 0.963 | 0.961 | 0.982 | 0.948 | 0.965 | 0.992 | 0.993 | 0.989 | 0.996 | 0.967 |
| 223 | 0.997 | 0.963 | 0.964 | 0.950 | 0.934 | 0.920 | 0.995 | 0.991 | 0.968 | 0.903 |
