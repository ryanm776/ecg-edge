# Pooled OOF per-class AUC

Runs: `rr_norm` = `artifacts/dechazal/stage2/step3b_rr_norm/cv`, `item1` = `artifacts/dechazal/stage2/item1_207_fix/cv`. Reference: `rr_norm`.
Seeds 42 43 44 45 46; mean ± sample std (ddof=1) over seeds. Scores are
`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.

## Decision: gain over the reference

`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate
seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.

| run | model | metric | reference | candidate | gain | spread | exceeds_spread | separated |
|---|---|---|---|---|---|---|---|---|
| item1 | int8 | S_roc_auc | 0.850 ± 0.042 | 0.890 ± 0.008 | 0.040 | 0.042 | False | False |
| item1 | int8 | S_pr_auc | 0.230 ± 0.038 | 0.284 ± 0.057 | 0.054 | 0.057 | False | False |
| item1 | int8 | V_roc_auc | 0.983 ± 0.006 | 0.984 ± 0.005 | 0.001 | 0.006 | False | False |
| item1 | int8 | V_pr_auc | 0.933 ± 0.014 | 0.931 ± 0.019 | -0.002 | 0.019 | False | False |
| item1 | float | S_roc_auc | 0.850 ± 0.044 | 0.891 ± 0.008 | 0.041 | 0.044 | False | False |
| item1 | float | S_pr_auc | 0.232 ± 0.039 | 0.285 ± 0.059 | 0.053 | 0.059 | False | False |
| item1 | float | V_roc_auc | 0.983 ± 0.006 | 0.984 ± 0.005 | 0.001 | 0.006 | False | False |
| item1 | float | V_pr_auc | 0.933 ± 0.014 | 0.932 ± 0.018 | -0.002 | 0.018 | False | False |

## Mean ± std over seeds

Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.
`*_pS` / `*_pV`: S vs. N AUC on P(S), V vs. N AUC on P(V); `S_as_V` / `V_as_S`: S beats called V and
V beats called S (N unless 1 - P(N) > threshold, else the likelier of S and V). N/S/V runs only (`—`
for binary); descriptive, never decisive.

| run | model | S_roc_auc | S_pr_auc | V_roc_auc | V_pr_auc | threshold | S_recall | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rr_norm | int8 | 0.850 ± 0.042 | 0.230 ± 0.038 | 0.983 ± 0.006 | 0.933 ± 0.014 | 0.520 ± 0.057 | 0.362 ± 0.092 | 0.891 ± 0.032 | 0.015 ± 0.006 | 0.360 ± 0.074 | 0.845 ± 0.043 | 0.297 ± 0.094 | 0.984 ± 0.005 | 0.914 ± 0.016 | 136.000 ± 50.498 | 268.200 ± 110.936 |
| rr_norm | float | 0.850 ± 0.044 | 0.232 ± 0.039 | 0.983 ± 0.006 | 0.933 ± 0.014 | 0.520 ± 0.057 | 0.364 ± 0.094 | 0.890 ± 0.033 | 0.016 ± 0.006 | 0.362 ± 0.073 | 0.845 ± 0.045 | 0.299 ± 0.095 | 0.984 ± 0.004 | 0.915 ± 0.016 | 138.200 ± 50.465 | 262.200 ± 110.638 |
| item1 | int8 | 0.890 ± 0.008 | 0.284 ± 0.057 | 0.984 ± 0.005 | 0.931 ± 0.019 | 0.520 ± 0.120 | 0.455 ± 0.038 | 0.903 ± 0.023 | 0.014 ± 0.003 | 0.456 ± 0.094 | 0.857 ± 0.032 | 0.333 ± 0.034 | 0.983 ± 0.002 | 0.923 ± 0.012 | 154.400 ± 28.571 | 290.800 ± 92.383 |
| item1 | float | 0.891 ± 0.008 | 0.285 ± 0.059 | 0.984 ± 0.005 | 0.932 ± 0.018 | 0.520 ± 0.120 | 0.455 ± 0.037 | 0.902 ± 0.023 | 0.014 ± 0.003 | 0.458 ± 0.096 | 0.858 ± 0.032 | 0.334 ± 0.034 | 0.983 ± 0.002 | 0.924 ± 0.012 | 155.600 ± 27.392 | 285.600 ± 89.888 |

## INT8 minus float, decision metrics

Per run, over seeds: mean ± std of (INT8 − float) and the largest |gap| of any seed.

| run | metric | int8_minus_float | max_abs_gap |
|---|---|---|---|
| rr_norm | S_roc_auc | 0.000 ± 0.003 | 0.004 |
| rr_norm | S_pr_auc | -0.001 ± 0.005 | 0.009 |
| rr_norm | V_roc_auc | -0.000 ± 0.000 | 0.000 |
| rr_norm | V_pr_auc | -0.000 ± 0.001 | 0.001 |
| item1 | S_roc_auc | -0.001 ± 0.001 | 0.001 |
| item1 | S_pr_auc | -0.000 ± 0.002 | 0.003 |
| item1 | V_roc_auc | -0.000 ± 0.000 | 0.000 |
| item1 | V_pr_auc | -0.001 ± 0.001 | 0.001 |

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
| rr_norm | 0.900 | 0.220 ± 0.039 | 0.901 ± 0.001 | 0.535 ± 0.092 | 3968.400 ± 2.510 | 3610.400 ± 1233.168 | 0.637 ± 0.027 | 0.961 ± 0.007 | 0.435 ± 0.044 | 0.865 ± 0.027 | 0.192 ± 0.052 | 0.568 ± 0.103 |
| rr_norm | 0.950 | 0.094 ± 0.020 | 0.952 ± 0.001 | 0.238 ± 0.057 | 4192.400 ± 3.209 | 14333.200 ± 4700.168 | 0.824 ± 0.043 | 0.981 ± 0.010 | 0.618 ± 0.057 | 0.877 ± 0.026 | 0.061 ± 0.023 | 0.352 ± 0.053 |
| item1 | 0.900 | 0.255 ± 0.055 | 0.900 ± 0.000 | 0.597 ± 0.033 | 3965.800 ± 1.483 | 2691.000 ± 362.382 | 0.643 ± 0.013 | 0.959 ± 0.003 | 0.418 ± 0.033 | 0.849 ± 0.020 | 0.160 ± 0.040 | 0.698 ± 0.080 |
| item1 | 0.950 | 0.116 ± 0.020 | 0.951 ± 0.000 | 0.304 ± 0.018 | 4188.400 ± 1.517 | 9627.000 ± 806.343 | 0.823 ± 0.030 | 0.980 ± 0.007 | 0.587 ± 0.054 | 0.858 ± 0.020 | 0.080 ± 0.019 | 0.412 ± 0.061 |

## Per seed

| run | seed | model | n_beats | threshold | S_roc_auc | S_pr_auc | S_recall | V_roc_auc | V_pr_auc | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
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
| item1 | 42 | int8 | 48585 | 0.4500 | 0.8949 | 0.3184 | 0.5031 | 0.9786 | 0.9242 | 0.9159 | 0.0175 | 0.4570 | 0.8622 | 0.3496 | 0.9845 | 0.9284 | 151 | 359 |
| item1 | 42 | float | 48585 | 0.4500 | 0.8963 | 0.3209 | 0.5031 | 0.9788 | 0.9254 | 0.9153 | 0.0176 | 0.4545 | 0.8627 | 0.3482 | 0.9846 | 0.9295 | 155 | 355 |
| item1 | 43 | int8 | 48585 | 0.4500 | 0.8761 | 0.3450 | 0.4540 | 0.9911 | 0.9529 | 0.9086 | 0.0094 | 0.3828 | 0.8065 | 0.2809 | 0.9851 | 0.9267 | 194 | 292 |
| item1 | 43 | float | 48585 | 0.4500 | 0.8775 | 0.3463 | 0.4564 | 0.9910 | 0.9532 | 0.9086 | 0.0093 | 0.3842 | 0.8075 | 0.2816 | 0.9852 | 0.9278 | 194 | 283 |
| item1 | 44 | int8 | 48585 | 0.6500 | 0.8917 | 0.2535 | 0.4613 | 0.9851 | 0.9083 | 0.8624 | 0.0126 | 0.5977 | 0.8532 | 0.3205 | 0.9847 | 0.9270 | 148 | 199 |
| item1 | 44 | float | 48585 | 0.6500 | 0.8920 | 0.2523 | 0.4613 | 0.9851 | 0.9093 | 0.8618 | 0.0128 | 0.6002 | 0.8540 | 0.3209 | 0.9847 | 0.9278 | 148 | 196 |
| item1 | 45 | int8 | 48585 | 0.6500 | 0.8980 | 0.2012 | 0.3963 | 0.9853 | 0.9226 | 0.9053 | 0.0159 | 0.4844 | 0.8715 | 0.3451 | 0.9818 | 0.9024 | 115 | 404 |
| item1 | 45 | float | 48585 | 0.6500 | 0.8986 | 0.1997 | 0.3988 | 0.9855 | 0.9225 | 0.9067 | 0.0164 | 0.4909 | 0.8716 | 0.3475 | 0.9822 | 0.9041 | 118 | 395 |
| item1 | 46 | int8 | 48585 | 0.4000 | 0.8889 | 0.3027 | 0.4589 | 0.9788 | 0.9474 | 0.9206 | 0.0147 | 0.3594 | 0.8938 | 0.3684 | 0.9805 | 0.9317 | 164 | 200 |
| item1 | 46 | float | 48585 | 0.4000 | 0.8901 | 0.3035 | 0.4564 | 0.9791 | 0.9479 | 0.9195 | 0.0146 | 0.3576 | 0.8955 | 0.3714 | 0.9808 | 0.9329 | 163 | 199 |

## Per record: S vs. N ROC-AUC, INT8 (records with >= 20 S beats)

| record_id | item1 s42 | item1 s43 | item1 s44 | item1 s45 | item1 s46 | rr_norm s42 | rr_norm s43 | rr_norm s44 | rr_norm s45 | rr_norm s46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 118 | 0.999 | 0.999 | 0.999 | 0.998 | 1.000 | 0.979 | 0.996 | 0.998 | 0.997 | 0.996 |
| 124 | 0.274 | 0.432 | 0.392 | 0.589 | 0.288 | 0.386 | 0.616 | 0.456 | 0.414 | 0.533 |
| 207 | 0.989 | 0.995 | 0.996 | 0.993 | 0.987 | 0.991 | 0.991 | 0.996 | 0.995 | 0.969 |
| 209 | 0.940 | 0.943 | 0.815 | 0.972 | 0.893 | 0.940 | 0.976 | 0.851 | 0.850 | 0.830 |
| 220 | 0.999 | 0.999 | 0.990 | 0.998 | 0.998 | 1.000 | 0.999 | 0.999 | 0.990 | 0.986 |
| 223 | 0.996 | 0.995 | 0.988 | 0.986 | 0.983 | 0.980 | 0.980 | 0.984 | 0.997 | 0.995 |
