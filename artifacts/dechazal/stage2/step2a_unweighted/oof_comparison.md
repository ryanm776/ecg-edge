# Pooled OOF per-class AUC

Runs: `binary_unweighted` = `artifacts/dechazal/stage2/step2a_binary_unweighted/cv`, `nsv_unweighted` = `artifacts/dechazal/stage2/step2a_nsv_unweighted/cv`. Reference: `binary_unweighted`.
Seeds 42 43 44; mean ± sample std (ddof=1) over seeds. Scores are
`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.

## Decision: gain over the reference

`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate
seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.

| run | model | metric | reference | candidate | gain | spread | exceeds_spread | separated |
|---|---|---|---|---|---|---|---|---|
| nsv_unweighted | int8 | S_roc_auc | 0.821 ± 0.049 | 0.824 ± 0.038 | 0.003 | 0.049 | False | False |
| nsv_unweighted | int8 | S_pr_auc | 0.213 ± 0.025 | 0.144 ± 0.060 | -0.069 | 0.060 | True | False |
| nsv_unweighted | int8 | V_roc_auc | 0.979 ± 0.010 | 0.969 ± 0.001 | -0.010 | 0.010 | True | False |
| nsv_unweighted | int8 | V_pr_auc | 0.921 ± 0.022 | 0.883 ± 0.003 | -0.038 | 0.022 | True | True |
| nsv_unweighted | float | S_roc_auc | 0.820 ± 0.050 | 0.823 ± 0.037 | 0.003 | 0.050 | False | False |
| nsv_unweighted | float | S_pr_auc | 0.219 ± 0.038 | 0.143 ± 0.059 | -0.076 | 0.059 | True | False |
| nsv_unweighted | float | V_roc_auc | 0.979 ± 0.010 | 0.969 ± 0.001 | -0.010 | 0.010 | True | False |
| nsv_unweighted | float | V_pr_auc | 0.921 ± 0.023 | 0.885 ± 0.003 | -0.036 | 0.023 | True | True |

## Mean ± std over seeds

Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.
`*_pS` / `*_pV`: S vs. N AUC on P(S), V vs. N AUC on P(V); `S_as_V` / `V_as_S`: S beats called V and
V beats called S (N unless 1 - P(N) > threshold, else the likelier of S and V). N/S/V runs only (`—`
for binary); descriptive, never decisive.

| run | model | S_roc_auc | S_pr_auc | V_roc_auc | V_pr_auc | threshold | S_recall | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_unweighted | int8 | 0.821 ± 0.049 | 0.213 ± 0.025 | 0.979 ± 0.010 | 0.921 ± 0.022 | 0.583 ± 0.076 | 0.307 ± 0.083 | 0.856 ± 0.045 | 0.010 ± 0.005 | 0.395 ± 0.041 | — | — | — | — | — | — |
| binary_unweighted | float | 0.820 ± 0.050 | 0.219 ± 0.038 | 0.979 ± 0.010 | 0.921 ± 0.023 | 0.583 ± 0.076 | 0.319 ± 0.083 | 0.855 ± 0.045 | 0.010 ± 0.005 | 0.407 ± 0.035 | — | — | — | — | — | — |
| nsv_unweighted | int8 | 0.824 ± 0.038 | 0.144 ± 0.060 | 0.969 ± 0.001 | 0.883 ± 0.003 | 0.667 ± 0.058 | 0.273 ± 0.143 | 0.843 ± 0.023 | 0.013 ± 0.005 | 0.473 ± 0.135 | 0.752 ± 0.074 | 0.296 ± 0.087 | 0.971 ± 0.003 | 0.853 ± 0.022 | 111.000 ± 81.019 | 105.333 ± 41.789 |
| nsv_unweighted | float | 0.823 ± 0.037 | 0.143 ± 0.059 | 0.969 ± 0.001 | 0.885 ± 0.003 | 0.667 ± 0.058 | 0.273 ± 0.141 | 0.845 ± 0.023 | 0.014 ± 0.005 | 0.476 ± 0.128 | 0.753 ± 0.072 | 0.299 ± 0.083 | 0.971 ± 0.003 | 0.854 ± 0.021 | 112.000 ± 81.019 | 104.000 ± 42.036 |

## INT8 minus float, decision metrics

Per run, over seeds: mean ± std of (INT8 − float) and the largest |gap| of any seed.

| run | metric | int8_minus_float | max_abs_gap |
|---|---|---|---|
| binary_unweighted | S_roc_auc | 0.001 ± 0.002 | 0.003 |
| binary_unweighted | S_pr_auc | -0.006 ± 0.014 | 0.021 |
| binary_unweighted | V_roc_auc | 0.000 ± 0.000 | 0.000 |
| binary_unweighted | V_pr_auc | 0.000 ± 0.001 | 0.001 |
| nsv_unweighted | S_roc_auc | 0.001 ± 0.003 | 0.004 |
| nsv_unweighted | S_pr_auc | 0.001 ± 0.002 | 0.002 |
| nsv_unweighted | V_roc_auc | 0.000 ± 0.000 | 0.001 |
| nsv_unweighted | V_pr_auc | -0.001 ± 0.001 | 0.002 |

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
| binary_unweighted | 0.900 | 0.283 ± 0.047 | 0.901 ± 0.001 | 0.370 ± 0.111 | 3970.333 ± 2.309 | 7539.667 ± 4007.562 | 0.630 ± 0.039 | 0.963 ± 0.008 | — | — | — | — |
| binary_unweighted | 0.950 | 0.177 ± 0.058 | 0.952 ± 0.002 | 0.211 ± 0.077 | 4194.000 ± 8.000 | 17417.333 ± 7233.588 | 0.813 ± 0.041 | 0.984 ± 0.009 | — | — | — | — |
| nsv_unweighted | 0.900 | 0.297 ± 0.041 | 0.901 ± 0.000 | 0.384 ± 0.014 | 3967.333 ± 0.577 | 6368.000 ± 383.415 | 0.678 ± 0.030 | 0.951 ± 0.007 | 0.426 ± 0.036 | 0.892 ± 0.015 | 0.210 ± 0.155 | 0.399 ± 0.049 |
| nsv_unweighted | 0.950 | 0.151 ± 0.009 | 0.951 ± 0.001 | 0.191 ± 0.034 | 4188.667 ± 3.786 | 18264.000 ± 4432.891 | 0.853 ± 0.023 | 0.973 ± 0.005 | 0.593 ± 0.042 | 0.905 ± 0.017 | 0.048 ± 0.019 | 0.291 ± 0.033 |

## Per seed

| run | seed | model | n_beats | threshold | S_roc_auc | S_pr_auc | S_recall | V_roc_auc | V_pr_auc | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_unweighted | 42 | int8 | 48596 | 0.6500 | 0.8210 | 0.2006 | 0.2233 | 0.9684 | 0.8954 | 0.8100 | 0.0072 | 0.4414 | — | — | — | — | — | — |
| binary_unweighted | 42 | float | 48596 | 0.6500 | 0.8205 | 0.1960 | 0.2282 | 0.9679 | 0.8946 | 0.8092 | 0.0074 | 0.4420 | — | — | — | — | — | — |
| binary_unweighted | 43 | int8 | 48596 | 0.6000 | 0.8688 | 0.2421 | 0.3080 | 0.9874 | 0.9359 | 0.8577 | 0.0069 | 0.3711 | — | — | — | — | — | — |
| binary_unweighted | 43 | float | 48596 | 0.6000 | 0.8697 | 0.2635 | 0.3387 | 0.9870 | 0.9354 | 0.8563 | 0.0070 | 0.4050 | — | — | — | — | — | — |
| binary_unweighted | 44 | int8 | 48596 | 0.5000 | 0.7717 | 0.1964 | 0.3890 | 0.9817 | 0.9320 | 0.8994 | 0.0149 | 0.3711 | — | — | — | — | — | — |
| binary_unweighted | 44 | float | 48596 | 0.5000 | 0.7688 | 0.1983 | 0.3902 | 0.9821 | 0.9325 | 0.8997 | 0.0150 | 0.3727 | — | — | — | — | — | — |
| nsv_unweighted | 42 | int8 | 48596 | 0.7000 | 0.8569 | 0.2131 | 0.4294 | 0.9689 | 0.8798 | 0.8454 | 0.0132 | 0.6094 | 0.7863 | 0.3926 | 0.9687 | 0.8314 | 193 | 88 |
| nsv_unweighted | 42 | float | 48596 | 0.7000 | 0.8573 | 0.2109 | 0.4282 | 0.9689 | 0.8820 | 0.8476 | 0.0133 | 0.6082 | 0.7870 | 0.3919 | 0.9687 | 0.8316 | 194 | 91 |
| nsv_unweighted | 43 | int8 | 48596 | 0.6000 | 0.8318 | 0.1039 | 0.2393 | 0.9706 | 0.8836 | 0.8655 | 0.0181 | 0.3398 | 0.8039 | 0.2250 | 0.9749 | 0.8745 | 109 | 75 |
| nsv_unweighted | 43 | float | 48596 | 0.6000 | 0.8277 | 0.1049 | 0.2368 | 0.9701 | 0.8841 | 0.8666 | 0.0183 | 0.3524 | 0.8014 | 0.2307 | 0.9747 | 0.8740 | 110 | 70 |
| nsv_unweighted | 44 | int8 | 48596 | 0.7000 | 0.7822 | 0.1153 | 0.1497 | 0.9680 | 0.8862 | 0.8187 | 0.0089 | 0.4688 | 0.6670 | 0.2714 | 0.9700 | 0.8536 | 31 | 153 |
| nsv_unweighted | 44 | float | 48596 | 0.7000 | 0.7837 | 0.1141 | 0.1534 | 0.9683 | 0.8874 | 0.8212 | 0.0091 | 0.4674 | 0.6701 | 0.2741 | 0.9703 | 0.8553 | 32 | 151 |

## Per record: S vs. N ROC-AUC, INT8 (records with >= 20 S beats)

| record_id | binary_unweighted s42 | binary_unweighted s43 | binary_unweighted s44 | nsv_unweighted s42 | nsv_unweighted s43 | nsv_unweighted s44 |
|---|---|---|---|---|---|---|
| 118 | 0.960 | 0.947 | 0.995 | 0.956 | 0.905 | 0.564 |
| 124 | 0.528 | 0.580 | 0.404 | 0.676 | 0.359 | 0.580 |
| 207 | 0.994 | 0.994 | 0.991 | 0.991 | 0.990 | 0.987 |
| 209 | 0.736 | 0.860 | 0.827 | 0.955 | 0.947 | 0.713 |
| 220 | 0.914 | 0.990 | 0.886 | 0.889 | 0.991 | 0.980 |
| 223 | 0.963 | 0.995 | 0.931 | 0.927 | 0.903 | 0.920 |
