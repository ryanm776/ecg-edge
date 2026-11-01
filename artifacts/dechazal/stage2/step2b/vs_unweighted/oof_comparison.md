# Pooled OOF per-class AUC

Runs: `nsv_unweighted` = `artifacts/dechazal/stage2/step2a_nsv_unweighted/cv`, `nsv_balanced` = `artifacts/dechazal/stage2/step2b_nsv_balanced/cv`. Reference: `nsv_unweighted`.
Seeds 42 43 44; mean ± sample std (ddof=1) over seeds. Scores are
`abnormal_score`; S vs. N and V vs. N leave the other class out. DS2 is not used.

## Decision: gain over the reference

`exceeds_spread`: |gain| > the larger of the two runs' std over seeds. `separated`: every candidate
seed is on the gain's side of every reference seed. The INT8 rows decide; float rows are shown alongside.

| run | model | metric | reference | candidate | gain | spread | exceeds_spread | separated |
|---|---|---|---|---|---|---|---|---|
| nsv_balanced | int8 | S_roc_auc | 0.824 ± 0.038 | 0.821 ± 0.046 | -0.003 | 0.046 | False | False |
| nsv_balanced | int8 | S_pr_auc | 0.144 ± 0.060 | 0.134 ± 0.039 | -0.010 | 0.060 | False | False |
| nsv_balanced | int8 | V_roc_auc | 0.969 ± 0.001 | 0.971 ± 0.006 | 0.002 | 0.006 | False | False |
| nsv_balanced | int8 | V_pr_auc | 0.883 ± 0.003 | 0.853 ± 0.027 | -0.030 | 0.027 | True | True |
| nsv_balanced | float | S_roc_auc | 0.823 ± 0.037 | 0.821 ± 0.046 | -0.002 | 0.046 | False | False |
| nsv_balanced | float | S_pr_auc | 0.143 ± 0.059 | 0.134 ± 0.038 | -0.009 | 0.059 | False | False |
| nsv_balanced | float | V_roc_auc | 0.969 ± 0.001 | 0.971 ± 0.007 | 0.002 | 0.007 | False | False |
| nsv_balanced | float | V_pr_auc | 0.885 ± 0.003 | 0.859 ± 0.024 | -0.025 | 0.024 | True | True |

## Mean ± std over seeds

Decision metrics first; `threshold` onward are descriptive, at each run's pooled-OOF INT8 threshold.
`*_pS` / `*_pV`: S vs. N AUC on P(S), V vs. N AUC on P(V); `S_as_V` / `V_as_S`: S beats called V and
V beats called S (N unless 1 - P(N) > threshold, else the likelier of S and V). N/S/V runs only (`—`
for binary); descriptive, never decisive.

| run | model | S_roc_auc | S_pr_auc | V_roc_auc | V_pr_auc | threshold | S_recall | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nsv_unweighted | int8 | 0.824 ± 0.038 | 0.144 ± 0.060 | 0.969 ± 0.001 | 0.883 ± 0.003 | 0.667 ± 0.058 | 0.273 ± 0.143 | 0.843 ± 0.023 | 0.013 ± 0.005 | 0.473 ± 0.135 | 0.752 ± 0.074 | 0.296 ± 0.087 | 0.971 ± 0.003 | 0.853 ± 0.022 | 111.000 ± 81.019 | 105.333 ± 41.789 |
| nsv_unweighted | float | 0.823 ± 0.037 | 0.143 ± 0.059 | 0.969 ± 0.001 | 0.885 ± 0.003 | 0.667 ± 0.058 | 0.273 ± 0.141 | 0.845 ± 0.023 | 0.014 ± 0.005 | 0.476 ± 0.128 | 0.753 ± 0.072 | 0.299 ± 0.083 | 0.971 ± 0.003 | 0.854 ± 0.021 | 112.000 ± 81.019 | 104.000 ± 42.036 |
| nsv_balanced | int8 | 0.821 ± 0.046 | 0.134 ± 0.039 | 0.971 ± 0.006 | 0.853 ± 0.027 | 0.800 ± 0.050 | 0.321 ± 0.086 | 0.844 ± 0.021 | 0.023 ± 0.005 | 0.624 ± 0.033 | 0.697 ± 0.041 | 0.297 ± 0.060 | 0.964 ± 0.006 | 0.817 ± 0.028 | 98.667 ± 80.823 | 233.000 ± 59.808 |
| nsv_balanced | float | 0.821 ± 0.046 | 0.134 ± 0.038 | 0.971 ± 0.007 | 0.859 ± 0.024 | 0.800 ± 0.050 | 0.322 ± 0.090 | 0.845 ± 0.022 | 0.023 ± 0.005 | 0.629 ± 0.030 | 0.697 ± 0.038 | 0.296 ± 0.059 | 0.965 ± 0.005 | 0.820 ± 0.024 | 101.000 ± 84.445 | 226.333 ± 50.797 |

## INT8 minus float, decision metrics

Per run, over seeds: mean ± std of (INT8 − float) and the largest |gap| of any seed.

| run | metric | int8_minus_float | max_abs_gap |
|---|---|---|---|
| nsv_unweighted | S_roc_auc | 0.001 ± 0.003 | 0.004 |
| nsv_unweighted | S_pr_auc | 0.001 ± 0.002 | 0.002 |
| nsv_unweighted | V_roc_auc | 0.000 ± 0.000 | 0.001 |
| nsv_unweighted | V_pr_auc | -0.001 ± 0.001 | 0.002 |
| nsv_balanced | S_roc_auc | -0.000 ± 0.003 | 0.003 |
| nsv_balanced | S_pr_auc | -0.000 ± 0.001 | 0.001 |
| nsv_balanced | V_roc_auc | 0.000 ± 0.001 | 0.001 |
| nsv_balanced | V_pr_auc | -0.006 ± 0.003 | 0.008 |

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
| nsv_unweighted | 0.900 | 0.297 ± 0.041 | 0.901 ± 0.000 | 0.384 ± 0.014 | 3967.333 ± 0.577 | 6368.000 ± 383.415 | 0.678 ± 0.030 | 0.951 ± 0.007 | 0.426 ± 0.036 | 0.892 ± 0.015 | 0.210 ± 0.155 | 0.399 ± 0.049 |
| nsv_unweighted | 0.950 | 0.151 ± 0.009 | 0.951 ± 0.001 | 0.191 ± 0.034 | 4188.667 ± 3.786 | 18264.000 ± 4432.891 | 0.853 ± 0.023 | 0.973 ± 0.005 | 0.593 ± 0.042 | 0.905 ± 0.017 | 0.048 ± 0.019 | 0.291 ± 0.033 |
| nsv_balanced | 0.900 | 0.451 ± 0.031 | 0.900 ± 0.001 | 0.345 ± 0.057 | 3966.333 ± 2.309 | 7741.667 ± 1940.911 | 0.665 ± 0.037 | 0.954 ± 0.009 | 0.455 ± 0.111 | 0.856 ± 0.027 | 0.094 ± 0.012 | 0.421 ± 0.121 |
| nsv_balanced | 0.950 | 0.312 ± 0.031 | 0.950 ± 0.000 | 0.200 ± 0.052 | 4186.667 ± 2.082 | 17739.667 ± 5771.827 | 0.825 ± 0.056 | 0.979 ± 0.013 | 0.584 ± 0.088 | 0.867 ± 0.033 | 0.041 ± 0.014 | 0.348 ± 0.094 |

## Per seed

| run | seed | model | n_beats | threshold | S_roc_auc | S_pr_auc | S_recall | V_roc_auc | V_pr_auc | V_recall | N_fp_rate | S_median_score | S_roc_auc_pS | S_pr_auc_pS | V_roc_auc_pV | V_pr_auc_pV | S_as_V | V_as_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nsv_unweighted | 42 | int8 | 48596 | 0.7000 | 0.8569 | 0.2131 | 0.4294 | 0.9689 | 0.8798 | 0.8454 | 0.0132 | 0.6094 | 0.7863 | 0.3926 | 0.9687 | 0.8314 | 193 | 88 |
| nsv_unweighted | 42 | float | 48596 | 0.7000 | 0.8573 | 0.2109 | 0.4282 | 0.9689 | 0.8820 | 0.8476 | 0.0133 | 0.6082 | 0.7870 | 0.3919 | 0.9687 | 0.8316 | 194 | 91 |
| nsv_unweighted | 43 | int8 | 48596 | 0.6000 | 0.8318 | 0.1039 | 0.2393 | 0.9706 | 0.8836 | 0.8655 | 0.0181 | 0.3398 | 0.8039 | 0.2250 | 0.9749 | 0.8745 | 109 | 75 |
| nsv_unweighted | 43 | float | 48596 | 0.6000 | 0.8277 | 0.1049 | 0.2368 | 0.9701 | 0.8841 | 0.8666 | 0.0183 | 0.3524 | 0.8014 | 0.2307 | 0.9747 | 0.8740 | 110 | 70 |
| nsv_unweighted | 44 | int8 | 48596 | 0.7000 | 0.7822 | 0.1153 | 0.1497 | 0.9680 | 0.8862 | 0.8187 | 0.0089 | 0.4688 | 0.6670 | 0.2714 | 0.9700 | 0.8536 | 31 | 153 |
| nsv_unweighted | 44 | float | 48596 | 0.7000 | 0.7837 | 0.1141 | 0.1534 | 0.9683 | 0.8874 | 0.8212 | 0.0091 | 0.4674 | 0.6701 | 0.2741 | 0.9703 | 0.8553 | 32 | 151 |
| nsv_balanced | 42 | int8 | 48596 | 0.7500 | 0.8076 | 0.1357 | 0.4209 | 0.9642 | 0.8256 | 0.8674 | 0.0264 | 0.6406 | 0.7152 | 0.3046 | 0.9650 | 0.8137 | 175 | 177 |
| nsv_balanced | 42 | float | 48596 | 0.7500 | 0.8042 | 0.1371 | 0.4258 | 0.9631 | 0.8340 | 0.8696 | 0.0262 | 0.6345 | 0.7129 | 0.3024 | 0.9649 | 0.8149 | 180 | 179 |
| nsv_balanced | 43 | int8 | 48596 | 0.8000 | 0.8717 | 0.1715 | 0.2761 | 0.9732 | 0.8787 | 0.8376 | 0.0171 | 0.5859 | 0.7266 | 0.3527 | 0.9700 | 0.8468 | 14 | 296 |
| nsv_balanced | 43 | float | 48596 | 0.8000 | 0.8728 | 0.1703 | 0.2663 | 0.9730 | 0.8815 | 0.8354 | 0.0168 | 0.5959 | 0.7241 | 0.3516 | 0.9697 | 0.8459 | 12 | 280 |
| nsv_balanced | 44 | int8 | 48596 | 0.8500 | 0.7830 | 0.0940 | 0.2675 | 0.9753 | 0.8555 | 0.8267 | 0.0251 | 0.6445 | 0.6501 | 0.2338 | 0.9585 | 0.7910 | 107 | 226 |
| nsv_balanced | 44 | float | 48596 | 0.8500 | 0.7855 | 0.0946 | 0.2736 | 0.9759 | 0.8616 | 0.8290 | 0.0252 | 0.6555 | 0.6535 | 0.2345 | 0.9590 | 0.7978 | 111 | 220 |

## Per record: S vs. N ROC-AUC, INT8 (records with >= 20 S beats)

| record_id | nsv_balanced s42 | nsv_balanced s43 | nsv_balanced s44 | nsv_unweighted s42 | nsv_unweighted s43 | nsv_unweighted s44 |
|---|---|---|---|---|---|---|
| 118 | 0.972 | 0.818 | 0.930 | 0.956 | 0.905 | 0.564 |
| 124 | 0.375 | 0.566 | 0.252 | 0.676 | 0.359 | 0.580 |
| 207 | 0.991 | 0.982 | 0.987 | 0.991 | 0.990 | 0.987 |
| 209 | 0.867 | 0.895 | 0.907 | 0.955 | 0.947 | 0.713 |
| 220 | 0.968 | 0.973 | 0.931 | 0.889 | 0.991 | 0.980 |
| 223 | 0.946 | 0.966 | 0.814 | 0.927 | 0.903 | 0.920 |
