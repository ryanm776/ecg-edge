# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | pp_V | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded+nsv+rr_norm | 100 | float | int8_val | 5 | 42 43 44 45 46 | 0.867 ± 0.032 | 0.770 ± 0.039 | 1317.0 ± 1006.6 | 1072.8 ± 550.2 | 0.840 ± 0.078 | 0.326 ± 0.226 | 0.812 ± 0.138 | 0.349 ± 0.101 | 319.8 ± 268.3 | 360.4 ± 304.9 | 0.774 ± 0.035 | 1233.4 ± 920.7 | 1061.0 ± 543.3 | 0.869 ± 0.031 | 0.520 ± 0.057 | 17.4 ± 2.4 |
| bn_folded+nsv+rr_norm | 100 | float | float_val | 5 | 42 43 44 45 46 | 0.867 ± 0.032 | 0.773 ± 0.037 | 1278.6 ± 1019.8 | 1081.4 ± 489.9 | 0.842 ± 0.079 | 0.313 ± 0.194 | 0.811 ± 0.140 | 0.359 ± 0.102 | 327.0 ± 270.9 | 362.2 ± 302.2 | 0.777 ± 0.033 | 1197.6 ± 931.5 | 1069.8 ± 483.4 | 0.869 ± 0.031 | 0.520 ± 0.057 | 17.4 ± 2.4 |
| bn_folded+nsv+rr_norm | 100 | int8 | int8_val | 5 | 42 43 44 45 46 | 0.867 ± 0.033 | 0.771 ± 0.038 | 1320.6 ± 1012.7 | 1064.8 ± 543.6 | 0.839 ± 0.079 | 0.329 ± 0.225 | 0.811 ± 0.138 | 0.355 ± 0.100 | 321.2 ± 269.8 | 363.8 ± 307.7 | 0.775 ± 0.035 | 1238.0 ± 925.9 | 1052.8 ± 536.5 | 0.868 ± 0.032 | 0.520 ± 0.057 | 17.4 ± 2.4 |

## Per run

DS2, one row per run (seed in `seed`), final model trained on all 22 DS1 records for `final_epochs`
(median of the 5 folds' best epochs). `threshold_from`: `int8_val` is the threshold chosen on the
pooled out-of-fold INT8 scores (the deployed convention); `float_val` is the float model's own pooled-OOF
threshold. `val_f1` is the pooled-OOF F1 at the row's threshold. `r95_*`: the highest threshold reaching
>= 0.95 pooled-OOF recall, then DS2 precision / recall / FP there. Record 201 is never held out, so it
is not in the OOF pool.
`se_V` / `se_S`: AAMI sensitivity for V (V, E) and S (A, a, J, S). `pp_V` / `pp_S` (+P) are undefined
for a binary model (an abnormal prediction names neither class) and show `—`. For an N/S/V model
(`+nsv` in `variant`) a beat is called N unless 1 - P(N) > threshold, else the likelier of S and V;
Se and +P count that call, so an S beat called V is an S miss. `S_as_V` / `V_as_S` count those
mix-ups (`—` for binary). `pr_auc`, `test_*`, `tp`/`fp`/`fn` are N vs. abnormal for every model.
N/S/V models drop F and Q beats, like the binary ones.
`*_excl_202`: sensitivity check, DS2 without record 202 (same patient as DS1 record 201);
reported alongside the full-DS2 columns, not instead of them.
Per-fold OOF results and a diagnostic DS2 evaluation of each fold model are in each run's folds.json.

| run | variant | calib_samples | seed | final_epochs | model | threshold_from | threshold | val_f1 | test_precision | test_recall | test_f1 | tp | fp | fn | se_V | pp_V | se_S | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | pr_auc | roc_auc | r95_threshold | r95_test_precision | r95_test_recall | r95_fp | tflite_kb | min_int8_levels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded_seed42 | bn_folded+nsv+rr_norm | 100 | 42 | 18 | float | int8_val | 0.500 | 0.800 | 0.704 | 0.760 | 0.731 | 3843 | 1617 | 1212 | 0.757 | 0.893 | 0.366 | 0.246 | 74 | 661 | 0.741 | 1426 | 1208 | 0.829 | 0.826 | 0.933 | 0.0988 | 0.250 | 0.928 | 14034 | — | — |
| bn_folded_seed43 | bn_folded+nsv+rr_norm | 100 | 43 | 19 | float | int8_val | 0.500 | 0.815 | 0.620 | 0.940 | 0.747 | 4751 | 2918 | 304 | 0.776 | 0.601 | 0.497 | 0.260 | 668 | 671 | 0.756 | 2727 | 301 | 0.891 | 0.891 | 0.980 | 0.0964 | 0.171 | 0.998 | 24431 | — | — |
| bn_folded_seed44 | bn_folded+nsv+rr_norm | 100 | 44 | 20 | float | int8_val | 0.600 | 0.811 | 0.897 | 0.651 | 0.754 | 3289 | 379 | 1766 | 0.844 | 0.843 | 0.081 | 0.332 | 353 | 71 | 0.753 | 379 | 1745 | 0.877 | 0.875 | 0.964 | 0.0587 | 0.301 | 0.964 | 11326 | — | — |
| bn_folded_seed45 | bn_folded+nsv+rr_norm | 100 | 45 | 14 | float | int8_val | 0.450 | 0.831 | 0.850 | 0.746 | 0.794 | 3771 | 668 | 1284 | 0.953 | 0.763 | 0.104 | 0.454 | 471 | 42 | 0.794 | 664 | 1261 | 0.844 | 0.844 | 0.946 | 0.1064 | 0.257 | 0.940 | 13709 | — | — |
| bn_folded_seed46 | bn_folded+nsv+rr_norm | 100 | 46 | 16 | float | int8_val | 0.550 | 0.811 | 0.809 | 0.842 | 0.825 | 4257 | 1003 | 798 | 0.870 | 0.961 | 0.580 | 0.455 | 33 | 357 | 0.826 | 971 | 790 | 0.902 | 0.902 | 0.976 | 0.1081 | 0.179 | 0.992 | 23069 | — | — |
| bn_folded_seed42 | bn_folded+nsv+rr_norm | 100 | 42 | 18 | float | float_val | 0.500 | 0.800 | 0.704 | 0.760 | 0.731 | 3843 | 1617 | 1212 | 0.757 | 0.893 | 0.366 | 0.246 | 74 | 661 | 0.741 | 1426 | 1208 | 0.829 | 0.826 | 0.933 | 0.0988 | 0.250 | 0.928 | 14034 | — | — |
| bn_folded_seed43 | bn_folded+nsv+rr_norm | 100 | 43 | 19 | float | float_val | 0.500 | 0.815 | 0.620 | 0.940 | 0.747 | 4751 | 2918 | 304 | 0.776 | 0.601 | 0.497 | 0.260 | 668 | 671 | 0.756 | 2727 | 301 | 0.891 | 0.891 | 0.980 | 0.0964 | 0.171 | 0.998 | 24431 | — | — |
| bn_folded_seed44 | bn_folded+nsv+rr_norm | 100 | 44 | 20 | float | float_val | 0.550 | 0.811 | 0.887 | 0.680 | 0.770 | 3436 | 436 | 1619 | 0.856 | 0.829 | 0.112 | 0.377 | 392 | 83 | 0.769 | 435 | 1599 | 0.877 | 0.875 | 0.964 | 0.0587 | 0.301 | 0.964 | 11326 | — | — |
| bn_folded_seed45 | bn_folded+nsv+rr_norm | 100 | 45 | 14 | float | float_val | 0.450 | 0.831 | 0.850 | 0.746 | 0.794 | 3771 | 668 | 1284 | 0.953 | 0.763 | 0.104 | 0.454 | 471 | 42 | 0.794 | 664 | 1261 | 0.844 | 0.844 | 0.946 | 0.1064 | 0.257 | 0.940 | 13709 | — | — |
| bn_folded_seed46 | bn_folded+nsv+rr_norm | 100 | 46 | 16 | float | float_val | 0.600 | 0.811 | 0.844 | 0.805 | 0.824 | 4067 | 754 | 988 | 0.868 | 0.968 | 0.484 | 0.459 | 30 | 354 | 0.823 | 736 | 980 | 0.902 | 0.902 | 0.976 | 0.1081 | 0.179 | 0.992 | 23069 | — | — |
| bn_folded_seed42 | bn_folded+nsv+rr_norm | 100 | 42 | 18 | int8 | int8_val | 0.500 | 0.800 | 0.704 | 0.758 | 0.730 | 3833 | 1609 | 1222 | 0.754 | 0.892 | 0.367 | 0.248 | 71 | 662 | 0.740 | 1426 | 1218 | 0.826 | 0.823 | 0.932 | 0.0977 | 0.252 | 0.925 | 13878 | 208.17 | 221 |
| bn_folded_seed43 | bn_folded+nsv+rr_norm | 100 | 43 | 19 | int8 | int8_val | 0.500 | 0.815 | 0.618 | 0.942 | 0.747 | 4760 | 2937 | 295 | 0.774 | 0.600 | 0.499 | 0.259 | 672 | 680 | 0.756 | 2742 | 292 | 0.892 | 0.892 | 0.980 | 0.0977 | 0.175 | 0.998 | 23811 | 208.17 | 251 |
| bn_folded_seed44 | bn_folded+nsv+rr_norm | 100 | 44 | 20 | int8 | int8_val | 0.600 | 0.811 | 0.897 | 0.658 | 0.759 | 3327 | 382 | 1728 | 0.848 | 0.840 | 0.092 | 0.366 | 362 | 68 | 0.758 | 382 | 1707 | 0.875 | 0.873 | 0.964 | 0.0586 | 0.302 | 0.961 | 11255 | 208.17 | 198 |
| bn_folded_seed45 | bn_folded+nsv+rr_norm | 100 | 45 | 14 | int8 | int8_val | 0.450 | 0.833 | 0.850 | 0.745 | 0.794 | 3767 | 666 | 1288 | 0.952 | 0.765 | 0.105 | 0.450 | 467 | 44 | 0.794 | 663 | 1264 | 0.844 | 0.844 | 0.947 | 0.1055 | 0.259 | 0.941 | 13634 | 208.17 | 215 |
| bn_folded_seed46 | bn_folded+nsv+rr_norm | 100 | 46 | 16 | int8 | int8_val | 0.550 | 0.813 | 0.809 | 0.844 | 0.826 | 4264 | 1009 | 791 | 0.868 | 0.960 | 0.583 | 0.453 | 34 | 365 | 0.827 | 977 | 783 | 0.901 | 0.901 | 0.976 | 0.1094 | 0.183 | 0.992 | 22463 | 208.17 | 166 |
