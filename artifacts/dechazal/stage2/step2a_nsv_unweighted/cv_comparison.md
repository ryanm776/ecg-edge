# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | pp_V | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded+nsv+unweighted | 100 | float | int8_val | 3 | 42 43 44 | 0.791 ± 0.083 | 0.698 ± 0.099 | 1978.3 ± 1277.8 | 1327.0 ± 423.9 | 0.887 ± 0.076 | 0.235 ± 0.194 | 0.653 ± 0.162 | 0.354 ± 0.139 | 272.3 ± 50.6 | 171.0 ± 182.1 | 0.699 ± 0.102 | 1925.7 ± 1278.1 | 1317.7 ± 422.1 | 0.791 ± 0.086 | 0.667 ± 0.058 | 16.7 ± 9.8 |
| bn_folded+nsv+unweighted | 100 | float | float_val | 3 | 42 43 44 | 0.791 ± 0.083 | 0.698 ± 0.099 | 1978.3 ± 1277.8 | 1327.0 ± 423.9 | 0.887 ± 0.076 | 0.235 ± 0.194 | 0.653 ± 0.162 | 0.354 ± 0.139 | 272.3 ± 50.6 | 171.0 ± 182.1 | 0.699 ± 0.102 | 1925.7 ± 1278.1 | 1317.7 ± 422.1 | 0.791 ± 0.086 | 0.667 ± 0.058 | 16.7 ± 9.8 |
| bn_folded+nsv+unweighted | 100 | int8 | int8_val | 3 | 42 43 44 | 0.789 ± 0.083 | 0.698 ± 0.097 | 1962.3 ± 1279.6 | 1340.7 ± 388.3 | 0.887 ± 0.076 | 0.225 ± 0.178 | 0.653 ± 0.161 | 0.354 ± 0.137 | 274.0 ± 57.0 | 171.7 ± 178.9 | 0.698 ± 0.101 | 1913.3 ± 1282.1 | 1331.3 ± 386.5 | 0.789 ± 0.085 | 0.667 ± 0.058 | 16.7 ± 9.8 |

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
| bn_folded_seed42 | bn_folded+nsv+unweighted | 100 | 42 | 28 | float | int8_val | 0.700 | 0.809 | 0.838 | 0.701 | 0.763 | 3543 | 684 | 1512 | 0.807 | 0.802 | 0.129 | 0.240 | 327 | 381 | 0.762 | 679 | 1500 | 0.810 | 0.810 | 0.918 | 0.1580 | 0.279 | 0.888 | 11588 | — | — |
| bn_folded_seed43 | bn_folded+nsv+unweighted | 100 | 43 | 11 | float | int8_val | 0.600 | 0.776 | 0.677 | 0.833 | 0.747 | 4213 | 2012 | 842 | 0.959 | 0.675 | 0.459 | 0.509 | 227 | 57 | 0.754 | 1865 | 835 | 0.865 | 0.863 | 0.970 | 0.1519 | 0.171 | 0.993 | 24411 | — | — |
| bn_folded_seed44 | bn_folded+nsv+unweighted | 100 | 44 | 11 | float | int8_val | 0.700 | 0.780 | 0.514 | 0.678 | 0.585 | 3428 | 3239 | 1627 | 0.893 | 0.480 | 0.117 | 0.314 | 263 | 75 | 0.581 | 3233 | 1618 | 0.697 | 0.700 | 0.875 | 0.1396 | 0.142 | 0.987 | 30092 | — | — |
| bn_folded_seed42 | bn_folded+nsv+unweighted | 100 | 42 | 28 | float | float_val | 0.700 | 0.809 | 0.838 | 0.701 | 0.763 | 3543 | 684 | 1512 | 0.807 | 0.802 | 0.129 | 0.240 | 327 | 381 | 0.762 | 679 | 1500 | 0.810 | 0.810 | 0.918 | 0.1580 | 0.279 | 0.888 | 11588 | — | — |
| bn_folded_seed43 | bn_folded+nsv+unweighted | 100 | 43 | 11 | float | float_val | 0.600 | 0.776 | 0.677 | 0.833 | 0.747 | 4213 | 2012 | 842 | 0.959 | 0.675 | 0.459 | 0.509 | 227 | 57 | 0.754 | 1865 | 835 | 0.865 | 0.863 | 0.970 | 0.1519 | 0.171 | 0.993 | 24411 | — | — |
| bn_folded_seed44 | bn_folded+nsv+unweighted | 100 | 44 | 11 | float | float_val | 0.700 | 0.780 | 0.514 | 0.678 | 0.585 | 3428 | 3239 | 1627 | 0.893 | 0.480 | 0.117 | 0.314 | 263 | 75 | 0.581 | 3233 | 1618 | 0.697 | 0.700 | 0.875 | 0.1396 | 0.142 | 0.987 | 30092 | — | — |
| bn_folded_seed42 | bn_folded+nsv+unweighted | 100 | 42 | 28 | int8 | int8_val | 0.700 | 0.809 | 0.837 | 0.701 | 0.763 | 3542 | 691 | 1513 | 0.808 | 0.798 | 0.124 | 0.234 | 336 | 378 | 0.761 | 686 | 1501 | 0.807 | 0.808 | 0.918 | 0.1562 | 0.273 | 0.889 | 11982 | 208.17 | 254 |
| bn_folded_seed43 | bn_folded+nsv+unweighted | 100 | 43 | 11 | int8 | int8_val | 0.600 | 0.776 | 0.681 | 0.823 | 0.745 | 4159 | 1946 | 896 | 0.958 | 0.681 | 0.431 | 0.502 | 224 | 60 | 0.752 | 1810 | 889 | 0.863 | 0.861 | 0.970 | 0.1562 | 0.176 | 0.992 | 23529 | 208.17 | 255 |
| bn_folded_seed44 | bn_folded+nsv+unweighted | 100 | 44 | 11 | int8 | int8_val | 0.700 | 0.779 | 0.514 | 0.681 | 0.586 | 3442 | 3250 | 1613 | 0.895 | 0.479 | 0.121 | 0.326 | 262 | 77 | 0.582 | 3244 | 1604 | 0.696 | 0.699 | 0.876 | 0.1406 | 0.144 | 0.986 | 29712 | 208.17 | 245 |
