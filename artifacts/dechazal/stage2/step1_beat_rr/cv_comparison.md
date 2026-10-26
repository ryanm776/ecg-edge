# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded | 100 | float | int8_val | 3 | 42 43 44 | 0.826 ± 0.087 | 0.635 ± 0.178 | 4422.7 ± 7316.2 | 1370.0 ± 1081.8 | 0.896 ± 0.094 | 0.436 ± 0.471 | 0.638 ± 0.172 | 4196.0 ± 6923.7 | 1346.7 ± 1061.9 | 0.827 ± 0.089 | 0.483 ± 0.076 | 17.0 ± 5.3 |
| bn_folded | 100 | float | float_val | 3 | 42 43 44 | 0.826 ± 0.087 | 0.635 ± 0.178 | 4422.7 ± 7316.2 | 1370.0 ± 1081.8 | 0.896 ± 0.094 | 0.436 ± 0.471 | 0.638 ± 0.172 | 4196.0 ± 6923.7 | 1346.7 ± 1061.9 | 0.827 ± 0.089 | 0.483 ± 0.076 | 17.0 ± 5.3 |
| bn_folded | 100 | int8 | int8_val | 3 | 42 43 44 | 0.824 ± 0.087 | 0.633 ± 0.177 | 4437.7 ± 7344.0 | 1384.0 ± 1086.5 | 0.894 ± 0.098 | 0.432 ± 0.471 | 0.636 ± 0.172 | 4211.3 ± 6952.0 | 1360.3 ± 1067.1 | 0.825 ± 0.089 | 0.483 ± 0.076 | 17.0 ± 5.3 |

## Per run

DS2, one row per run (seed in `seed`), final model trained on all 22 DS1 records for `final_epochs`
(median of the 5 folds' best epochs). `threshold_from`: `int8_val` is the threshold chosen on the
pooled out-of-fold INT8 scores (the deployed convention); `float_val` is the float model's own pooled-OOF
threshold. `val_f1` is the pooled-OOF F1 at the row's threshold. `r95_*`: the highest threshold reaching
>= 0.95 pooled-OOF recall, then DS2 precision / recall / FP there. Record 201 is never held out, so it
is not in the OOF pool.
`se_V` / `se_S`: AAMI sensitivity for V (V, E) and S (A, a, J, S). `pp_V` / `pp_S` (+P) are undefined
for a binary model (an abnormal prediction names neither class) and show `—`.
`*_excl_202`: sensitivity check, DS2 without record 202 (same patient as DS1 record 201);
reported alongside the full-DS2 columns, not instead of them.
Per-fold OOF results and a diagnostic DS2 evaluation of each fold model are in each run's folds.json.

| run | variant | calib_samples | seed | final_epochs | model | threshold_from | threshold | val_f1 | test_precision | test_recall | test_f1 | tp | fp | fn | se_V | pp_V | se_S | pp_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | pr_auc | roc_auc | r95_threshold | r95_test_precision | r95_test_recall | r95_fp | tflite_kb | min_int8_levels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded_seed42 | bn_folded | 100 | 42 | 15 | float | int8_val | 0.500 | 0.761 | 0.965 | 0.574 | 0.720 | 2900 | 104 | 2155 | 0.791 | — | 0.193 | — | 0.723 | 104 | 2105 | 0.929 | 0.926 | 0.986 | 0.1257 | 0.619 | 0.950 | 2953 | — | — |
| bn_folded_seed43 | bn_folded | 100 | 43 | 13 | float | int8_val | 0.550 | 0.775 | 0.917 | 0.640 | 0.754 | 3236 | 294 | 1819 | 0.928 | — | 0.136 | — | 0.752 | 294 | 1802 | 0.771 | 0.771 | 0.890 | 0.0995 | 0.184 | 0.871 | 19480 | — | — |
| bn_folded_seed44 | bn_folded | 100 | 44 | 23 | float | int8_val | 0.400 | 0.801 | 0.277 | 0.973 | 0.431 | 4919 | 12870 | 136 | 0.970 | — | 0.979 | — | 0.440 | 12190 | 133 | 0.780 | 0.781 | 0.953 | 0.0927 | 0.109 | 1.000 | 41188 | — | — |
| bn_folded_seed42 | bn_folded | 100 | 42 | 15 | float | float_val | 0.500 | 0.761 | 0.965 | 0.574 | 0.720 | 2900 | 104 | 2155 | 0.791 | — | 0.193 | — | 0.723 | 104 | 2105 | 0.929 | 0.926 | 0.986 | 0.1257 | 0.619 | 0.950 | 2953 | — | — |
| bn_folded_seed43 | bn_folded | 100 | 43 | 13 | float | float_val | 0.550 | 0.775 | 0.917 | 0.640 | 0.754 | 3236 | 294 | 1819 | 0.928 | — | 0.136 | — | 0.752 | 294 | 1802 | 0.771 | 0.771 | 0.890 | 0.0995 | 0.184 | 0.871 | 19480 | — | — |
| bn_folded_seed44 | bn_folded | 100 | 44 | 23 | float | float_val | 0.400 | 0.801 | 0.277 | 0.973 | 0.431 | 4919 | 12870 | 136 | 0.970 | — | 0.979 | — | 0.440 | 12190 | 133 | 0.780 | 0.781 | 0.953 | 0.0927 | 0.109 | 1.000 | 41188 | — | — |
| bn_folded_seed42 | bn_folded | 100 | 42 | 15 | int8 | int8_val | 0.500 | 0.759 | 0.968 | 0.568 | 0.716 | 2871 | 96 | 2184 | 0.784 | — | 0.190 | — | 0.719 | 96 | 2134 | 0.928 | 0.925 | 0.986 | 0.1289 | 0.645 | 0.949 | 2643 | 207.98 | 225 |
| bn_folded_seed43 | bn_folded | 100 | 43 | 13 | int8 | int8_val | 0.550 | 0.771 | 0.915 | 0.640 | 0.753 | 3234 | 300 | 1821 | 0.929 | — | 0.132 | — | 0.751 | 300 | 1804 | 0.770 | 0.770 | 0.891 | 0.0977 | 0.183 | 0.877 | 19817 | 207.98 | 217 |
| bn_folded_seed44 | bn_folded | 100 | 44 | 23 | int8 | int8_val | 0.400 | 0.804 | 0.275 | 0.971 | 0.429 | 4908 | 12917 | 147 | 0.969 | — | 0.974 | — | 0.439 | 12238 | 143 | 0.776 | 0.777 | 0.953 | 0.0937 | 0.110 | 1.000 | 41039 | 207.98 | 194 |
