# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded | 100 | float | int8_val | 3 | 42 43 44 | 0.820 ± 0.007 | 0.754 ± 0.023 | 761.7 ± 277.8 | 1539.3 ± 146.5 | 0.906 ± 0.030 | 0.326 ± 0.040 | 0.757 ± 0.025 | 683.0 ± 230.1 | 1529.3 ± 147.5 | 0.822 ± 0.005 | 0.450 ± 0.000 | 18.0 ± 7.0 |
| bn_folded | 100 | float | float_val | 3 | 42 43 44 | 0.820 ± 0.007 | 0.754 ± 0.023 | 761.7 ± 277.8 | 1539.3 ± 146.5 | 0.906 ± 0.030 | 0.326 ± 0.040 | 0.757 ± 0.025 | 683.0 ± 230.1 | 1529.3 ± 147.5 | 0.822 ± 0.005 | 0.450 ± 0.000 | 18.0 ± 7.0 |
| bn_folded | 100 | int8 | int8_val | 3 | 42 43 44 | 0.817 ± 0.007 | 0.754 ± 0.023 | 758.3 ± 283.5 | 1535.7 ± 133.5 | 0.908 ± 0.027 | 0.326 ± 0.040 | 0.758 ± 0.024 | 677.7 ± 233.0 | 1525.7 ± 134.4 | 0.819 ± 0.005 | 0.450 ± 0.000 | 18.0 ± 7.0 |

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
| bn_folded_seed42 | bn_folded | 100 | 42 | 13 | float | int8_val | 0.450 | 0.765 | 0.822 | 0.664 | 0.735 | 3356 | 725 | 1699 | 0.872 | — | 0.299 | — | 0.732 | 714 | 1690 | 0.827 | 0.826 | 0.967 | 0.1062 | 0.213 | 0.992 | 18529 | — | — |
| bn_folded_seed43 | bn_folded | 100 | 43 | 15 | float | int8_val | 0.450 | 0.810 | 0.876 | 0.702 | 0.779 | 3547 | 504 | 1508 | 0.927 | — | 0.306 | — | 0.782 | 439 | 1498 | 0.823 | 0.821 | 0.937 | 0.1805 | 0.361 | 0.849 | 7591 | — | — |
| bn_folded_seed44 | bn_folded | 100 | 44 | 26 | float | int8_val | 0.450 | 0.833 | 0.775 | 0.721 | 0.747 | 3644 | 1056 | 1411 | 0.920 | — | 0.372 | — | 0.757 | 896 | 1400 | 0.817 | 0.813 | 0.957 | 0.0930 | 0.174 | 0.998 | 23906 | — | — |
| bn_folded_seed42 | bn_folded | 100 | 42 | 13 | float | float_val | 0.450 | 0.765 | 0.822 | 0.664 | 0.735 | 3356 | 725 | 1699 | 0.872 | — | 0.299 | — | 0.732 | 714 | 1690 | 0.827 | 0.826 | 0.967 | 0.1062 | 0.213 | 0.992 | 18529 | — | — |
| bn_folded_seed43 | bn_folded | 100 | 43 | 15 | float | float_val | 0.450 | 0.810 | 0.876 | 0.702 | 0.779 | 3547 | 504 | 1508 | 0.927 | — | 0.306 | — | 0.782 | 439 | 1498 | 0.823 | 0.821 | 0.937 | 0.1805 | 0.361 | 0.849 | 7591 | — | — |
| bn_folded_seed44 | bn_folded | 100 | 44 | 26 | float | float_val | 0.450 | 0.833 | 0.775 | 0.721 | 0.747 | 3644 | 1056 | 1411 | 0.920 | — | 0.372 | — | 0.757 | 896 | 1400 | 0.817 | 0.813 | 0.957 | 0.0930 | 0.174 | 0.998 | 23906 | — | — |
| bn_folded_seed42 | bn_folded | 100 | 42 | 13 | int8 | int8_val | 0.450 | 0.765 | 0.820 | 0.668 | 0.736 | 3376 | 739 | 1679 | 0.877 | — | 0.301 | — | 0.734 | 724 | 1670 | 0.822 | 0.821 | 0.966 | 0.1055 | 0.215 | 0.991 | 18339 | 207.98 | 226 |
| bn_folded_seed43 | bn_folded | 100 | 43 | 15 | int8 | int8_val | 0.450 | 0.809 | 0.880 | 0.701 | 0.780 | 3542 | 485 | 1513 | 0.927 | — | 0.304 | — | 0.783 | 425 | 1503 | 0.822 | 0.820 | 0.938 | 0.1836 | 0.381 | 0.843 | 6933 | 207.98 | 196 |
| bn_folded_seed44 | bn_folded | 100 | 44 | 26 | int8 | int8_val | 0.450 | 0.831 | 0.776 | 0.720 | 0.747 | 3640 | 1051 | 1415 | 0.919 | — | 0.371 | — | 0.758 | 884 | 1404 | 0.814 | 0.809 | 0.956 | 0.0937 | 0.181 | 0.997 | 22799 | 207.98 | 246 |
