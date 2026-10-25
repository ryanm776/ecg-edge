# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded | 100 | float | int8_val | 3 | 42 43 44 | 0.750 ± 0.011 | 0.689 ± 0.036 | 1416.0 ± 618.1 | 1665.0 ± 175.1 | 0.883 ± 0.106 | 0.298 ± 0.103 | 0.689 ± 0.034 | 1383.0 ± 569.1 | 1643.3 ± 157.5 | 0.750 ± 0.011 | 0.500 ± 0.000 | 23.3 ± 6.4 |
| bn_folded | 100 | float | float_val | 3 | 42 43 44 | 0.750 ± 0.011 | 0.689 ± 0.036 | 1416.0 ± 618.1 | 1665.0 ± 175.1 | 0.883 ± 0.106 | 0.298 ± 0.103 | 0.689 ± 0.034 | 1383.0 ± 569.1 | 1643.3 ± 157.5 | 0.750 ± 0.011 | 0.500 ± 0.000 | 23.3 ± 6.4 |
| bn_folded | 100 | int8 | int8_val | 3 | 42 43 44 | 0.748 ± 0.011 | 0.690 ± 0.036 | 1365.7 ± 597.8 | 1681.3 ± 177.1 | 0.881 ± 0.107 | 0.293 ± 0.101 | 0.691 ± 0.034 | 1333.7 ± 549.0 | 1659.0 ± 159.8 | 0.748 ± 0.011 | 0.500 ± 0.000 | 23.3 ± 6.4 |

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
| bn_folded_seed42 | bn_folded | 100 | 42 | 28 | float | int8_val | 0.500 | 0.736 | 0.709 | 0.635 | 0.670 | 3213 | 1316 | 1843 | 0.762 | — | 0.414 | — | 0.671 | 1315 | 1801 | 0.763 | 0.763 | 0.949 | 0.1171 | 0.180 | 0.995 | 22878 | — | — |
| bn_folded_seed43 | bn_folded | 100 | 43 | 16 | float | int8_val | 0.500 | 0.718 | 0.632 | 0.705 | 0.666 | 3563 | 2078 | 1493 | 0.957 | — | 0.263 | — | 0.668 | 1983 | 1486 | 0.743 | 0.743 | 0.922 | 0.0982 | 0.156 | 0.992 | 27131 | — | — |
| bn_folded_seed44 | bn_folded | 100 | 44 | 26 | float | int8_val | 0.500 | 0.741 | 0.799 | 0.672 | 0.730 | 3397 | 854 | 1659 | 0.931 | — | 0.217 | — | 0.728 | 851 | 1643 | 0.744 | 0.745 | 0.894 | 0.1091 | 0.189 | 0.880 | 19076 | — | — |
| bn_folded_seed42 | bn_folded | 100 | 42 | 28 | float | float_val | 0.500 | 0.736 | 0.709 | 0.635 | 0.670 | 3213 | 1316 | 1843 | 0.762 | — | 0.414 | — | 0.671 | 1315 | 1801 | 0.763 | 0.763 | 0.949 | 0.1171 | 0.180 | 0.995 | 22878 | — | — |
| bn_folded_seed43 | bn_folded | 100 | 43 | 16 | float | float_val | 0.500 | 0.718 | 0.632 | 0.705 | 0.666 | 3563 | 2078 | 1493 | 0.957 | — | 0.263 | — | 0.668 | 1983 | 1486 | 0.743 | 0.743 | 0.922 | 0.0982 | 0.156 | 0.992 | 27131 | — | — |
| bn_folded_seed44 | bn_folded | 100 | 44 | 26 | float | float_val | 0.500 | 0.741 | 0.799 | 0.672 | 0.730 | 3397 | 854 | 1659 | 0.931 | — | 0.217 | — | 0.728 | 851 | 1643 | 0.744 | 0.745 | 0.894 | 0.1091 | 0.189 | 0.880 | 19076 | — | — |
| bn_folded_seed42 | bn_folded | 100 | 42 | 28 | int8 | int8_val | 0.500 | 0.737 | 0.717 | 0.631 | 0.671 | 3190 | 1258 | 1866 | 0.758 | — | 0.408 | — | 0.672 | 1257 | 1824 | 0.761 | 0.761 | 0.949 | 0.1172 | 0.181 | 0.995 | 22776 | 207.98 | 136 |
| bn_folded_seed43 | bn_folded | 100 | 43 | 16 | int8 | int8_val | 0.500 | 0.716 | 0.638 | 0.701 | 0.668 | 3543 | 2010 | 1513 | 0.955 | — | 0.255 | — | 0.670 | 1917 | 1505 | 0.741 | 0.740 | 0.922 | 0.0977 | 0.157 | 0.992 | 26892 | 207.98 | 243 |
| bn_folded_seed44 | bn_folded | 100 | 44 | 26 | int8 | int8_val | 0.500 | 0.741 | 0.804 | 0.671 | 0.731 | 3391 | 829 | 1665 | 0.930 | — | 0.216 | — | 0.729 | 827 | 1648 | 0.742 | 0.743 | 0.895 | 0.1094 | 0.191 | 0.883 | 18921 | 207.98 | 174 |
