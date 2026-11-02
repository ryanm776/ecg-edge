# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | pp_V | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded+nsv | 100 | float | int8_val | 5 | 42 43 44 45 46 | 0.765 ± 0.045 | 0.667 ± 0.050 | 1915.6 ± 1292.8 | 1602.8 ± 379.5 | 0.888 ± 0.076 | 0.165 ± 0.112 | 0.732 ± 0.193 | 0.298 ± 0.071 | 168.2 ± 96.5 | 123.8 ± 180.4 | 0.667 ± 0.050 | 1878.8 ± 1257.8 | 1586.6 ± 364.5 | 0.764 ± 0.045 | 0.570 ± 0.057 | 14.4 ± 3.5 |
| bn_folded+nsv | 100 | float | float_val | 5 | 42 43 44 45 46 | 0.765 ± 0.045 | 0.666 ± 0.067 | 2009.4 ± 1735.7 | 1599.2 ± 391.8 | 0.888 ± 0.075 | 0.168 ± 0.116 | 0.738 ± 0.204 | 0.293 ± 0.065 | 163.8 ± 92.7 | 124.2 ± 180.5 | 0.666 ± 0.066 | 1970.6 ± 1690.9 | 1582.4 ± 376.8 | 0.764 ± 0.045 | 0.570 ± 0.084 | 14.4 ± 3.5 |
| bn_folded+nsv | 100 | int8 | int8_val | 5 | 42 43 44 45 46 | 0.763 ± 0.045 | 0.669 ± 0.048 | 1882.0 ± 1264.8 | 1606.8 ± 381.8 | 0.887 ± 0.076 | 0.165 ± 0.112 | 0.737 ± 0.190 | 0.292 ± 0.063 | 161.8 ± 92.5 | 128.0 ± 182.0 | 0.668 ± 0.048 | 1847.4 ± 1231.9 | 1590.0 ± 366.5 | 0.762 ± 0.045 | 0.570 ± 0.057 | 14.4 ± 3.5 |

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
| bn_folded_seed42 | bn_folded+nsv | 100 | 42 | 12 | float | int8_val | 0.650 | 0.781 | 0.961 | 0.550 | 0.700 | 2780 | 113 | 2275 | 0.784 | 0.883 | 0.007 | 0.394 | 228 | 15 | 0.701 | 113 | 2232 | 0.789 | 0.789 | 0.948 | 0.1519 | 0.301 | 0.959 | 11278 | — | — |
| bn_folded_seed43 | bn_folded+nsv | 100 | 43 | 15 | float | int8_val | 0.600 | 0.768 | 0.595 | 0.703 | 0.645 | 3554 | 2417 | 1501 | 0.959 | 0.593 | 0.111 | 0.266 | 247 | 16 | 0.647 | 2324 | 1487 | 0.716 | 0.716 | 0.883 | 0.1066 | 0.183 | 0.892 | 20105 | — | — |
| bn_folded_seed44 | bn_folded+nsv | 100 | 44 | 19 | float | int8_val | 0.500 | 0.818 | 0.498 | 0.720 | 0.589 | 3642 | 3671 | 1413 | 0.921 | 0.474 | 0.202 | 0.351 | 221 | 85 | 0.589 | 3592 | 1406 | 0.753 | 0.754 | 0.920 | 0.1548 | 0.188 | 0.985 | 21543 | — | — |
| bn_folded_seed45 | bn_folded+nsv | 100 | 45 | 10 | float | int8_val | 0.550 | 0.810 | 0.679 | 0.710 | 0.694 | 3590 | 1694 | 1465 | 0.943 | 0.787 | 0.198 | 0.255 | 129 | 61 | 0.691 | 1691 | 1457 | 0.734 | 0.737 | 0.918 | 0.1166 | 0.180 | 0.986 | 22777 | — | — |
| bn_folded_seed46 | bn_folded+nsv | 100 | 46 | 16 | float | int8_val | 0.550 | 0.797 | 0.687 | 0.731 | 0.708 | 3695 | 1683 | 1360 | 0.831 | 0.925 | 0.306 | 0.226 | 16 | 442 | 0.706 | 1674 | 1351 | 0.828 | 0.829 | 0.957 | 0.1123 | 0.148 | 0.997 | 29053 | — | — |
| bn_folded_seed42 | bn_folded+nsv | 100 | 42 | 12 | float | float_val | 0.650 | 0.781 | 0.961 | 0.550 | 0.700 | 2780 | 113 | 2275 | 0.784 | 0.883 | 0.007 | 0.394 | 228 | 15 | 0.701 | 113 | 2232 | 0.789 | 0.789 | 0.948 | 0.1519 | 0.301 | 0.959 | 11278 | — | — |
| bn_folded_seed43 | bn_folded+nsv | 100 | 43 | 15 | float | float_val | 0.650 | 0.769 | 0.672 | 0.687 | 0.679 | 3472 | 1693 | 1583 | 0.949 | 0.681 | 0.103 | 0.281 | 213 | 14 | 0.681 | 1629 | 1566 | 0.716 | 0.716 | 0.883 | 0.1066 | 0.183 | 0.892 | 20105 | — | — |
| bn_folded_seed44 | bn_folded+nsv | 100 | 44 | 19 | float | float_val | 0.450 | 0.818 | 0.435 | 0.740 | 0.548 | 3742 | 4864 | 1313 | 0.933 | 0.413 | 0.228 | 0.312 | 233 | 89 | 0.548 | 4746 | 1306 | 0.753 | 0.754 | 0.920 | 0.1548 | 0.188 | 0.985 | 21543 | — | — |
| bn_folded_seed45 | bn_folded+nsv | 100 | 45 | 10 | float | float_val | 0.550 | 0.810 | 0.679 | 0.710 | 0.694 | 3590 | 1694 | 1465 | 0.943 | 0.787 | 0.198 | 0.255 | 129 | 61 | 0.691 | 1691 | 1457 | 0.734 | 0.737 | 0.918 | 0.1166 | 0.180 | 0.986 | 22777 | — | — |
| bn_folded_seed46 | bn_folded+nsv | 100 | 46 | 16 | float | float_val | 0.550 | 0.797 | 0.687 | 0.731 | 0.708 | 3695 | 1683 | 1360 | 0.831 | 0.925 | 0.306 | 0.226 | 16 | 442 | 0.706 | 1674 | 1351 | 0.828 | 0.829 | 0.957 | 0.1123 | 0.148 | 0.997 | 29053 | — | — |
| bn_folded_seed42 | bn_folded+nsv | 100 | 42 | 12 | int8 | int8_val | 0.650 | 0.780 | 0.960 | 0.549 | 0.698 | 2774 | 115 | 2281 | 0.785 | 0.885 | 0.007 | 0.371 | 218 | 17 | 0.700 | 115 | 2237 | 0.786 | 0.787 | 0.947 | 0.1523 | 0.302 | 0.956 | 11193 | 208.17 | 232 |
| bn_folded_seed43 | bn_folded+nsv | 100 | 43 | 15 | int8 | int8_val | 0.600 | 0.769 | 0.601 | 0.700 | 0.647 | 3539 | 2345 | 1516 | 0.959 | 0.603 | 0.112 | 0.270 | 231 | 15 | 0.649 | 2260 | 1501 | 0.714 | 0.714 | 0.882 | 0.1055 | 0.184 | 0.884 | 19843 | 208.17 | 237 |
| bn_folded_seed44 | bn_folded+nsv | 100 | 44 | 19 | int8 | int8_val | 0.500 | 0.814 | 0.503 | 0.723 | 0.593 | 3653 | 3606 | 1402 | 0.920 | 0.480 | 0.205 | 0.345 | 220 | 94 | 0.593 | 3530 | 1394 | 0.754 | 0.754 | 0.920 | 0.1562 | 0.189 | 0.984 | 21367 | 208.17 | 255 |
| bn_folded_seed45 | bn_folded+nsv | 100 | 45 | 10 | int8 | int8_val | 0.550 | 0.810 | 0.678 | 0.708 | 0.693 | 3580 | 1701 | 1475 | 0.942 | 0.793 | 0.195 | 0.245 | 125 | 66 | 0.689 | 1698 | 1467 | 0.731 | 0.734 | 0.918 | 0.1133 | 0.178 | 0.987 | 23015 | 208.17 | 200 |
| bn_folded_seed46 | bn_folded+nsv | 100 | 46 | 16 | int8 | int8_val | 0.550 | 0.796 | 0.692 | 0.731 | 0.711 | 3695 | 1643 | 1360 | 0.830 | 0.925 | 0.306 | 0.229 | 15 | 448 | 0.709 | 1634 | 1351 | 0.827 | 0.828 | 0.957 | 0.1133 | 0.152 | 0.996 | 28107 | 208.17 | 195 |
