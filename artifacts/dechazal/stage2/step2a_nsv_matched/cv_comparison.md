# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | pp_V | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded+nsv | 100 | float | int8_val | 3 | 42 43 44 | 0.753 ± 0.036 | 0.644 ± 0.055 | 2067.0 ± 1804.6 | 1729.7 ± 474.3 | 0.888 ± 0.092 | 0.107 ± 0.098 | 0.650 ± 0.210 | 0.337 ± 0.065 | 232.0 ± 13.5 | 38.7 ± 40.1 | 0.646 ± 0.056 | 2009.7 ± 1760.7 | 1708.3 ± 455.3 | 0.753 ± 0.036 | 0.583 ± 0.076 | 15.3 ± 3.5 |
| bn_folded+nsv | 100 | float | float_val | 3 | 42 43 44 | 0.753 ± 0.036 | 0.642 ± 0.082 | 2223.3 ± 2419.5 | 1723.7 ± 496.2 | 0.889 ± 0.091 | 0.113 ± 0.111 | 0.659 ± 0.235 | 0.329 ± 0.058 | 224.7 ± 10.4 | 39.3 ± 43.0 | 0.644 ± 0.083 | 2162.7 ± 2362.2 | 1701.3 ± 477.6 | 0.753 ± 0.036 | 0.583 ± 0.115 | 15.3 ± 3.5 |
| bn_folded+nsv | 100 | int8 | int8_val | 3 | 42 43 44 | 0.752 ± 0.036 | 0.646 ± 0.053 | 2022.0 ± 1767.8 | 1733.0 ± 478.0 | 0.888 ± 0.092 | 0.108 ± 0.099 | 0.656 ± 0.208 | 0.329 ± 0.053 | 223.0 ± 7.0 | 42.0 ± 45.0 | 0.647 ± 0.054 | 1968.3 ± 1726.1 | 1710.7 ± 458.9 | 0.751 ± 0.036 | 0.583 ± 0.076 | 15.3 ± 3.5 |

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
| bn_folded_seed42 | bn_folded+nsv | 100 | 42 | 12 | float | float_val | 0.650 | 0.781 | 0.961 | 0.550 | 0.700 | 2780 | 113 | 2275 | 0.784 | 0.883 | 0.007 | 0.394 | 228 | 15 | 0.701 | 113 | 2232 | 0.789 | 0.789 | 0.948 | 0.1519 | 0.301 | 0.959 | 11278 | — | — |
| bn_folded_seed43 | bn_folded+nsv | 100 | 43 | 15 | float | float_val | 0.650 | 0.769 | 0.672 | 0.687 | 0.679 | 3472 | 1693 | 1583 | 0.949 | 0.681 | 0.103 | 0.281 | 213 | 14 | 0.681 | 1629 | 1566 | 0.716 | 0.716 | 0.883 | 0.1066 | 0.183 | 0.892 | 20105 | — | — |
| bn_folded_seed44 | bn_folded+nsv | 100 | 44 | 19 | float | float_val | 0.450 | 0.818 | 0.435 | 0.740 | 0.548 | 3742 | 4864 | 1313 | 0.933 | 0.413 | 0.228 | 0.312 | 233 | 89 | 0.548 | 4746 | 1306 | 0.753 | 0.754 | 0.920 | 0.1548 | 0.188 | 0.985 | 21543 | — | — |
| bn_folded_seed42 | bn_folded+nsv | 100 | 42 | 12 | int8 | int8_val | 0.650 | 0.780 | 0.960 | 0.549 | 0.698 | 2774 | 115 | 2281 | 0.785 | 0.885 | 0.007 | 0.371 | 218 | 17 | 0.700 | 115 | 2237 | 0.786 | 0.787 | 0.947 | 0.1523 | 0.302 | 0.956 | 11193 | 208.17 | 232 |
| bn_folded_seed43 | bn_folded+nsv | 100 | 43 | 15 | int8 | int8_val | 0.600 | 0.769 | 0.601 | 0.700 | 0.647 | 3539 | 2345 | 1516 | 0.959 | 0.603 | 0.112 | 0.270 | 231 | 15 | 0.649 | 2260 | 1501 | 0.714 | 0.714 | 0.882 | 0.1055 | 0.184 | 0.884 | 19843 | 208.17 | 237 |
| bn_folded_seed44 | bn_folded+nsv | 100 | 44 | 19 | int8 | int8_val | 0.500 | 0.814 | 0.503 | 0.723 | 0.593 | 3653 | 3606 | 1402 | 0.920 | 0.480 | 0.205 | 0.345 | 220 | 94 | 0.593 | 3530 | 1394 | 0.754 | 0.754 | 0.920 | 0.1562 | 0.189 | 0.984 | 21367 | 208.17 | 255 |
