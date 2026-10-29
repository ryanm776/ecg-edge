# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | pp_V | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded+unweighted | 100 | float | int8_val | 3 | 42 43 44 | 0.796 ± 0.059 | 0.680 ± 0.064 | 1933.7 ± 1338.1 | 1401.7 ± 1040.6 | 0.868 ± 0.135 | 0.469 ± 0.391 | — ± — | — ± — | nan ± — | nan ± — | 0.682 ± 0.062 | 1861.0 ± 1262.2 | 1379.7 ± 1029.5 | 0.796 ± 0.060 | 0.583 ± 0.076 | 18.0 ± 7.9 |
| bn_folded+unweighted | 100 | float | float_val | 3 | 42 43 44 | 0.796 ± 0.059 | 0.673 ± 0.085 | 2096.3 ± 1770.1 | 1410.3 ± 1001.8 | 0.859 ± 0.133 | 0.480 ± 0.376 | — ± — | — ± — | nan ± — | nan ± — | 0.676 ± 0.082 | 2004.7 ± 1629.0 | 1388.3 ± 989.7 | 0.796 ± 0.060 | 0.583 ± 0.058 | 18.0 ± 7.9 |
| bn_folded+unweighted | 100 | int8 | int8_val | 3 | 42 43 44 | 0.793 ± 0.060 | 0.678 ± 0.066 | 1930.0 ± 1326.8 | 1416.0 ± 1058.0 | 0.865 ± 0.139 | 0.466 ± 0.391 | — ± — | — ± — | nan ± — | nan ± — | 0.680 ± 0.065 | 1856.7 ± 1250.3 | 1394.3 ± 1047.2 | 0.793 ± 0.061 | 0.583 ± 0.076 | 18.0 ± 7.9 |

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
| bn_folded_seed42 | bn_folded+unweighted | 100 | 42 | 9 | float | int8_val | 0.650 | 0.790 | 0.860 | 0.496 | 0.629 | 2505 | 409 | 2550 | 0.718 | — | 0.105 | — | — | — | 0.628 | 409 | 2512 | 0.729 | 0.731 | 0.923 | 0.2278 | 0.360 | 0.825 | 7420 | — | — |
| bn_folded_seed43 | bn_folded+unweighted | 100 | 43 | 21 | float | int8_val | 0.600 | 0.831 | 0.574 | 0.776 | 0.660 | 3921 | 2913 | 1134 | 0.979 | — | 0.418 | — | — | — | 0.668 | 2696 | 1127 | 0.814 | 0.811 | 0.952 | 0.1948 | 0.213 | 0.985 | 18439 | — | — |
| bn_folded_seed44 | bn_folded+unweighted | 100 | 44 | 24 | float | int8_val | 0.500 | 0.823 | 0.647 | 0.897 | 0.751 | 4534 | 2479 | 521 | 0.905 | — | 0.882 | — | — | — | 0.751 | 2478 | 500 | 0.845 | 0.846 | 0.975 | 0.1113 | 0.159 | 0.996 | 26723 | — | — |
| bn_folded_seed42 | bn_folded+unweighted | 100 | 42 | 9 | float | float_val | 0.650 | 0.790 | 0.860 | 0.496 | 0.629 | 2505 | 409 | 2550 | 0.718 | — | 0.105 | — | — | — | 0.628 | 409 | 2512 | 0.729 | 0.731 | 0.923 | 0.2278 | 0.360 | 0.825 | 7420 | — | — |
| bn_folded_seed43 | bn_folded+unweighted | 100 | 43 | 21 | float | float_val | 0.550 | 0.831 | 0.507 | 0.800 | 0.620 | 4043 | 3939 | 1012 | 0.984 | — | 0.477 | — | — | — | 0.630 | 3665 | 1007 | 0.814 | 0.811 | 0.952 | 0.1948 | 0.213 | 0.985 | 18439 | — | — |
| bn_folded_seed44 | bn_folded+unweighted | 100 | 44 | 24 | float | float_val | 0.550 | 0.825 | 0.693 | 0.868 | 0.771 | 4386 | 1941 | 669 | 0.874 | — | 0.857 | — | — | — | 0.770 | 1940 | 646 | 0.845 | 0.846 | 0.975 | 0.1113 | 0.159 | 0.996 | 26723 | — | — |
| bn_folded_seed42 | bn_folded+unweighted | 100 | 42 | 9 | int8 | int8_val | 0.650 | 0.791 | 0.855 | 0.488 | 0.621 | 2466 | 418 | 2589 | 0.710 | — | 0.097 | — | — | — | 0.621 | 418 | 2552 | 0.725 | 0.727 | 0.923 | 0.2266 | 0.355 | 0.827 | 7597 | 207.98 | 239 |
| bn_folded_seed43 | bn_folded+unweighted | 100 | 43 | 21 | int8 | int8_val | 0.600 | 0.828 | 0.575 | 0.777 | 0.661 | 3930 | 2900 | 1125 | 0.979 | — | 0.424 | — | — | — | 0.670 | 2681 | 1118 | 0.811 | 0.809 | 0.952 | 0.1914 | 0.213 | 0.984 | 18424 | 207.98 | 231 |
| bn_folded_seed44 | bn_folded+unweighted | 100 | 44 | 24 | int8 | int8_val | 0.500 | 0.824 | 0.647 | 0.894 | 0.750 | 4521 | 2472 | 534 | 0.905 | — | 0.876 | — | — | — | 0.750 | 2471 | 513 | 0.844 | 0.844 | 0.975 | 0.1133 | 0.166 | 0.996 | 25266 | 207.98 | 239 |
