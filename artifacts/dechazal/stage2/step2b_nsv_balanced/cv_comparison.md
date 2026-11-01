# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | pp_V | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded+nsv+balanced | 100 | float | int8_val | 3 | 42 43 44 | 0.792 ± 0.068 | 0.731 ± 0.007 | 1489.3 ± 940.4 | 1278.3 ± 602.2 | 0.915 ± 0.009 | 0.164 ± 0.024 | 0.758 ± 0.189 | 0.263 ± 0.038 | 423.3 ± 485.8 | 106.7 ± 63.0 | 0.732 ± 0.012 | 1434.7 ± 864.2 | 1268.3 ± 600.5 | 0.792 ± 0.070 | 0.800 ± 0.050 | 18.0 ± 6.9 |
| bn_folded+nsv+balanced | 100 | float | float_val | 3 | 42 43 44 | 0.792 ± 0.068 | 0.731 ± 0.007 | 1489.3 ± 940.4 | 1278.3 ± 602.2 | 0.915 ± 0.009 | 0.164 ± 0.024 | 0.758 ± 0.189 | 0.263 ± 0.038 | 423.3 ± 485.8 | 106.7 ± 63.0 | 0.732 ± 0.012 | 1434.7 ± 864.2 | 1268.3 ± 600.5 | 0.792 ± 0.070 | 0.800 ± 0.050 | 18.0 ± 6.9 |
| bn_folded+nsv+balanced | 100 | int8 | int8_val | 3 | 42 43 44 | 0.786 ± 0.062 | 0.729 ± 0.003 | 1544.7 ± 1031.9 | 1266.0 ± 610.6 | 0.915 ± 0.010 | 0.164 ± 0.022 | 0.754 ± 0.198 | 0.261 ± 0.045 | 433.0 ± 502.9 | 109.7 ± 63.0 | 0.730 ± 0.008 | 1488.3 ± 954.5 | 1256.0 ± 608.9 | 0.786 ± 0.064 | 0.800 ± 0.050 | 18.0 ± 6.9 |

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
| bn_folded_seed42 | bn_folded+nsv+balanced | 100 | 42 | 14 | float | int8_val | 0.750 | 0.768 | 0.787 | 0.678 | 0.729 | 3428 | 927 | 1627 | 0.915 | 0.847 | 0.138 | 0.288 | 177 | 53 | 0.727 | 907 | 1619 | 0.751 | 0.753 | 0.902 | 0.2711 | 0.214 | 0.859 | 15969 | — | — |
| bn_folded_seed43 | bn_folded+nsv+balanced | 100 | 43 | 26 | float | int8_val | 0.800 | 0.769 | 0.635 | 0.885 | 0.739 | 4472 | 2575 | 583 | 0.924 | 0.541 | 0.185 | 0.219 | 983 | 176 | 0.746 | 2432 | 575 | 0.872 | 0.871 | 0.974 | 0.3433 | 0.226 | 0.993 | 17176 | — | — |
| bn_folded_seed44 | bn_folded+nsv+balanced | 100 | 44 | 14 | float | int8_val | 0.850 | 0.734 | 0.780 | 0.679 | 0.726 | 3430 | 966 | 1625 | 0.906 | 0.887 | 0.170 | 0.282 | 110 | 91 | 0.723 | 965 | 1611 | 0.752 | 0.753 | 0.888 | 0.3132 | 0.160 | 0.927 | 24528 | — | — |
| bn_folded_seed42 | bn_folded+nsv+balanced | 100 | 42 | 14 | float | float_val | 0.750 | 0.768 | 0.787 | 0.678 | 0.729 | 3428 | 927 | 1627 | 0.915 | 0.847 | 0.138 | 0.288 | 177 | 53 | 0.727 | 907 | 1619 | 0.751 | 0.753 | 0.902 | 0.2711 | 0.214 | 0.859 | 15969 | — | — |
| bn_folded_seed43 | bn_folded+nsv+balanced | 100 | 43 | 26 | float | float_val | 0.800 | 0.769 | 0.635 | 0.885 | 0.739 | 4472 | 2575 | 583 | 0.924 | 0.541 | 0.185 | 0.219 | 983 | 176 | 0.746 | 2432 | 575 | 0.872 | 0.871 | 0.974 | 0.3433 | 0.226 | 0.993 | 17176 | — | — |
| bn_folded_seed44 | bn_folded+nsv+balanced | 100 | 44 | 14 | float | float_val | 0.850 | 0.734 | 0.780 | 0.679 | 0.726 | 3430 | 966 | 1625 | 0.906 | 0.887 | 0.170 | 0.282 | 110 | 91 | 0.723 | 965 | 1611 | 0.752 | 0.753 | 0.888 | 0.3132 | 0.160 | 0.927 | 24528 | — | — |
| bn_folded_seed42 | bn_folded+nsv+balanced | 100 | 42 | 14 | int8 | int8_val | 0.750 | 0.766 | 0.787 | 0.682 | 0.731 | 3445 | 931 | 1610 | 0.917 | 0.845 | 0.139 | 0.291 | 182 | 54 | 0.729 | 909 | 1601 | 0.750 | 0.751 | 0.903 | 0.2812 | 0.221 | 0.854 | 15224 | 208.17 | 209 |
| bn_folded_seed43 | bn_folded+nsv+balanced | 100 | 43 | 26 | int8 | int8_val | 0.800 | 0.770 | 0.622 | 0.889 | 0.732 | 4494 | 2736 | 561 | 0.923 | 0.527 | 0.181 | 0.209 | 1012 | 178 | 0.738 | 2590 | 553 | 0.859 | 0.858 | 0.974 | 0.3437 | 0.226 | 0.993 | 17186 | 208.17 | 214 |
| bn_folded_seed44 | bn_folded+nsv+balanced | 100 | 44 | 14 | int8 | int8_val | 0.850 | 0.732 | 0.780 | 0.678 | 0.726 | 3428 | 967 | 1627 | 0.903 | 0.890 | 0.173 | 0.282 | 105 | 97 | 0.723 | 966 | 1614 | 0.747 | 0.749 | 0.887 | 0.3125 | 0.160 | 0.925 | 24477 | 208.17 | 252 |
