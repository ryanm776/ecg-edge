# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | pp_V | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded+nsv+rr30 | 100 | float | int8_val | 5 | 42 43 44 45 46 | 0.778 ± 0.059 | 0.737 ± 0.026 | 874.2 ± 779.8 | 1575.8 ± 651.0 | 0.885 ± 0.070 | 0.095 ± 0.034 | 0.792 ± 0.177 | 0.390 ± 0.158 | 353.8 ± 505.9 | 102.0 ± 110.5 | 0.736 ± 0.027 | 873.8 ± 778.9 | 1552.4 ± 652.9 | 0.777 ± 0.060 | 0.540 ± 0.082 | 13.0 ± 4.8 |
| bn_folded+nsv+rr30 | 100 | float | float_val | 5 | 42 43 44 45 46 | 0.778 ± 0.059 | 0.739 ± 0.032 | 765.8 ± 606.3 | 1619.2 ± 599.8 | 0.878 ± 0.064 | 0.088 ± 0.028 | 0.800 ± 0.168 | 0.400 ± 0.164 | 348.0 ± 494.9 | 100.6 ± 111.5 | 0.739 ± 0.032 | 765.6 ± 605.8 | 1594.0 ± 604.5 | 0.777 ± 0.060 | 0.560 ± 0.065 | 13.0 ± 4.8 |
| bn_folded+nsv+rr30 | 100 | int8 | int8_val | 5 | 42 43 44 45 46 | 0.777 ± 0.059 | 0.734 ± 0.025 | 872.2 ± 782.0 | 1595.8 ± 657.3 | 0.886 ± 0.068 | 0.092 ± 0.036 | 0.796 ± 0.178 | 0.381 ± 0.154 | 337.6 ± 507.7 | 100.0 ± 104.5 | 0.733 ± 0.026 | 871.6 ± 780.7 | 1572.2 ± 660.4 | 0.776 ± 0.060 | 0.540 ± 0.082 | 13.0 ± 4.8 |

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
| bn_folded_seed42 | bn_folded+nsv+rr30 | 100 | 42 | 15 | float | int8_val | 0.450 | 0.789 | 0.671 | 0.916 | 0.775 | 4629 | 2269 | 426 | 0.947 | 0.480 | 0.151 | 0.507 | 1252 | 50 | 0.775 | 2267 | 398 | 0.882 | 0.880 | 0.979 | 0.1199 | 0.200 | 0.997 | 20092 | — | — |
| bn_folded_seed43 | bn_folded+nsv+rr30 | 100 | 43 | 8 | float | int8_val | 0.500 | 0.806 | 0.859 | 0.632 | 0.728 | 3193 | 522 | 1862 | 0.933 | 0.888 | 0.069 | 0.375 | 42 | 23 | 0.727 | 522 | 1841 | 0.748 | 0.749 | 0.902 | 0.0882 | 0.209 | 0.895 | 17093 | — | — |
| bn_folded_seed44 | bn_folded+nsv+rr30 | 100 | 44 | 20 | float | int8_val | 0.600 | 0.783 | 0.851 | 0.602 | 0.705 | 3042 | 534 | 2013 | 0.779 | 0.899 | 0.075 | 0.175 | 117 | 281 | 0.704 | 534 | 1983 | 0.751 | 0.751 | 0.898 | 0.1300 | 0.240 | 0.849 | 13627 | — | — |
| bn_folded_seed45 | bn_folded+nsv+rr30 | 100 | 45 | 13 | float | int8_val | 0.500 | 0.815 | 0.858 | 0.633 | 0.728 | 3198 | 531 | 1857 | 0.849 | 0.870 | 0.101 | 0.317 | 142 | 136 | 0.726 | 531 | 1837 | 0.734 | 0.736 | 0.869 | 0.1334 | 0.236 | 0.793 | 12956 | — | — |
| bn_folded_seed46 | bn_folded+nsv+rr30 | 100 | 46 | 9 | float | int8_val | 0.650 | 0.767 | 0.866 | 0.660 | 0.749 | 3334 | 515 | 1721 | 0.917 | 0.821 | 0.080 | 0.575 | 216 | 20 | 0.747 | 515 | 1703 | 0.771 | 0.773 | 0.916 | 0.0976 | 0.153 | 0.995 | 27756 | — | — |
| bn_folded_seed42 | bn_folded+nsv+rr30 | 100 | 42 | 15 | float | float_val | 0.500 | 0.790 | 0.709 | 0.888 | 0.788 | 4491 | 1846 | 564 | 0.926 | 0.506 | 0.130 | 0.538 | 1226 | 46 | 0.790 | 1845 | 529 | 0.882 | 0.880 | 0.979 | 0.1199 | 0.200 | 0.997 | 20092 | — | — |
| bn_folded_seed43 | bn_folded+nsv+rr30 | 100 | 43 | 8 | float | float_val | 0.550 | 0.807 | 0.885 | 0.616 | 0.727 | 3114 | 403 | 1941 | 0.918 | 0.906 | 0.055 | 0.395 | 39 | 20 | 0.725 | 403 | 1918 | 0.748 | 0.749 | 0.902 | 0.0882 | 0.209 | 0.895 | 17093 | — | — |
| bn_folded_seed44 | bn_folded+nsv+rr30 | 100 | 44 | 20 | float | float_val | 0.600 | 0.783 | 0.851 | 0.602 | 0.705 | 3042 | 534 | 2013 | 0.779 | 0.899 | 0.075 | 0.175 | 117 | 281 | 0.704 | 534 | 1983 | 0.751 | 0.751 | 0.898 | 0.1300 | 0.240 | 0.849 | 13627 | — | — |
| bn_folded_seed45 | bn_folded+nsv+rr30 | 100 | 45 | 13 | float | float_val | 0.500 | 0.815 | 0.858 | 0.633 | 0.728 | 3198 | 531 | 1857 | 0.849 | 0.870 | 0.101 | 0.317 | 142 | 136 | 0.726 | 531 | 1837 | 0.734 | 0.736 | 0.869 | 0.1334 | 0.236 | 0.793 | 12956 | — | — |
| bn_folded_seed46 | bn_folded+nsv+rr30 | 100 | 46 | 9 | float | float_val | 0.650 | 0.767 | 0.866 | 0.660 | 0.749 | 3334 | 515 | 1721 | 0.917 | 0.821 | 0.080 | 0.575 | 216 | 20 | 0.747 | 515 | 1703 | 0.771 | 0.773 | 0.916 | 0.0976 | 0.153 | 0.995 | 27756 | — | — |
| bn_folded_seed42 | bn_folded+nsv+rr30 | 100 | 42 | 15 | int8 | int8_val | 0.450 | 0.791 | 0.671 | 0.915 | 0.774 | 4627 | 2271 | 428 | 0.948 | 0.481 | 0.153 | 0.513 | 1242 | 51 | 0.775 | 2268 | 398 | 0.881 | 0.879 | 0.979 | 0.1172 | 0.200 | 0.997 | 20130 | 208.17 | 235 |
| bn_folded_seed43 | bn_folded+nsv+rr30 | 100 | 43 | 8 | int8 | int8_val | 0.500 | 0.806 | 0.860 | 0.630 | 0.727 | 3184 | 517 | 1871 | 0.932 | 0.890 | 0.066 | 0.367 | 40 | 23 | 0.726 | 517 | 1850 | 0.747 | 0.747 | 0.902 | 0.0898 | 0.219 | 0.875 | 15740 | 208.17 | 246 |
| bn_folded_seed44 | bn_folded+nsv+rr30 | 100 | 44 | 20 | int8 | int8_val | 0.600 | 0.783 | 0.853 | 0.606 | 0.708 | 3061 | 528 | 1994 | 0.786 | 0.894 | 0.071 | 0.173 | 135 | 266 | 0.708 | 528 | 1964 | 0.753 | 0.753 | 0.901 | 0.1289 | 0.245 | 0.850 | 13247 | 208.17 | 255 |
| bn_folded_seed45 | bn_folded+nsv+rr30 | 100 | 45 | 13 | int8 | int8_val | 0.500 | 0.816 | 0.855 | 0.623 | 0.721 | 3149 | 536 | 1906 | 0.847 | 0.881 | 0.097 | 0.304 | 104 | 139 | 0.719 | 536 | 1887 | 0.731 | 0.733 | 0.868 | 0.1328 | 0.236 | 0.792 | 12955 | 208.17 | 242 |
| bn_folded_seed46 | bn_folded+nsv+rr30 | 100 | 46 | 9 | int8 | int8_val | 0.650 | 0.767 | 0.865 | 0.648 | 0.741 | 3275 | 509 | 1780 | 0.917 | 0.835 | 0.074 | 0.548 | 167 | 21 | 0.739 | 509 | 1762 | 0.768 | 0.770 | 0.916 | 0.0977 | 0.155 | 0.996 | 27496 | 208.17 | 162 |
