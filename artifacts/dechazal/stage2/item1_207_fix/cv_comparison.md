# Grouped CV over DS1: final model on DS2

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | se_V | se_S | pp_V | pp_S | S_as_V | V_as_S | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold | final_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bn_folded+nsv+rr_norm | 100 | float | int8_val | 5 | 42 43 44 45 46 | 0.886 ± 0.020 | 0.809 ± 0.022 | 1014.0 ± 605.6 | 932.6 ± 397.0 | 0.908 ± 0.065 | 0.441 ± 0.225 | 0.846 ± 0.073 | 0.533 ± 0.105 | 242.6 ± 162.9 | 148.6 ± 174.5 | 0.817 ± 0.020 | 895.0 ± 474.1 | 919.4 ± 389.4 | 0.888 ± 0.021 | 0.520 ± 0.120 | 15.6 ± 3.8 |
| bn_folded+nsv+rr_norm | 100 | float | float_val | 5 | 42 43 44 45 46 | 0.886 ± 0.020 | 0.809 ± 0.022 | 1014.0 ± 605.6 | 932.6 ± 397.0 | 0.908 ± 0.065 | 0.441 ± 0.225 | 0.846 ± 0.073 | 0.533 ± 0.105 | 242.6 ± 162.9 | 148.6 ± 174.5 | 0.817 ± 0.020 | 895.0 ± 474.1 | 919.4 ± 389.4 | 0.888 ± 0.021 | 0.520 ± 0.120 | 15.6 ± 3.8 |
| bn_folded+nsv+rr_norm | 100 | int8 | int8_val | 5 | 42 43 44 45 46 | 0.884 ± 0.020 | 0.809 ± 0.023 | 1009.2 ± 594.0 | 940.0 ± 398.5 | 0.906 ± 0.067 | 0.438 ± 0.225 | 0.847 ± 0.072 | 0.526 ± 0.101 | 239.4 ± 160.1 | 156.8 ± 183.4 | 0.816 ± 0.021 | 889.4 ± 461.4 | 926.6 ± 391.1 | 0.887 ± 0.021 | 0.520 ± 0.120 | 15.6 ± 3.8 |

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
| bn_folded_seed42 | bn_folded+nsv+rr_norm | 100 | 42 | 18 | float | int8_val | 0.450 | 0.832 | 0.721 | 0.883 | 0.794 | 4465 | 1732 | 590 | 0.846 | 0.855 | 0.666 | 0.406 | 127 | 393 | 0.816 | 1397 | 589 | 0.895 | 0.891 | 0.965 | 0.1206 | 0.261 | 0.966 | 13826 | — | — |
| bn_folded_seed43 | bn_folded+nsv+rr_norm | 100 | 43 | 17 | float | int8_val | 0.450 | 0.860 | 0.743 | 0.882 | 0.807 | 4459 | 1540 | 596 | 0.973 | 0.727 | 0.445 | 0.482 | 467 | 44 | 0.819 | 1359 | 583 | 0.905 | 0.901 | 0.979 | 0.1190 | 0.386 | 0.968 | 7768 | — | — |
| bn_folded_seed44 | bn_folded+nsv+rr_norm | 100 | 44 | 9 | float | int8_val | 0.650 | 0.822 | 0.914 | 0.691 | 0.787 | 3491 | 328 | 1564 | 0.932 | 0.837 | 0.085 | 0.661 | 331 | 4 | 0.787 | 326 | 1538 | 0.856 | 0.856 | 0.949 | 0.1153 | 0.248 | 0.955 | 14651 | — | — |
| bn_folded_seed45 | bn_folded+nsv+rr_norm | 100 | 45 | 18 | float | int8_val | 0.650 | 0.822 | 0.876 | 0.811 | 0.842 | 4100 | 579 | 955 | 0.831 | 0.919 | 0.596 | 0.619 | 57 | 274 | 0.843 | 576 | 932 | 0.906 | 0.904 | 0.982 | 0.1390 | 0.418 | 0.969 | 6825 | — | — |
| bn_folded_seed46 | bn_folded+nsv+rr_norm | 100 | 46 | 16 | float | int8_val | 0.400 | 0.842 | 0.821 | 0.810 | 0.816 | 4097 | 891 | 958 | 0.957 | 0.890 | 0.412 | 0.495 | 231 | 28 | 0.820 | 817 | 955 | 0.879 | 0.876 | 0.956 | 0.0843 | 0.365 | 0.943 | 8300 | — | — |
| bn_folded_seed42 | bn_folded+nsv+rr_norm | 100 | 42 | 18 | float | float_val | 0.450 | 0.832 | 0.721 | 0.883 | 0.794 | 4465 | 1732 | 590 | 0.846 | 0.855 | 0.666 | 0.406 | 127 | 393 | 0.816 | 1397 | 589 | 0.895 | 0.891 | 0.965 | 0.1206 | 0.261 | 0.966 | 13826 | — | — |
| bn_folded_seed43 | bn_folded+nsv+rr_norm | 100 | 43 | 17 | float | float_val | 0.450 | 0.860 | 0.743 | 0.882 | 0.807 | 4459 | 1540 | 596 | 0.973 | 0.727 | 0.445 | 0.482 | 467 | 44 | 0.819 | 1359 | 583 | 0.905 | 0.901 | 0.979 | 0.1190 | 0.386 | 0.968 | 7768 | — | — |
| bn_folded_seed44 | bn_folded+nsv+rr_norm | 100 | 44 | 9 | float | float_val | 0.650 | 0.822 | 0.914 | 0.691 | 0.787 | 3491 | 328 | 1564 | 0.932 | 0.837 | 0.085 | 0.661 | 331 | 4 | 0.787 | 326 | 1538 | 0.856 | 0.856 | 0.949 | 0.1153 | 0.248 | 0.955 | 14651 | — | — |
| bn_folded_seed45 | bn_folded+nsv+rr_norm | 100 | 45 | 18 | float | float_val | 0.650 | 0.822 | 0.876 | 0.811 | 0.842 | 4100 | 579 | 955 | 0.831 | 0.919 | 0.596 | 0.619 | 57 | 274 | 0.843 | 576 | 932 | 0.906 | 0.904 | 0.982 | 0.1390 | 0.418 | 0.969 | 6825 | — | — |
| bn_folded_seed46 | bn_folded+nsv+rr_norm | 100 | 46 | 16 | float | float_val | 0.400 | 0.842 | 0.821 | 0.810 | 0.816 | 4097 | 891 | 958 | 0.957 | 0.890 | 0.412 | 0.495 | 231 | 28 | 0.820 | 817 | 955 | 0.879 | 0.876 | 0.956 | 0.0843 | 0.365 | 0.943 | 8300 | — | — |
| bn_folded_seed42 | bn_folded+nsv+rr_norm | 100 | 42 | 18 | int8 | int8_val | 0.450 | 0.833 | 0.721 | 0.880 | 0.792 | 4449 | 1724 | 606 | 0.838 | 0.855 | 0.657 | 0.399 | 128 | 419 | 0.815 | 1385 | 605 | 0.893 | 0.888 | 0.964 | 0.1211 | 0.269 | 0.965 | 13226 | 208.17 | 150 |
| bn_folded_seed43 | bn_folded+nsv+rr_norm | 100 | 43 | 17 | int8 | int8_val | 0.450 | 0.860 | 0.747 | 0.879 | 0.808 | 4443 | 1504 | 612 | 0.969 | 0.731 | 0.440 | 0.480 | 464 | 52 | 0.820 | 1325 | 598 | 0.904 | 0.900 | 0.979 | 0.1172 | 0.390 | 0.968 | 7654 | 208.17 | 235 |
| bn_folded_seed44 | bn_folded+nsv+rr_norm | 100 | 44 | 9 | int8 | int8_val | 0.650 | 0.823 | 0.914 | 0.686 | 0.784 | 3470 | 328 | 1585 | 0.932 | 0.840 | 0.080 | 0.643 | 319 | 4 | 0.784 | 326 | 1559 | 0.855 | 0.855 | 0.949 | 0.1133 | 0.248 | 0.955 | 14604 | 208.17 | 234 |
| bn_folded_seed45 | bn_folded+nsv+rr_norm | 100 | 45 | 18 | int8 | int8_val | 0.650 | 0.823 | 0.876 | 0.812 | 0.843 | 4107 | 584 | 948 | 0.831 | 0.920 | 0.597 | 0.614 | 57 | 280 | 0.843 | 581 | 925 | 0.905 | 0.903 | 0.982 | 0.1406 | 0.423 | 0.969 | 6665 | 208.17 | 117 |
| bn_folded_seed46 | bn_folded+nsv+rr_norm | 100 | 46 | 16 | int8 | int8_val | 0.400 | 0.842 | 0.819 | 0.812 | 0.816 | 4106 | 906 | 949 | 0.958 | 0.890 | 0.416 | 0.494 | 229 | 29 | 0.820 | 830 | 946 | 0.879 | 0.876 | 0.956 | 0.0859 | 0.374 | 0.942 | 7979 | 208.17 | 166 |
