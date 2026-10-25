# Float vs. INT8 by variant

## Mean ± std over seeds

Sample std (ddof=1) across the seeds listed; `—` with a single seed. `pr_auc` is on the full
test set; float and INT8 are separate rows.

| variant | calib_samples | model | threshold_from | n_seeds | seeds | pr_auc | test_f1 | fp | fn | recall_ventricular | recall_supraventricular | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 100 | float | int8_val | 3 | 42 43 44 | 0.771 ± 0.111 | 0.666 ± 0.062 | 996.3 ± 1005.9 | 2068.3 ± 69.1 | 0.794 ± 0.063 | 0.234 ± 0.078 | 0.665 ± 0.065 | 994.7 ± 1005.5 | 2037.0 ± 59.0 | 0.769 ± 0.114 | 0.583 ± 0.104 |
| baseline | 100 | float | float_val | 3 | 42 43 44 | 0.771 ± 0.111 | 0.682 ± 0.076 | 2267.3 ± 182.4 | 1237.0 ± 669.0 | 0.919 ± 0.048 | 0.469 ± 0.282 | 0.680 ± 0.077 | 2259.3 ± 173.1 | 1221.0 ± 668.0 | 0.769 ± 0.114 | 0.383 ± 0.104 |
| baseline | 100 | int8 | int8_val | 3 | 42 43 44 | 0.676 ± 0.042 | 0.609 ± 0.022 | 2156.0 ± 844.9 | 1910.0 ± 220.6 | 0.825 ± 0.061 | 0.267 ± 0.125 | 0.609 ± 0.020 | 2108.0 ± 788.4 | 1883.7 ± 224.0 | 0.677 ± 0.047 | 0.583 ± 0.104 |
| bn_folded | 100 | float | int8_val | 3 | 42 43 44 | 0.771 ± 0.010 | 0.636 ± 0.107 | 3642.3 ± 3533.2 | 1205.0 ± 506.7 | 0.951 ± 0.042 | 0.430 ± 0.208 | 0.641 ± 0.105 | 3461.0 ± 3356.6 | 1195.7 ± 498.1 | 0.772 ± 0.013 | 0.233 ± 0.076 |
| bn_folded | 100 | float | float_val | 3 | 42 43 44 | 0.771 ± 0.010 | 0.636 ± 0.107 | 3642.3 ± 3533.2 | 1205.0 ± 506.7 | 0.951 ± 0.042 | 0.430 ± 0.208 | 0.641 ± 0.105 | 3461.0 ± 3356.6 | 1195.7 ± 498.1 | 0.772 ± 0.013 | 0.233 ± 0.076 |
| bn_folded | 100 | int8 | int8_val | 3 | 42 43 44 | 0.766 ± 0.007 | 0.636 ± 0.108 | 3665.0 ± 3629.6 | 1207.7 ± 528.9 | 0.950 ± 0.044 | 0.430 ± 0.218 | 0.640 ± 0.106 | 3486.7 ± 3448.1 | 1198.3 ± 520.6 | 0.767 ± 0.010 | 0.233 ± 0.076 |

## Per run

Test set, one row per run (seed in `seed`). `threshold_from`: `int8_val` is the threshold chosen on the INT8 model's
validation predictions (the deployed convention); `float_val` is the float model's own
val-optimal threshold. For `qat`, the float row is the float model before QAT fine-tuning.
`val_f1` is the row's own model's validation F1 at the row's threshold.
Threshold-independent columns: `pr_auc` (average precision) and `roc_auc` on test, and the
fixed-recall operating point `r95_*`: the highest threshold reaching >= 0.95 recall on the
model's own validation predictions, then test precision / recall / FP at that threshold.
`min_int8_levels` is the smallest number of INT8 levels any intermediate tensor spans.
`*_excl_202`: sensitivity check, test re-evaluated without record 202 (same patient as train
record 201); reported alongside the full-test columns, not instead of them.
`recall_ventricular` (V, E) / `recall_supraventricular` (A, a, J, S): test recall per abnormal type.

| run | variant | calib_samples | seed | model | threshold_from | threshold | val_f1 | test_precision | test_recall | test_f1 | tp | fp | fn | recall_ventricular | recall_supraventricular | fp_rec214 | fp_excl_214 | f1_excl_202 | fp_excl_202 | fn_excl_202 | pr_auc_excl_202 | pr_auc | roc_auc | r95_threshold | r95_test_precision | r95_test_recall | r95_fp | tflite_kb | min_int8_levels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_seed42 | baseline | 100 | 42 | float | int8_val | 0.500 | 0.794 | 0.591 | 0.607 | 0.599 | 3067 | 2119 | 1989 | 0.864 | 0.156 | 9 | 2110 | 0.596 | 2117 | 1970 | 0.650 | 0.654 | 0.912 | 0.1497 | 0.240 | 0.933 | 14977 | — | — |
| baseline_seed43 | baseline | 100 | 43 | float | int8_val | 0.700 | 0.686 | 0.943 | 0.584 | 0.722 | 2955 | 177 | 2101 | 0.740 | 0.311 | 1 | 176 | 0.723 | 176 | 2060 | 0.876 | 0.876 | 0.973 | 0.2048 | 0.449 | 0.947 | 5877 | — | — |
| baseline_seed44 | baseline | 100 | 44 | float | int8_val | 0.550 | 0.677 | 0.809 | 0.582 | 0.677 | 2941 | 693 | 2115 | 0.779 | 0.235 | 2 | 691 | 0.677 | 691 | 2081 | 0.781 | 0.782 | 0.955 | 0.1229 | 0.242 | 0.989 | 15637 | — | — |
| baseline_seed42 | baseline | 100 | 42 | float | float_val | 0.500 | 0.794 | 0.591 | 0.607 | 0.599 | 3067 | 2119 | 1989 | 0.864 | 0.156 | 9 | 2110 | 0.596 | 2117 | 1970 | 0.650 | 0.654 | 0.912 | 0.1497 | 0.240 | 0.933 | 14977 | — | — |
| baseline_seed43 | baseline | 100 | 43 | float | float_val | 0.350 | 0.856 | 0.663 | 0.860 | 0.749 | 4348 | 2212 | 708 | 0.949 | 0.704 | 172 | 2040 | 0.748 | 2209 | 687 | 0.876 | 0.876 | 0.973 | 0.2048 | 0.449 | 0.947 | 5877 | — | — |
| baseline_seed44 | baseline | 100 | 44 | float | float_val | 0.300 | 0.806 | 0.621 | 0.799 | 0.699 | 4042 | 2471 | 1014 | 0.944 | 0.546 | 289 | 2182 | 0.697 | 2452 | 1006 | 0.781 | 0.782 | 0.955 | 0.1229 | 0.242 | 0.989 | 15637 | — | — |
| baseline_seed42 | baseline | 100 | 42 | int8 | int8_val | 0.500 | 0.794 | 0.576 | 0.624 | 0.599 | 3155 | 2326 | 1901 | 0.891 | 0.156 | 11 | 2315 | 0.596 | 2314 | 1885 | 0.628 | 0.631 | 0.895 | 0.1602 | 0.235 | 0.885 | 14551 | 208.38 | 6 |
| baseline_seed43 | baseline | 100 | 43 | int8 | int8_val | 0.700 | 0.649 | 0.537 | 0.665 | 0.594 | 3362 | 2903 | 1694 | 0.815 | 0.403 | 1147 | 1756 | 0.600 | 2773 | 1659 | 0.722 | 0.714 | 0.927 | 0.1875 | 0.305 | 0.924 | 10641 | 208.38 | 4 |
| baseline_seed44 | baseline | 100 | 44 | int8 | int8_val | 0.550 | 0.704 | 0.702 | 0.578 | 0.634 | 2921 | 1239 | 2135 | 0.770 | 0.241 | 99 | 1140 | 0.632 | 1237 | 2107 | 0.682 | 0.683 | 0.910 | 0.1641 | 0.210 | 0.941 | 17887 | 208.38 | 5 |
| bn_folded_seed42 | bn_folded | 100 | 42 | float | int8_val | 0.150 | 0.876 | 0.363 | 0.871 | 0.513 | 4402 | 7715 | 654 | 0.989 | 0.664 | 31 | 7684 | 0.520 | 7334 | 653 | 0.768 | 0.768 | 0.944 | 0.0712 | 0.173 | 0.996 | 24089 | — | — |
| bn_folded_seed43 | bn_folded | 100 | 43 | float | int8_val | 0.250 | 0.823 | 0.674 | 0.741 | 0.706 | 3746 | 1814 | 1310 | 0.959 | 0.359 | 430 | 1384 | 0.714 | 1653 | 1302 | 0.787 | 0.783 | 0.939 | 0.1407 | 0.333 | 0.882 | 8908 | — | — |
| bn_folded_seed44 | bn_folded | 100 | 44 | float | int8_val | 0.300 | 0.816 | 0.709 | 0.673 | 0.691 | 3405 | 1398 | 1651 | 0.906 | 0.266 | 14 | 1384 | 0.689 | 1396 | 1632 | 0.762 | 0.763 | 0.957 | 0.1467 | 0.444 | 0.903 | 5720 | — | — |
| bn_folded_seed42 | bn_folded | 100 | 42 | float | float_val | 0.150 | 0.876 | 0.363 | 0.871 | 0.513 | 4402 | 7715 | 654 | 0.989 | 0.664 | 31 | 7684 | 0.520 | 7334 | 653 | 0.768 | 0.768 | 0.944 | 0.0712 | 0.173 | 0.996 | 24089 | — | — |
| bn_folded_seed43 | bn_folded | 100 | 43 | float | float_val | 0.250 | 0.823 | 0.674 | 0.741 | 0.706 | 3746 | 1814 | 1310 | 0.959 | 0.359 | 430 | 1384 | 0.714 | 1653 | 1302 | 0.787 | 0.783 | 0.939 | 0.1407 | 0.333 | 0.882 | 8908 | — | — |
| bn_folded_seed44 | bn_folded | 100 | 44 | float | float_val | 0.300 | 0.816 | 0.709 | 0.673 | 0.691 | 3405 | 1398 | 1651 | 0.906 | 0.266 | 14 | 1384 | 0.689 | 1396 | 1632 | 0.762 | 0.763 | 0.957 | 0.1467 | 0.444 | 0.903 | 5720 | — | — |
| bn_folded_seed42 | bn_folded | 100 | 42 | int8 | int8_val | 0.150 | 0.878 | 0.361 | 0.876 | 0.511 | 4428 | 7849 | 628 | 0.989 | 0.678 | 31 | 7818 | 0.518 | 7465 | 627 | 0.764 | 0.764 | 0.945 | 0.0703 | 0.176 | 0.996 | 23575 | 207.98 | 232 |
| bn_folded_seed43 | bn_folded | 100 | 43 | int8 | int8_val | 0.250 | 0.821 | 0.676 | 0.737 | 0.705 | 3725 | 1785 | 1331 | 0.958 | 0.349 | 410 | 1375 | 0.712 | 1636 | 1322 | 0.779 | 0.775 | 0.938 | 0.1406 | 0.339 | 0.876 | 8635 | 207.98 | 229 |
| bn_folded_seed44 | bn_folded | 100 | 44 | int8 | int8_val | 0.300 | 0.817 | 0.714 | 0.671 | 0.692 | 3392 | 1361 | 1664 | 0.903 | 0.264 | 15 | 1346 | 0.689 | 1359 | 1646 | 0.759 | 0.761 | 0.957 | 0.1445 | 0.459 | 0.890 | 5296 | 207.98 | 226 |
