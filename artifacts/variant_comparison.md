# Float vs. INT8 by variant

> **Old split:** these results use the old hand-picked train/val/test split. De Chazal-split results are in `artifacts/dechazal/variant_comparison.md`.

Test set, seed 42. `threshold_from`: `int8_val` is the threshold chosen on the INT8 model's
validation predictions (the deployed convention); `float_val` is the float model's own
val-optimal threshold. For `qat`, the float row is the float model before QAT fine-tuning.
`val_f1` is the row's own model's validation F1 at the row's threshold.
Threshold-independent columns: `pr_auc` (average precision) and `roc_auc` on test, and the
fixed-recall operating point `r95_*`: the highest threshold reaching >= 0.95 recall on the
model's own validation predictions, then test precision / recall / FP at that threshold.
`min_int8_levels` is the smallest number of INT8 levels any intermediate tensor spans.

| run | variant | calib_samples | model | threshold_from | threshold | val_f1 | test_precision | test_recall | test_f1 | tp | fp | fn | fp_rec214 | fp_excl_214 | pr_auc | roc_auc | r95_threshold | r95_test_precision | r95_test_recall | r95_fp | tflite_kb | min_int8_levels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | baseline | 100 | float | int8_val | 0.350 | 0.425 | 0.810 | 0.959 | 0.878 | 1202 | 282 | 51 | 79 | 203 | 0.961 | 0.996 | 0.1226 | 0.355 | 0.998 | 2276 | — | — |
| baseline | baseline | 100 | float | float_val | 0.150 | 0.499 | 0.444 | 0.996 | 0.614 | 1248 | 1565 | 5 | 470 | 1095 | 0.961 | 0.996 | 0.1226 | 0.355 | 0.998 | 2276 | — | — |
| baseline | baseline | 100 | int8 | int8_val | 0.350 | 0.488 | 0.433 | 0.917 | 0.589 | 1149 | 1502 | 104 | 1146 | 356 | 0.770 | 0.970 | 0.1367 | 0.241 | 0.991 | 3905 | 208.38 | 5 |
| bn_folded | bn_folded | 100 | float | int8_val | 0.450 | 0.452 | 0.853 | 0.820 | 0.836 | 1028 | 177 | 225 | 6 | 171 | 0.893 | 0.985 | 0.1779 | 0.363 | 0.968 | 2126 | — | — |
| bn_folded | bn_folded | 100 | float | float_val | 0.400 | 0.455 | 0.820 | 0.852 | 0.836 | 1068 | 235 | 185 | 7 | 228 | 0.893 | 0.985 | 0.1779 | 0.363 | 0.968 | 2126 | — | — |
| bn_folded | bn_folded | 100 | int8 | int8_val | 0.450 | 0.451 | 0.852 | 0.821 | 0.836 | 1029 | 179 | 224 | 6 | 173 | 0.891 | 0.985 | 0.1836 | 0.369 | 0.966 | 2068 | 207.98 | 230 |
