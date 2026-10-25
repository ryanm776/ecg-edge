# Step 2b: R-R-only S-vs-N baseline under the grouped folds

Each classifier is trained on DS1 minus the held-out fold (201 always trains; N capped at 600 per record like train_cv.py; V beats dropped) and scores the held-out fold. `logreg`: balanced logistic regression on standardized log features. `hgb`: balanced HistGradientBoosting. Columns are OOF S-vs-N ROC-AUC, pooled and within each patient with >= 20 S beats. `5 features, <ratio>` swaps rr_ratio for that ratio; `5 features, ratio_mean10` is the current input set. CNN rows are the grouped-CV fold models' OOF INT8 scores on the same beats. 10 N/S beats with undefined beat-only features are dropped from every row.

## ROC-AUC

| source | version | feature_set | pooled | 118 | 124 | 207 | 209 | 220 | 223 |
|---|---|---|---|---|---|---|---|---|---|
| R-R only, logreg | all_ann | 5 features, ratio_mean10 | 0.916 | 0.982 | 0.812 | 0.988 | 0.981 | 0.988 | 0.980 |
| R-R only, logreg | all_ann | 5 features, ratio_q5010 | 0.918 | 0.983 | 0.844 | 0.988 | 0.981 | 0.982 | 0.979 |
| R-R only, logreg | all_ann | 5 features, ratio_q9010 | 0.915 | 0.982 | 0.788 | 0.988 | 0.983 | 0.986 | 0.976 |
| R-R only, logreg | all_ann | 5 features, ratio_max10 | 0.914 | 0.982 | 0.790 | 0.988 | 0.983 | 0.986 | 0.975 |
| R-R only, logreg | all_ann | 5 features, ratio_mean30 | 0.923 | 0.984 | 0.819 | 0.988 | 0.984 | 0.983 | 0.978 |
| R-R only, logreg | all_ann | 5 features, ratio_q5030 | 0.923 | 0.983 | 0.846 | 0.988 | 0.985 | 0.983 | 0.978 |
| R-R only, logreg | all_ann | 5 features, ratio_q9030 | 0.916 | 0.983 | 0.832 | 0.988 | 0.984 | 0.990 | 0.981 |
| R-R only, logreg | all_ann | 5 features, ratio_max30 | 0.914 | 0.982 | 0.816 | 0.988 | 0.983 | 0.991 | 0.981 |
| R-R only, logreg | all_ann | ratio_mean10 alone | 0.727 | 0.980 | 0.262 | 0.581 | 0.756 | 0.949 | 0.946 |
| R-R only, logreg | all_ann | ratio_q5010 alone | 0.722 | 0.988 | 0.566 | 0.575 | 0.731 | 0.894 | 0.952 |
| R-R only, logreg | all_ann | ratio_q9010 alone | 0.699 | 0.986 | 0.710 | 0.564 | 0.726 | 0.995 | 0.965 |
| R-R only, logreg | all_ann | ratio_max10 alone | 0.679 | 0.936 | 0.753 | 0.620 | 0.699 | 0.994 | 0.931 |
| R-R only, logreg | all_ann | ratio_mean30 alone | 0.849 | 0.985 | 0.595 | 0.678 | 0.897 | 0.966 | 0.953 |
| R-R only, logreg | all_ann | ratio_q5030 alone | 0.819 | 0.987 | 0.655 | 0.615 | 0.851 | 0.956 | 0.955 |
| R-R only, logreg | all_ann | ratio_q9030 alone | 0.820 | 0.984 | 0.783 | 0.682 | 0.873 | 0.995 | 0.943 |
| R-R only, logreg | all_ann | ratio_max30 alone | 0.784 | 0.887 | 0.794 | 0.749 | 0.839 | 0.994 | 0.898 |
| R-R only, logreg | all_ann | pre_post_ratio alone | 0.675 | 0.976 | 0.472 | 0.547 | 0.705 | 0.814 | 0.939 |
| R-R only, logreg | beat_only | 5 features, ratio_mean10 | 0.929 | 0.996 | 0.836 | 0.989 | 0.992 | 0.993 | 0.996 |
| R-R only, logreg | beat_only | 5 features, ratio_q5010 | 0.932 | 0.998 | 0.844 | 0.998 | 0.990 | 0.995 | 0.989 |
| R-R only, logreg | beat_only | 5 features, ratio_q9010 | 0.928 | 0.995 | 0.796 | 0.990 | 0.992 | 0.988 | 0.982 |
| R-R only, logreg | beat_only | 5 features, ratio_max10 | 0.929 | 0.996 | 0.790 | 0.992 | 0.991 | 0.991 | 0.981 |
| R-R only, logreg | beat_only | 5 features, ratio_mean30 | 0.932 | 0.998 | 0.843 | 0.972 | 0.995 | 0.993 | 0.990 |
| R-R only, logreg | beat_only | 5 features, ratio_q5030 | 0.936 | 0.999 | 0.837 | 0.994 | 0.991 | 0.996 | 0.987 |
| R-R only, logreg | beat_only | 5 features, ratio_q9030 | 0.929 | 0.997 | 0.839 | 0.986 | 0.993 | 0.992 | 0.996 |
| R-R only, logreg | beat_only | 5 features, ratio_max30 | 0.928 | 0.997 | 0.829 | 0.988 | 0.992 | 0.993 | 0.995 |
| R-R only, logreg | beat_only | ratio_mean10 alone | 0.773 | 0.984 | 0.592 | 0.568 | 0.799 | 0.991 | 0.989 |
| R-R only, logreg | beat_only | ratio_q5010 alone | 0.765 | 0.993 | 0.599 | 0.586 | 0.769 | 1.000 | 0.977 |
| R-R only, logreg | beat_only | ratio_q9010 alone | 0.723 | 0.990 | 0.712 | 0.572 | 0.750 | 1.000 | 0.969 |
| R-R only, logreg | beat_only | ratio_max10 alone | 0.698 | 0.932 | 0.763 | 0.629 | 0.722 | 0.999 | 0.937 |
| R-R only, logreg | beat_only | ratio_mean30 alone | 0.862 | 0.988 | 0.665 | 0.648 | 0.910 | 0.990 | 0.972 |
| R-R only, logreg | beat_only | ratio_q5030 alone | 0.839 | 0.992 | 0.664 | 0.633 | 0.872 | 0.999 | 0.965 |
| R-R only, logreg | beat_only | ratio_q9030 alone | 0.831 | 0.989 | 0.772 | 0.678 | 0.888 | 1.000 | 0.955 |
| R-R only, logreg | beat_only | ratio_max30 alone | 0.786 | 0.862 | 0.783 | 0.743 | 0.849 | 0.999 | 0.900 |
| R-R only, logreg | beat_only | pre_post_ratio alone | 0.678 | 0.987 | 0.482 | 0.537 | 0.701 | 0.738 | 0.981 |
| R-R only, hgb | all_ann | 5 features, ratio_mean10 | 0.934 | 0.998 | 0.474 | 0.846 | 0.995 | 0.970 | 0.974 |
| R-R only, hgb | all_ann | 5 features, ratio_q5010 | 0.928 | 0.999 | 0.303 | 0.776 | 0.995 | 0.962 | 0.976 |
| R-R only, hgb | all_ann | 5 features, ratio_q9010 | 0.906 | 0.999 | 0.706 | 0.540 | 0.996 | 0.971 | 0.990 |
| R-R only, hgb | all_ann | 5 features, ratio_max10 | 0.917 | 0.998 | 0.568 | 0.674 | 0.996 | 0.966 | 0.987 |
| R-R only, hgb | all_ann | 5 features, ratio_mean30 | 0.917 | 0.999 | 0.731 | 0.744 | 0.995 | 0.985 | 0.973 |
| R-R only, hgb | all_ann | 5 features, ratio_q5030 | 0.919 | 0.999 | 0.775 | 0.709 | 0.995 | 0.992 | 0.966 |
| R-R only, hgb | all_ann | 5 features, ratio_q9030 | 0.914 | 0.999 | 0.917 | 0.624 | 0.996 | 0.980 | 0.965 |
| R-R only, hgb | all_ann | 5 features, ratio_max30 | 0.920 | 0.998 | 0.757 | 0.752 | 0.995 | 0.979 | 0.981 |
| R-R only, hgb | all_ann | ratio_mean10 alone | 0.735 | 0.987 | 0.490 | 0.630 | 0.724 | 0.966 | 0.929 |
| R-R only, hgb | all_ann | ratio_q5010 alone | 0.738 | 0.986 | 0.505 | 0.567 | 0.691 | 0.921 | 0.954 |
| R-R only, hgb | all_ann | ratio_q9010 alone | 0.720 | 0.967 | 0.631 | 0.569 | 0.718 | 0.991 | 0.921 |
| R-R only, hgb | all_ann | ratio_max10 alone | 0.682 | 0.915 | 0.660 | 0.532 | 0.680 | 0.974 | 0.854 |
| R-R only, hgb | all_ann | ratio_mean30 alone | 0.847 | 0.986 | 0.661 | 0.671 | 0.878 | 0.975 | 0.941 |
| R-R only, hgb | all_ann | ratio_q5030 alone | 0.816 | 0.987 | 0.762 | 0.612 | 0.826 | 0.969 | 0.953 |
| R-R only, hgb | all_ann | ratio_q9030 alone | 0.810 | 0.985 | 0.733 | 0.638 | 0.857 | 0.990 | 0.890 |
| R-R only, hgb | all_ann | ratio_max30 alone | 0.766 | 0.879 | 0.623 | 0.653 | 0.830 | 0.982 | 0.818 |
| R-R only, hgb | all_ann | pre_post_ratio alone | 0.689 | 0.985 | 0.548 | 0.572 | 0.665 | 0.891 | 0.912 |
| R-R only, hgb | beat_only | 5 features, ratio_mean10 | 0.929 | 0.999 | 0.931 | 0.592 | 0.995 | 0.986 | 0.978 |
| R-R only, hgb | beat_only | 5 features, ratio_q5010 | 0.923 | 1.000 | 0.822 | 0.738 | 0.995 | 0.986 | 0.962 |
| R-R only, hgb | beat_only | 5 features, ratio_q9010 | 0.930 | 0.999 | 0.850 | 0.586 | 0.997 | 0.983 | 0.991 |
| R-R only, hgb | beat_only | 5 features, ratio_max10 | 0.928 | 0.999 | 0.725 | 0.715 | 0.993 | 0.986 | 0.989 |
| R-R only, hgb | beat_only | 5 features, ratio_mean30 | 0.923 | 0.999 | 0.379 | 0.603 | 0.991 | 0.996 | 0.976 |
| R-R only, hgb | beat_only | 5 features, ratio_q5030 | 0.909 | 1.000 | 0.382 | 0.643 | 0.993 | 0.996 | 0.943 |
| R-R only, hgb | beat_only | 5 features, ratio_q9030 | 0.899 | 0.999 | 0.808 | 0.627 | 0.994 | 0.995 | 0.983 |
| R-R only, hgb | beat_only | 5 features, ratio_max30 | 0.906 | 0.999 | 0.904 | 0.626 | 0.988 | 0.985 | 0.973 |
| R-R only, hgb | beat_only | ratio_mean10 alone | 0.780 | 0.982 | 0.622 | 0.627 | 0.774 | 0.992 | 0.960 |
| R-R only, hgb | beat_only | ratio_q5010 alone | 0.769 | 0.986 | 0.532 | 0.556 | 0.747 | 0.999 | 0.965 |
| R-R only, hgb | beat_only | ratio_q9010 alone | 0.724 | 0.968 | 0.532 | 0.571 | 0.723 | 0.996 | 0.921 |
| R-R only, hgb | beat_only | ratio_max10 alone | 0.702 | 0.900 | 0.654 | 0.552 | 0.708 | 0.991 | 0.864 |
| R-R only, hgb | beat_only | ratio_mean30 alone | 0.877 | 0.991 | 0.756 | 0.695 | 0.903 | 0.991 | 0.951 |
| R-R only, hgb | beat_only | ratio_q5030 alone | 0.824 | 0.988 | 0.703 | 0.636 | 0.825 | 0.991 | 0.959 |
| R-R only, hgb | beat_only | ratio_q9030 alone | 0.820 | 0.983 | 0.705 | 0.652 | 0.880 | 0.992 | 0.911 |
| R-R only, hgb | beat_only | ratio_max30 alone | 0.760 | 0.844 | 0.559 | 0.655 | 0.826 | 0.993 | 0.806 |
| R-R only, hgb | beat_only | pre_post_ratio alone | 0.694 | 0.995 | 0.518 | 0.559 | 0.662 | 0.882 | 0.913 |
| CNN int8 seed 42 | all_ann | spectrogram + 5 | 0.781 | 0.843 | 0.569 | 0.975 | 0.583 | 0.994 | 0.959 |
| CNN int8 seed 43 | all_ann | spectrogram + 5 | 0.707 | 0.964 | 0.589 | 0.983 | 0.869 | 0.974 | 0.951 |
| CNN int8 seed 44 | all_ann | spectrogram + 5 | 0.826 | 0.960 | 0.558 | 0.988 | 0.964 | 0.977 | 0.987 |

## Pooled PR-AUC

| source | version | feature_set | pr_auc |
|---|---|---|---|
| R-R only, logreg | all_ann | 5 features, ratio_mean10 | 0.166 |
| R-R only, hgb | all_ann | 5 features, ratio_mean10 | 0.462 |
| R-R only, logreg | all_ann | 5 features, ratio_q5010 | 0.171 |
| R-R only, hgb | all_ann | 5 features, ratio_q5010 | 0.442 |
| R-R only, logreg | all_ann | 5 features, ratio_q9010 | 0.161 |
| R-R only, hgb | all_ann | 5 features, ratio_q9010 | 0.477 |
| R-R only, logreg | all_ann | 5 features, ratio_max10 | 0.159 |
| R-R only, hgb | all_ann | 5 features, ratio_max10 | 0.494 |
| R-R only, logreg | all_ann | 5 features, ratio_mean30 | 0.184 |
| R-R only, hgb | all_ann | 5 features, ratio_mean30 | 0.418 |
| R-R only, logreg | all_ann | 5 features, ratio_q5030 | 0.184 |
| R-R only, hgb | all_ann | 5 features, ratio_q5030 | 0.408 |
| R-R only, logreg | all_ann | 5 features, ratio_q9030 | 0.161 |
| R-R only, hgb | all_ann | 5 features, ratio_q9030 | 0.449 |
| R-R only, logreg | all_ann | 5 features, ratio_max30 | 0.156 |
| R-R only, hgb | all_ann | 5 features, ratio_max30 | 0.477 |
| R-R only, logreg | all_ann | ratio_mean10 alone | 0.106 |
| R-R only, hgb | all_ann | ratio_mean10 alone | 0.118 |
| R-R only, logreg | all_ann | ratio_q5010 alone | 0.116 |
| R-R only, hgb | all_ann | ratio_q5010 alone | 0.125 |
| R-R only, logreg | all_ann | ratio_q9010 alone | 0.073 |
| R-R only, hgb | all_ann | ratio_q9010 alone | 0.067 |
| R-R only, logreg | all_ann | ratio_max10 alone | 0.059 |
| R-R only, hgb | all_ann | ratio_max10 alone | 0.048 |
| R-R only, logreg | all_ann | ratio_mean30 alone | 0.159 |
| R-R only, hgb | all_ann | ratio_mean30 alone | 0.213 |
| R-R only, logreg | all_ann | ratio_q5030 alone | 0.155 |
| R-R only, hgb | all_ann | ratio_q5030 alone | 0.180 |
| R-R only, logreg | all_ann | ratio_q9030 alone | 0.102 |
| R-R only, hgb | all_ann | ratio_q9030 alone | 0.102 |
| R-R only, logreg | all_ann | ratio_max30 alone | 0.080 |
| R-R only, hgb | all_ann | ratio_max30 alone | 0.081 |
| R-R only, logreg | all_ann | pre_post_ratio alone | 0.097 |
| R-R only, hgb | all_ann | pre_post_ratio alone | 0.116 |
| R-R only, logreg | beat_only | 5 features, ratio_mean10 | 0.240 |
| R-R only, hgb | beat_only | 5 features, ratio_mean10 | 0.499 |
| R-R only, logreg | beat_only | 5 features, ratio_q5010 | 0.262 |
| R-R only, hgb | beat_only | 5 features, ratio_q5010 | 0.483 |
| R-R only, logreg | beat_only | 5 features, ratio_q9010 | 0.247 |
| R-R only, hgb | beat_only | 5 features, ratio_q9010 | 0.491 |
| R-R only, logreg | beat_only | 5 features, ratio_max10 | 0.254 |
| R-R only, hgb | beat_only | 5 features, ratio_max10 | 0.490 |
| R-R only, logreg | beat_only | 5 features, ratio_mean30 | 0.265 |
| R-R only, hgb | beat_only | 5 features, ratio_mean30 | 0.513 |
| R-R only, logreg | beat_only | 5 features, ratio_q5030 | 0.288 |
| R-R only, hgb | beat_only | 5 features, ratio_q5030 | 0.520 |
| R-R only, logreg | beat_only | 5 features, ratio_q9030 | 0.232 |
| R-R only, hgb | beat_only | 5 features, ratio_q9030 | 0.483 |
| R-R only, logreg | beat_only | 5 features, ratio_max30 | 0.238 |
| R-R only, hgb | beat_only | 5 features, ratio_max30 | 0.487 |
| R-R only, logreg | beat_only | ratio_mean10 alone | 0.178 |
| R-R only, hgb | beat_only | ratio_mean10 alone | 0.172 |
| R-R only, logreg | beat_only | ratio_q5010 alone | 0.201 |
| R-R only, hgb | beat_only | ratio_q5010 alone | 0.205 |
| R-R only, logreg | beat_only | ratio_q9010 alone | 0.092 |
| R-R only, hgb | beat_only | ratio_q9010 alone | 0.077 |
| R-R only, logreg | beat_only | ratio_max10 alone | 0.067 |
| R-R only, hgb | beat_only | ratio_max10 alone | 0.052 |
| R-R only, logreg | beat_only | ratio_mean30 alone | 0.246 |
| R-R only, hgb | beat_only | ratio_mean30 alone | 0.302 |
| R-R only, logreg | beat_only | ratio_q5030 alone | 0.266 |
| R-R only, hgb | beat_only | ratio_q5030 alone | 0.250 |
| R-R only, logreg | beat_only | ratio_q9030 alone | 0.131 |
| R-R only, hgb | beat_only | ratio_q9030 alone | 0.118 |
| R-R only, logreg | beat_only | ratio_max30 alone | 0.088 |
| R-R only, hgb | beat_only | ratio_max30 alone | 0.074 |
| R-R only, logreg | beat_only | pre_post_ratio alone | 0.158 |
| R-R only, hgb | beat_only | pre_post_ratio alone | 0.132 |
| CNN int8 seed 42 | all_ann | spectrogram + 5 | 0.078 |
| CNN int8 seed 43 | all_ann | spectrogram + 5 | 0.045 |
| CNN int8 seed 44 | all_ann | spectrogram + 5 | 0.101 |
