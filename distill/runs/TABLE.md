
## Summary (mean per extractor x suite)

| extractor | suite | n | seq_agree | exact_count |
|---|---|---|---|---|
| sae_decompile | interp | 20 | 0.945 | 7/20 |
| sae_decompile | messy | 6 | 0.829 | 0/6 |
| tl_decompile | interp | 20 | 0.680 | 5/20 |
| tl_decompile | messy | 6 | 0.785 | 0/6 |
| tl_decompile+cegis | messy | 6 | 0.785 | 0/6 |
| tprogram | interp | 18 | 0.897 | 4/18 |
| tprogram | messy | 6 | 0.759 | 0/6 |
| tprogram | rasp | 10 | 0.883 | 3/10 |
| tracr_exact | rasp | 12 | 0.998 | 10/12 |
| tracr_exact+cegis | rasp | 12 | 0.998 | 11/12 |
| tree | interp | 20 | 0.950 | 8/20 |
| tree | messy | 6 | 0.928 | 0/6 |
| tree | rasp | 12 | 0.946 | 4/12 |
| tree+cegis | messy | 6 | 0.929 | 0/6 |

| suite | case | extractor | seq_agree | pos_agree | TV | exact | cex_rate | nodes | s |
|---|---|---|---|---|---|---|---|---|---|
| interp | 11 | sae_decompile | 1.000 | 1.000 | 0.0003 |  | 0.00e+00 | 11 | 7.3 |
| interp | 11 | tl_decompile | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 47 | 7.1 |
| interp | 11 | tprogram | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 2 | 201.9 |
| interp | 11 | tree | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 13 | 2.5 |
| interp | 13 | sae_decompile | 0.787 | 0.972 | 0.0357 |  | 2.12e-01 | 12 | 8.2 |
| interp | 13 | tl_decompile | 0.128 | 0.794 | 0.2809 |  | 8.71e-01 | 22 | 8.1 |
| interp | 13 | tprogram | 0.884 | 0.986 | 0.0198 |  | 1.16e-01 | 2 | 224.7 |
| interp | 13 | tree | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 69 | 3.0 |
| interp | 14 | sae_decompile | 1.000 | 1.000 | 0.0003 | YES | 0.00e+00 | 5 | 3.9 |
| interp | 14 | tl_decompile | 1.000 | 1.000 | 0.0001 |  | 5.08e-05 | 22 | 3.7 |
| interp | 14 | tprogram | 1.000 | 1.000 | 0.0004 |  | 5.08e-05 | 2 | 203.4 |
| interp | 14 | tree | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 19 | 1.0 |
| interp | 15 | sae_decompile | 1.000 | 1.000 | 0.0109 | YES | 0.00e+00 | 14 | 5.4 |
| interp | 15 | tl_decompile | 1.000 | 1.000 | 0.0126 | YES | 0.00e+00 | 32 | 29.7 |
| interp | 15 | tprogram | 1.000 | 1.000 | 0.0127 | YES | 0.00e+00 | 2 | 105.8 |
| interp | 15 | tree | 1.000 | 1.000 | 0.0127 | YES | 0.00e+00 | 39 | 0.6 |
| interp | 18 | sae_decompile | 1.000 | 1.000 | 0.0003 |  | 4.61e-06 | 12 | 41.8 |
| interp | 18 | tl_decompile | 0.066 | 0.673 | 0.4154 |  | 9.33e-01 | 32 | 60.9 |
| interp | 18 | tprogram | 1.000 | 1.000 | 0.0005 |  | 4.61e-06 | 2 | 283.8 |
| interp | 18 | tree | 1.000 | 1.000 | 0.0000 |  | 4.61e-06 | 175 | 36.8 |
| interp | 19 | sae_decompile | 1.000 | 1.000 | 0.0003 | YES | 0.00e+00 | 17 | 164.6 |
| interp | 19 | tl_decompile | 0.103 | 0.849 | 0.2002 |  | 9.00e-01 | 22 | 156.5 |
| interp | 19 | tprogram | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 2 | 538.2 |
| interp | 19 | tree | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 15 | 118.5 |
| interp | 2 | sae_decompile | 1.000 | 1.000 | 0.0025 |  | 1.50e-04 | 16 | 30.5 |
| interp | 2 | tl_decompile | 0.000 | 0.222 | 0.8470 |  | 1.00e+00 | 137 | 626.0 |
| interp | 2 | tprogram | 1.000 | 1.000 | 0.0000 |  | 1.50e-04 | 2 | 348.7 |
| interp | 2 | tree | 0.000 | 0.134 | 0.9172 |  | 1.00e+00 | 7067 | 22.2 |
| interp | 20 | sae_decompile | 1.000 | 1.000 | 0.0002 |  | 0.00e+00 | 11 | 8.0 |
| interp | 20 | tl_decompile | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 67 | 14.7 |
| interp | 20 | tprogram | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 2 | 203.9 |
| interp | 20 | tree | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 15 | 2.7 |
| interp | 21 | sae_decompile | 0.107 | 0.797 | 0.2471 |  | 8.91e-01 | 25 | 17.9 |
| interp | 21 | tl_decompile | 0.131 | 0.812 | 0.2334 |  | 8.66e-01 | 22 | 21.9 |
| interp | 21 | tprogram | 0.188 | 0.849 | 0.2088 |  | 8.13e-01 | 2 | 263.1 |
| interp | 21 | tree | 1.000 | 1.000 | 0.0001 |  | 5.59e-04 | 107 | 8.0 |
| interp | 24 | sae_decompile | 1.000 | 1.000 | 0.0002 | YES | 0.00e+00 | 12 | 9.8 |
| interp | 24 | tl_decompile | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 22 | 3.5 |
| interp | 24 | tprogram | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 2 | 222.9 |
| interp | 24 | tree | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 5 | 3.6 |
| interp | 25 | sae_decompile | 1.000 | 1.000 | 0.0004 |  | 5.08e-04 | 17 | 14.6 |
| interp | 25 | tl_decompile | 0.085 | 0.602 | 0.4846 |  | 9.14e-01 | 22 | 26.4 |
| interp | 25 | tprogram | 0.067 | 0.520 | 0.5288 |  | 9.31e-01 | 2 | 385.1 |
| interp | 25 | tree | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 171 | 5.4 |
| interp | 26 | sae_decompile | 1.000 | 1.000 | 0.0007 | YES | 0.00e+00 | 17 | 7.6 |
| interp | 26 | tl_decompile | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 22 | 12.6 |
| interp | 26 | tprogram | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 2 | 274.8 |
| interp | 26 | tree | 1.000 | 1.000 | 0.0000 | YES | 0.00e+00 | 53 | 3.3 |
| interp | 29 | sae_decompile | 1.000 | 1.000 | 0.0004 |  | 0.00e+00 | 14 | 7.9 |
| interp | 29 | tl_decompile | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 47 | 8.7 |
| interp | 29 | tprogram | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 2 | 209.8 |
| interp | 29 | tree | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 15 | 2.7 |
| interp | 3 | sae_decompile | 1.000 | 1.000 | nan | YES | 0.00e+00 | 33 | 2.8 |
| interp | 3 | tl_decompile | 1.000 | 1.000 | nan | YES | 0.00e+00 | 35 | 0.2 |
| interp | 3 | tree | 1.000 | 1.000 | nan | YES | 0.00e+00 | 657 | 0.4 |
| interp | 30 | sae_decompile | 1.000 | 1.000 | 0.0006 |  | 0.00e+00 | 11 | 9.2 |
| interp | 30 | tl_decompile | 1.000 | 1.000 | 0.0001 |  | 0.00e+00 | 157 | 46.1 |
| interp | 30 | tprogram | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 2 | 289.1 |
| interp | 30 | tree | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 9 | 2.4 |
| interp | 31 | sae_decompile | 1.000 | 1.000 | 0.0002 |  | 0.00e+00 | 11 | 5.7 |
| interp | 31 | tl_decompile | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 52 | 7.4 |
| interp | 31 | tprogram | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 2 | 191.9 |
| interp | 31 | tree | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 7 | 1.7 |
| interp | 33 | sae_decompile | 1.000 | 1.000 | 0.0004 |  | 0.00e+00 | 11 | 5.7 |
| interp | 33 | tl_decompile | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 47 | 7.4 |
| interp | 33 | tprogram | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 2 | 186.4 |
| interp | 33 | tree | 1.000 | 1.000 | 0.0000 |  | 0.00e+00 | 11 | 1.7 |
| interp | 4 | sae_decompile | 1.000 | 1.000 | nan | YES | 0.00e+00 | 20 | 38.4 |
| interp | 4 | tl_decompile | 1.000 | 1.000 | nan | YES | 0.00e+00 | 41 | 38.4 |
| interp | 4 | tree | 0.998 | 1.000 | nan |  | 2.13e-03 | 3531 | 30.8 |
| interp | 7 | sae_decompile | 1.000 | 1.000 | 0.0011 |  | 3.56e-04 | 14 | 10.6 |
| interp | 7 | tl_decompile | 0.085 | 0.602 | 0.4845 |  | 9.14e-01 | 22 | 24.9 |
| interp | 7 | tprogram | 0.999 | 1.000 | 0.0085 |  | 4.57e-04 | 2 | 201.1 |
| interp | 7 | tree | 1.000 | 1.000 | 0.0005 |  | 2.54e-04 | 191 | 3.1 |
| interp | 8 | sae_decompile | 1.000 | 1.000 | 0.0014 |  | 0.00e+00 | 14 | 6.4 |
| interp | 8 | tl_decompile | 1.000 | 1.000 | 0.0014 |  | 0.00e+00 | 62 | 12.0 |
| interp | 8 | tprogram | 1.000 | 1.000 | 0.0014 |  | 0.00e+00 | 2 | 200.4 |
| interp | 8 | tree | 1.000 | 1.000 | 0.0014 |  | 0.00e+00 | 21 | 2.2 |
| messy | m01_heuristic_vote | sae_decompile | 0.875 | 0.875 | 0.1738 |  | 1.24e-01 | 28 | 16.1 |
| messy | m01_heuristic_vote | tl_decompile | 0.899 | 0.899 | 0.1310 |  | 1.04e-01 | 34 | 9.7 |
| messy | m01_heuristic_vote | tl_decompile+cegis | 0.902 | 0.902 | 0.1330 |  | 1.01e-01 | 37 | 84.3 |
| messy | m01_heuristic_vote | tprogram | 0.991 | 0.991 | 0.0246 |  | 1.09e-02 | 2 | 257.1 |
| messy | m01_heuristic_vote | tree | 1.000 | 1.000 | 0.0006 |  | 4.95e-04 | 99 | 7.2 |
| messy | m01_heuristic_vote | tree+cegis | 1.000 | 1.000 | 0.0001 |  | 1.60e-04 | 85 | 32.1 |
| messy | m02_signal_mixture | sae_decompile | 1.000 | 1.000 | 0.0911 |  | 2.07e-01 | 20 | 35.9 |
| messy | m02_signal_mixture | tl_decompile | 0.995 | 0.995 | 0.0797 |  | 2.02e-01 | 26 | 37.3 |
| messy | m02_signal_mixture | tl_decompile+cegis | 0.993 | 0.993 | 0.0795 |  | 1.82e-01 | 26 | 83.1 |
| messy | m02_signal_mixture | tprogram | 1.000 | 1.000 | 0.0697 |  | 1.77e-01 | 2 | 286.4 |
| messy | m02_signal_mixture | tree | 1.000 | 1.000 | 0.0755 |  | 1.77e-01 | 9 | 28.4 |
| messy | m02_signal_mixture | tree+cegis | 1.000 | 1.000 | 0.0755 |  | 1.86e-01 | 9 | 43.6 |
| messy | m03_superposed_counts | sae_decompile | 0.992 | 0.992 | 0.0909 |  | 7.78e-03 | 32 | 12.1 |
| messy | m03_superposed_counts | tl_decompile | 0.993 | 0.993 | 0.0260 |  | 7.62e-03 | 47 | 33.4 |
| messy | m03_superposed_counts | tl_decompile+cegis | 0.992 | 0.992 | 0.0287 |  | 7.77e-03 | 47 | 285.0 |
| messy | m03_superposed_counts | tprogram | 0.787 | 0.787 | 0.2685 |  | 2.13e-01 | 2 | 244.2 |
| messy | m03_superposed_counts | tree | 0.992 | 0.992 | 0.0322 |  | 8.50e-03 | 663 | 2.7 |
| messy | m03_superposed_counts | tree+cegis | 0.990 | 0.990 | 0.0340 |  | 9.63e-03 | 727 | 17.7 |
| messy | m04_multitask | sae_decompile | 0.677 | 0.677 | 0.4026 |  | 3.18e-01 | 40 | 19.0 |
| messy | m04_multitask | tl_decompile | 0.883 | 0.883 | 0.1644 |  | 1.26e-01 | 61 | 40.5 |
| messy | m04_multitask | tl_decompile+cegis | 0.883 | 0.883 | 0.1650 |  | 1.25e-01 | 61 | 338.5 |
| messy | m04_multitask | tprogram | 0.878 | 0.878 | 0.1528 |  | 1.28e-01 | 2 | 237.2 |
| messy | m04_multitask | tree | 0.937 | 0.937 | 0.0820 |  | 6.87e-02 | 585 | 4.1 |
| messy | m04_multitask | tree+cegis | 0.946 | 0.946 | 0.0699 |  | 5.82e-02 | 515 | 26.9 |
| messy | m05_soft_mixture | sae_decompile | 0.894 | 0.894 | 0.1150 |  | 1.07e-01 | 38 | 15.3 |
| messy | m05_soft_mixture | tl_decompile | 0.920 | 0.920 | 0.0154 |  | 8.20e-02 | 52 | 39.1 |
| messy | m05_soft_mixture | tl_decompile+cegis | 0.918 | 0.918 | 0.0152 |  | 8.14e-02 | 52 | 352.2 |
| messy | m05_soft_mixture | tprogram | 0.886 | 0.886 | 0.0913 |  | 1.13e-01 | 2 | 250.8 |
| messy | m05_soft_mixture | tree | 0.925 | 0.925 | 0.0117 |  | 7.62e-02 | 1129 | 4.5 |
| messy | m05_soft_mixture | tree+cegis | 0.925 | 0.925 | 0.0120 |  | 7.61e-02 | 1165 | 29.9 |
| messy | m06_ngram_mixture | sae_decompile | 0.534 | 0.958 | 0.3502 |  | 4.61e-01 | 21 | 22.4 |
| messy | m06_ngram_mixture | tl_decompile | 0.022 | 0.793 | 0.2864 |  | 9.76e-01 | 37 | 65.9 |
| messy | m06_ngram_mixture | tl_decompile+cegis | 0.023 | 0.794 | 0.2860 |  | 9.75e-01 | 37 | 2129.0 |
| messy | m06_ngram_mixture | tprogram | 0.011 | 0.775 | 0.1187 |  | 9.89e-01 | 2 | 384.6 |
| messy | m06_ngram_mixture | tree | 0.717 | 0.982 | 0.0149 |  | 2.79e-01 | 771 | 6.2 |
| messy | m06_ngram_mixture | tree+cegis | 0.713 | 0.981 | 0.0148 |  | 2.84e-01 | 811 | 42.8 |
| rasp | p01_identity | tprogram | 1.000 | 1.000 | nan | YES | 0.00e+00 | 2 | 170.7 |
| rasp | p01_identity | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 3 | 0.4 |
| rasp | p01_identity | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 3 | 0.4 |
| rasp | p01_identity | tree | 1.000 | 1.000 | nan | YES | 0.00e+00 | 7 | 1.1 |
| rasp | p02_increment | tprogram | 1.000 | 1.000 | nan | YES | 0.00e+00 | 2 | 173.4 |
| rasp | p02_increment | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 3 | 1.6 |
| rasp | p02_increment | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 3 | 2.1 |
| rasp | p02_increment | tree | 1.000 | 1.000 | nan | YES | 0.00e+00 | 9 | 1.4 |
| rasp | p03_length | tprogram | 1.000 | 1.000 | nan | YES | 0.00e+00 | 2 | 171.5 |
| rasp | p03_length | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 4 | 0.1 |
| rasp | p03_length | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 5 | 0.1 |
| rasp | p03_length | tree | 1.000 | 1.000 | nan | YES | 0.00e+00 | 15 | 0.3 |
| rasp | p04_shift_right | tprogram | 0.876 | 0.968 | nan |  | 1.62e-01 | 2 | 160.2 |
| rasp | p04_shift_right | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 6 | 0.5 |
| rasp | p04_shift_right | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 6 | 0.7 |
| rasp | p04_shift_right | tree | 1.000 | 1.000 | nan |  | 2.40e-03 | 135 | 0.7 |
| rasp | p05_frac_prevs | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 6 | 0.1 |
| rasp | p05_frac_prevs | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 6 | 0.1 |
| rasp | p05_frac_prevs | tree | 1.000 | 1.000 | nan | YES | 0.00e+00 | 185 | 0.3 |
| rasp | p06_hist | tprogram | 0.999 | 0.999 | nan |  | 5.39e-03 | 2 | 157.2 |
| rasp | p06_hist | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 4 | 0.1 |
| rasp | p06_hist | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 5 | 0.1 |
| rasp | p06_hist | tree | 1.000 | 1.000 | nan |  | 1.02e-04 | 179 | 0.5 |
| rasp | p07_pair_balance | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 10 | 0.3 |
| rasp | p07_pair_balance | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 10 | 0.5 |
| rasp | p07_pair_balance | tree | 1.000 | 1.000 | nan |  | 8.13e-04 | 545 | 0.9 |
| rasp | p08_reverse | tprogram | 0.264 | 0.669 | nan |  | 9.72e-01 | 2 | 178.2 |
| rasp | p08_reverse | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 10 | 0.6 |
| rasp | p08_reverse | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 11 | 1.0 |
| rasp | p08_reverse | tree | 0.507 | 0.826 | nan |  | 9.65e-01 | 3397 | 2.2 |
| rasp | p09_detect_pattern | tprogram | 0.997 | 0.999 | nan |  | 3.05e-03 | 2 | 171.9 |
| rasp | p09_detect_pattern | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 12 | 0.6 |
| rasp | p09_detect_pattern | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 12 | 0.9 |
| rasp | p09_detect_pattern | tree | 0.967 | 0.995 | nan |  | 9.48e-02 | 77 | 1.3 |
| rasp | p10_sort | tprogram | 0.729 | 0.946 | nan |  | 6.15e-01 | 2 | 191.7 |
| rasp | p10_sort | tracr_exact | 1.000 | 1.000 | nan | YES | 0.00e+00 | 9 | 13.1 |
| rasp | p10_sort | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 10 | 26.1 |
| rasp | p10_sort | tree | 0.890 | 0.982 | nan |  | 2.92e-01 | 747 | 12.8 |
| rasp | p11_dyck1 | tprogram | 0.975 | 0.975 | nan |  | 1.03e-02 | 2 | 294.9 |
| rasp | p11_dyck1 | tracr_exact | 0.997 | 0.997 | nan |  | 2.05e-02 | 24 | 1.8 |
| rasp | p11_dyck1 | tracr_exact+cegis | 1.000 | 1.000 | nan | YES | 0.00e+00 | 25 | 2.5 |
| rasp | p11_dyck1 | tree | 0.988 | 0.992 | nan |  | 8.80e-03 | 67 | 6.2 |
| rasp | p12_dyck2 | tprogram | 0.992 | 0.994 | nan |  | 1.83e-03 | 2 | 946.2 |
| rasp | p12_dyck2 | tracr_exact | 0.981 | 0.981 | nan |  | 4.79e-04 | 42 | 551.4 |
| rasp | p12_dyck2 | tracr_exact+cegis | 0.981 | 0.981 | nan |  | 4.79e-04 | 43 | 599.6 |
| rasp | p12_dyck2 | tree | 0.996 | 0.996 | nan |  | 4.71e-04 | 115 | 535.6 |
