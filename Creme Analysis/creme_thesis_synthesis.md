# Creme Thesis Checks

This folder runs a few direct, low-overhead checks of the Cremieux-style thesis against the ARC human testing data we already have locally.

The core question is simple: do humans and models find the same ARC items easy and hard, and how strong is that alignment compared with human-to-human alignment?

## What I tested

- I restricted to Public Eval task pairs with at least 8 human attempts overall.
- I repeatedly split human sessions in half and correlated half-A vs half-B item solve rates. That gives a noisy but useful human-to-human baseline.
- I compared the same human item solve rates to three model profiles from the existing local ARC-AGI-2 model outputs: average model, best single model, and a per-pair oracle.

## Main result

- Human split-half median Pearson correlation is 0.399 with a 95% simulation interval of [0.278, 0.515].
- Human vs average-model item difficulty correlation is 0.402. That lands near the middle of the human split-half distribution (52.3 percentile), so it is not random with respect to humans here.
- Human vs best-single-model correlation is only 0.276. That lands near the bottom of the human split-half distribution (2.3 percentile), which is meaningfully weaker than human-to-human alignment.
- Human vs oracle correlation is 0.343, still below the human split-half median.

## Synthesis

- This ARC evidence does not support the strongest version of his claim, namely that model item performance is basically random with respect to humans. The average-model profile is clearly related to human difficulty.
- But it does support a milder and more defensible version: a single strong model still tracks human difficulty worse than one human subsample tracks another, so score comparability is not something we should assume.
- The average-model result should be interpreted cautiously because it is an ensemble-style aggregate across many systems, not one agent.
- Sparse human coverage matters a lot here. Because most items only have around 8 to 15 human attempts, the human split-half ceiling is itself noisy and fairly low.

## Benchmark table

```text
                    series  pearson  spearman  hard_jaccard  easy_jaccard  percentile_vs_split_half  perm_p
   Human split-half median    0.399     0.427         0.135         0.235                     0.500     NaN
    Human vs average model    0.402     0.454         0.100         0.419                     0.523   0.000
Human vs best single model    0.276     0.276         0.000         0.257                     0.023   0.003
  Human vs per-pair oracle    0.343     0.346         0.100         0.375                     0.179   0.000
```

## Gap table

```text
       comparison  human_mean  model_mean  mean_gap  median_gap  share_human_gt  share_equal  share_model_gt
    Average model       0.595       0.108     0.487       0.479           0.973        0.000           0.027
Best single model       0.595       0.491     0.104       0.200           0.509        0.064           0.427
  Per-pair oracle       0.595       0.718    -0.123      -0.222           0.282        0.064           0.655
```

## Biggest divergences vs the best single model

```text
task_pair_id  attempts  solve_rate  lm_mean  lm_best_single_model  gap_vs_lm_mean  gap_vs_best_single_model           direction
 a47bf94d__0        10       0.900    0.026                     0           0.874                     0.900 Human > best single
 8698868d__0        10       0.900    0.077                     0           0.823                     0.900 Human > best single
 a25697e4__1         9       0.889    0.000                     0           0.889                     0.889 Human > best single
 78332cb0__1         9       0.889    0.000                     0           0.889                     0.889 Human > best single
 135a2760__0         9       0.889    0.077                     0           0.812                     0.889 Human > best single
 7b5033c1__0         9       0.889    0.103                     0           0.786                     0.889 Human > best single
 53fb4810__0         9       0.889    0.205                     0           0.684                     0.889 Human > best single
 3dc255db__0         8       0.875    0.077                     0           0.798                     0.875 Human > best single
 de809cff__0        10       0.800    0.000                     0           0.800                     0.800 Human > best single
 28a6681f__0        10       0.800    0.000                     0           0.800                     0.800 Human > best single
 f931b4a8__0        15       0.133    0.256                     1          -0.123                    -0.867 Best single > human
 7b3084d4__0         9       0.222    0.051                     1           0.171                    -0.778 Best single > human
 6e4f6532__1        10       0.300    0.026                     1           0.274                    -0.700 Best single > human
 65b59efc__1         9       0.333    0.282                     1           0.051                    -0.667 Best single > human
 221dfab4__1        11       0.364    0.026                     1           0.338                    -0.636 Best single > human
 88bcf3b4__0         8       0.375    0.026                     1           0.349                    -0.625 Best single > human
 20270e3b__0         8       0.375    0.051                     1           0.324                    -0.625 Best single > human
 1ae2feb7__1         8       0.375    0.436                     1          -0.061                    -0.625 Best single > human
 1ae2feb7__2         8       0.375    0.436                     1          -0.061                    -0.625 Best single > human
 e87109e9__0        10       0.400    0.051                     1           0.349                    -0.600 Best single > human
```
