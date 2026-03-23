# Human Testing Psychometric Analysis

## Scope

This report analyzes the locally downloaded `arc_agi_2_human_testing` file (`Human data/test_pair_attempts.csv`) in the context of the ARC-AGI-2 task JSONs and the ARC-AGI-2 model prediction folder in this workspace.

## Data Structure

- 4681 human attempts across 509 sessions, 442 task IDs, and 502 task-test pairs.
- Human coverage spans 40.4% of all ARC-AGI-2 public task pairs (502 of 1243).
- The observed session-by-item matrix is extremely sparse: 1.83% density.
- One warm-up pair (`0a1d4ef5__0`) appears in 463 sessions with a 92.9% solve rate.

## Exploratory Findings

- Overall human solve rate is 75.3%; excluding the warm-up item it is still 73.4%.
- Public Train is much easier than Public Eval: 80.4% vs 61.3%.
- Sessions are short and uneven: median 8.0 attempts per session, mean 9.20.
- Items are also unevenly exposed: median 9.0 attempts per task pair.

## Psychometric Results

- A regularized person-plus-item logistic model fits the sparse matrix well (AUC 0.952, log loss 0.279, Brier 0.082).
- Item difficulty tracks human time cost more strongly than raw task size: item solve rate vs mean duration correlation is about -0.387.
- Raw visual size is a weak standalone predictor: item solve rate vs input cells correlation is about -0.028.
- Discrimination estimates are usable but noisy because most items only have around 9 exposures.

## Sampling Caveat

The human sample is not representative of the full ARC-AGI-2 pool. Attempted tasks are systematically larger than unattempted tasks, especially on Public Train.

```text
    task_set  attempted  n_tasks  mean_input_cells  mean_input_colors  mean_train_pairs  mean_test_pairs attempted_label
 Public Eval        0.0        5           269.160              6.297             3.600            1.200     Unattempted
 Public Eval        1.0      115           424.789              5.447             2.965            1.400       Attempted
Public Train        0.0      674           145.899              3.748             3.227            1.067     Unattempted
Public Train        1.0      326           293.059              4.356             3.242            1.095       Attempted
```

## Public Eval Human vs LM Cross-Reference

- Direct overlap covers 161 Public Eval task pairs; 110 of those have at least 8 human attempts.
- On the >=8-attempt overlap, humans average 59.5%, the average model in the current local prediction folder averages 10.8%, and the best single model (`gpt-5-2-2025-12-11-thinking-xhigh`) averages 49.1%.
- The average per-pair oracle over all local models is 71.8%, which is higher than the average human but is not achievable by any single model.
- Human and average-model difficulty are only moderately aligned (r = 0.402 on >=8-attempt pairs).
- Humans outperform the average model on 97.3% of well-sampled Public Eval pairs and outperform the best single model on 50.9%.

Robust human-advantage items (>=8 attempts):

```text
task_pair_id  solve_rate  attempts  lm_mean  lm_best_single_model  gap_vs_lm_mean
 a25697e4__1       0.889         9    0.000                     0           0.889
 78332cb0__1       0.889         9    0.000                     0           0.889
 a47bf94d__0       0.900        10    0.026                     0           0.874
 8698868d__0       0.900        10    0.077                     0           0.823
 135a2760__0       0.889         9    0.077                     0           0.812
```

Robust model-advantage items (>=8 attempts):

```text
task_pair_id  solve_rate  attempts  lm_mean  lm_best_single_model  gap_vs_lm_mean
 f931b4a8__0       0.133        15    0.256                     1          -0.123
 1ae2feb7__1       0.375         8    0.436                     1          -0.061
 1ae2feb7__2       0.375         8    0.436                     1          -0.061
 65b59efc__1       0.333         9    0.282                     1           0.051
 1ae2feb7__0       0.500         8    0.385                     1           0.115
```

Hardest gallery items:

```text
task_pair_id     task_set  solve_rate  attempts  mean_duration_seconds
 f931b4a8__0  Public Eval       0.133        15                216.386
 f3e14006__1 Public Train       0.200        10                195.204
 34b99a2b__0 Public Train       0.333         9                240.825
 1c56ad9f__0 Public Train       0.250         8                402.579
```

Easiest gallery items:

```text
task_pair_id     task_set  solve_rate  attempts  mean_duration_seconds
 66ac4c3b__0 Public Train       1.000         8                213.334
 9b5080bb__0 Public Train       1.000         9                204.819
 7e2bad24__1 Public Train       1.000         9                 80.439
 0a1d4ef5__0 Public Train       0.929       463                177.228
```

## Interpretation

- The human data do support a coherent difficulty structure, but this is an opportunistic sparse sample rather than a balanced exam form.
- Public Eval remains challenging for humans, yet the human floor is much higher than the average model floor observed in the earlier LM psychometric work.
- Difficulty is only partly shared between humans and models, suggesting some common latent structure plus substantial human-specific advantages in flexible abstraction.
- Unlike the earlier LM heterogeneity report, the main limitation here is not deterministic over-consistency; it is sparse, biased coverage and low item exposure.
