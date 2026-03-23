# Temporal / Family Hypothesis Checks

This is a quick look for the kind of temporal or family-pattern hypothesis you asked about: are newer models just stronger versions of the same profile, or do they become more human-like in their item pattern?

## Constraints

- We do not have `Opus 4` in the local ARC-AGI-2 Public Eval folder, only `Opus 4.5`, so there is no direct `Opus 4` vs `Opus 4.5` time comparison to run here.
- We do have useful temporal comparisons inside the GPT family (`gpt-5.1` to `gpt-5.2`) and compute-budget comparisons inside `Claude Opus 4.5` and `Gemini Flash`.

## Keepers

- Across the whole non-degenerate model panel, higher pair accuracy tends to come with higher human-alignment: correlation between model accuracy and human-correlation is 0.476 Pearson and 0.583 Spearman.
- In GPT, `5.2` is not just more accurate than `5.1`; it is also more human-aligned on the item pattern. The cleanest cases are `5.1 med -> 5.2 med` (0.082 -> 0.227 accuracy; 0.160 -> 0.306 human-correlation) and `5.1 high -> 5.2 high` (0.109 -> 0.291; 0.101 -> 0.224).
- But the gains mostly look like `more of the same but better`. For `5.1 med -> 5.2 med`, newly solved items have mean human solve rate 0.719, while items the new model still fails average 0.553. The same pattern holds for `5.1 high -> 5.2 high` (0.667 vs 0.565).
- `Claude Opus 4.5` is interesting because it already becomes very human-aligned at moderate budgets: `16k` reaches human-correlation 0.439 and `64k` is 0.430. Accuracy rises a lot (0.164 -> 0.291) but the human-correlation barely moves, which again looks like scaling up within the same ordering rather than a structural shift.
- A notable exception to a pure `score = human-likeness` story is that some `Opus 4.5` settings are more human-aligned than much higher-scoring GPT settings. So the overall trend is positive, but raw accuracy and human-style item pattern are not the same thing.

## Rejections / weak ideas

- I do not see a strong structural-break story here. The data are more consistent with newer or higher-budget models extending along an existing human difficulty gradient.
- The correlation gains within GPT are directionally positive, but bootstrap intervals on the correlation deltas are still fairly wide on this item set, so I would keep those as `suggestive`, not `definitive`.

## Tables

```text
                                 model          family  pair_accuracy  human_pearson
     gpt-5-2-2025-12-11-thinking-xhigh      OpenAI GPT          0.491          0.276
           gpt-5-2-pro-2025-12-11-high      OpenAI GPT          0.464          0.273
           gemini-3-deep-think-preview Gemini Pro/Deep          0.355          0.235
         gpt-5-2-pro-2025-12-11-medium      OpenAI GPT          0.300          0.273
      gpt-5-2-2025-12-11-thinking-high      OpenAI GPT          0.291          0.224
 claude-opus-4-5-20251101-thinking-64k     Claude Opus          0.291          0.430
  gemini-3-flash-preview-thinking-high    Gemini Flash          0.245          0.322
    gpt-5-2-2025-12-11-thinking-medium      OpenAI GPT          0.227          0.306
                  gemini-3-pro-preview Gemini Pro/Deep          0.218          0.289
 claude-opus-4-5-20251101-thinking-32k     Claude Opus          0.200          0.351
 claude-opus-4-5-20251101-thinking-16k     Claude Opus          0.164          0.439
                  gpt-5-pro-2025-10-06      OpenAI GPT          0.118          0.072
      gpt-5-1-2025-11-13-thinking-high      OpenAI GPT          0.109          0.101
gemini-3-flash-preview-thinking-medium    Gemini Flash          0.091          0.246
    gpt-5-1-2025-11-13-thinking-medium      OpenAI GPT          0.082          0.160
```

```text
              comparison  delta_accuracy  delta_human_corr  delta_human_corr_ci_lo  delta_human_corr_ci_hi  newly_solved_human_mean  still_failed_human_mean
  GPT 5.1 low -> 5.2 low           0.064             0.029                     NaN                     NaN                    0.635                    0.592
  GPT 5.1 med -> 5.2 med           0.145             0.145                  -0.048                   0.353                    0.719                    0.553
GPT 5.1 high -> 5.2 high           0.182             0.124                  -0.091                   0.345                    0.667                    0.565
   GPT 5.2 high -> xhigh           0.200             0.052                  -0.093                   0.206                    0.626                    0.528
     Opus 4.5 16k -> 64k           0.127            -0.009                  -0.147                   0.127                    0.678                    0.523
```
