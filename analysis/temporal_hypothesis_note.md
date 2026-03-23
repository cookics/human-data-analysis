# Temporal Hypothesis Note

Quick answer: yes, there is a little signal here.

- Across the local model panel, better ARC Public Eval pair accuracy tends to come with better human item-alignment (`r = 0.476` across models).
- `gpt-5.2` is generally more human-aligned than `gpt-5.1` at comparable thinking levels. Medium goes from 0.160 to 0.306; high goes from 0.101 to 0.224.
- The qualitative pattern still looks mostly like `more of the same but better`: newly solved GPT items are easier for humans than the items the new model still fails (0.719 vs 0.553 for the medium pair; 0.667 vs 0.565 for the high pair).
- We do not have `Opus 4` locally, so there is no direct `Opus 4 -> 4.5` time test. What we can say is that `Opus 4.5` already looks quite human-aligned by `16k` and stays that way at `64k` (0.439 -> 0.430) while accuracy rises a lot.
- One interesting exception: `Opus 4.5` can look more human-aligned than much higher-scoring GPT settings, so better raw score and more human-like item ordering are related but not identical.

I kept the fuller writeup and figure in `Human data/Creme Analysis/temporal_hypotheses.md` and `fig02_family_progressions.png` because that frame is closer to the original Cremieux-style question.
