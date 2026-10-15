# Frozen development selection rules

1. For each original outer fold, reproduce the nine V2 E1 candidates on the exact
   original eight-speaker training/two-speaker validation split (9 fits).
2. Within that fold's ten development speakers, sort each class by SHA256 of
   [v3-cv,42,outer_fold,label,speaker], assign round-robin to three validation
   buckets, and use the complement for training. Both classes appear in every
   role; all ten speakers have exactly one validation appearance. Audit speaker,
   logical ID, primary path and audio hash intersections. This replaces reliance
   on a single two-speaker split for V3 selection.
3. Evaluate the same nine B0/N0/unaugmented references in those folds (27 fits).
   Choose their anchor by mean development macro-F1, balanced accuracy, ROC-AUC,
   lower log loss, fewer changed factors, then lexical stable candidate ID. C and
   layer remain fixed at that development-selected anchor for modifications.
4. Run four one-factor ablations: B2/N0, B0/N1, B0/N2, B0/N0 with audio augmentation
   (12 fits). B1 is identical to B0 and gets no duplicate training.
5. At most one combined candidate is admitted. Each included factor must improve
   mean macro-F1 by >=0.01 absolute over reference, improve macro-F1 in at least
   two of three splits, and not lower mean balanced accuracy. If both N1/N2 qualify,
   use the same tie rule to choose one. Require at least two distinct qualifying
   factors. Evaluate that one combined candidate in the same three splits (<=3 fits).
6. Choose among anchor, four ablations and optional combination by the same tie
   rule. Refit reference and selected LR on all ten development speakers; if the
   selected recipe is the reference, reuse the same fit. No outer features fit
   normalizers, weights, models, calibration, thresholds or selection.
7. Freeze all three fold selections and all final checkpoint hashes together.
   Source/config/dependency/lineage identities must match on every resume and
   immediately before any explicitly authorized outer inference.

Maximum main budget: 51 candidate fits + 2 final fits per outer fold, 159 LR fits
across the study. This is a staged search, not 3x3x3x3x2 exhaustive combinations.
There are only 14 unique recipes at most per outer fold (9 references + 4 ablations
+ 1 conditional combination). Selection optimism remains possible with 15 speakers.
No MLP is implemented without development-only justification; no HuBERT fine-tuning.

The optional calibration flag is frozen in config (default false). Its diagnostic
uses a fixed layer-9/C=0.1/B0/N0 recipe independent of model-selection results.
Per-class SHA256 ranking assigns one development speaker to calibration and another
to assessment; the other six train the base classifier. Fit regularized sigmoid
LR (C=1) on calibration logits with equal speaker weights and assess on the two
remaining speakers. Fit/assessment roles are disjoint, unaugmented, and contain
both classes. Three extra base LR and three small calibration fits at most.
It neither changes the primary selected model nor produces calibrated outer scores.
These assessment speakers may participate elsewhere in the main development study;
the recipe here cannot consume those results. Calibration is exploratory and separate.

Threshold is always 0.5. No V2 outer scores or error rankings are accepted by the
selection API. Reports may compare historical outer metrics only after decisions
are frozen and current outer results are sealed. Do not use comparison tables
to rerun selection or discard unfavorable results.
