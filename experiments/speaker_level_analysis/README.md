# POST-HOC EXPLORATORY SPEAKER-LEVEL ANALYSIS

B2 and C2 both achieve **73.33% (11/15 speakers correct)**. Both make four errors; neither reaches 90%. These exploratory results do not replace the signed utterance-level results.

Start with [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md).

| File | Contents |
|---|---|
| speaker_metrics.json | All pooled/fold metrics, intervals, original comparisons and provenance |
| speaker_predictions.csv | Exactly 15 rows per model, full precision speaker means and decisions |
| record_predictions.csv | All 17,346 model-record rows; original IDs, labels, probabilities and derived decisions |
| fold_speaker_metrics.csv | Five-speaker metrics for each of six model-fold cells |
| confusion_matrix.png | Both complete speaker confusion matrices |
| comparison_chart.png | Original utterance fold means vs pooled speaker accuracy, explicitly distinct intervals |
| source_provenance.json | Six original native result paths and checksums, verified source identities |
| PRESERVATION_REPORT.md | Verification scope, counts and limitations |
| preservation_before.json / preservation_after.json | Original inventory/hashes and Git states |
| preservation_summary.json | Machine-readable preservation verdict |
| output_manifest.json | SHA-256 and byte size of published analysis files |
| analyze_speakers.py | Standalone reproducible analysis; imports no project/model code |

Class mapping: 0 = control, 1 = dysarthric. Matrix order: [[TN,FP],[FN,TP]]. Aggregation: all saved evaluation probabilities per speaker, arithmetic mean, >=0.5. All 8,673 frozen CORE records are retained per model.

To verify the retained package from this experimental worktree, using the existing global Python:

```powershell
python -B .\experiments\speaker_level_analysis\analyze_speakers.py --verify
```

For reproducibility, default execution generates a new package only when outputs do not exist; it refuses overwrites. It expects the unchanged original checkout and its frozen artifacts beside this experimental worktree. Exact library versions are in `speaker_metrics.json`.

No project code, checkpoint, original prediction, sign-off or release branch changed. No training, model forwards, commit or push. **STOP: await user review.**
