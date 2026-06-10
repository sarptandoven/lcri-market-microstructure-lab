# Human evaluation annotation UI prototype

Static vanilla HTML prototype for Preview Ladder DiT human evaluation.

Run locally:

```bash
cd prototypes/human_eval_annotation_ui
python3 -m http.server 8765
# open http://127.0.0.1:8765
```

Use cases covered:

1. Preview trust survey: Likert ratings for structural commitment, approve-preview likelihood, and expected drift risk.
2. Pairwise final preference: side-by-side final render comparison with rationale.
3. Time-to-acceptable-output interaction study: event buttons log first preview, prompt/mask changes, preview acceptance, final acceptance, and abandonment.
4. Quality controls: instruction check, self-reported video playback checks, total elapsed time, user agent capture.

Expected input: task JSON matching `sample_task.json` with source, mask overlay, preview, final_a, final_b media URIs.

Expected output: annotation JSON with schema version `preview-ladder-human-eval/v0.1`, suitable for later validation by a target-repo `human_eval.py` module.
