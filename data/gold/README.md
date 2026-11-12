# Gold annotations

No gold annotations are bundled with this repository. A template can be
generated from an existing audit, but the reviewer columns are blank and must be
filled by people.

Generate a template:

```powershell
python eval/make_gold_template.py outputs/sjs.phase2.json `
  --out data/gold/annotation_template.csv
```

Read [`ANNOTATION_GUIDELINES.md`](ANNOTATION_GUIDELINES.md) before annotating.
Score a filled copy with:

```powershell
python eval/score_benchmark.py data/gold/annotations.csv `
  --out eval/benchmark.json
```

The scoring report writes `status: "no_annotations"` and null metrics when the
reviewer columns are still blank. Invalid values yield
`status: "invalid_annotations"`; reviewer values without a model counterpart
yield `status: "no_pairs"`. Do not report benchmark numbers until the reviewer
columns have been filled and, for a benchmark claim, double-annotated with
agreement statistics recorded.
