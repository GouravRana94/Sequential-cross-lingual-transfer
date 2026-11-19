# Sequential cross-lingual transfer for few-shot NER in Indian languages

| File | Purpose |
|---|---|
| `nlp22.ipynb` | Original experiment notebook (v4.5, 924 reported runs) with logged outputs |
| `original_runs_parsed.csv` | Per-run F1/P/R of the original runs, parsed from the notebook log |
| `stats_revised.py` | Language-level significance analysis of the original runs -> `revised_significance.csv`, `model_mean_ci.csv` |
| `revision_experiments.py` | Revision runs: common evaluation (WikiAnn + Naamapadam-500) and 2x2x2 ablation (init x LALF x weighted loss); one GPU per process |
| `launch_kaggle.py` | Runs two processes (one per GPU) with an 11 h time limit |
| `merge_results.py`, `analyze_revision.py` | Merge the two outputs and produce the tables |
| `printed_*.csv` | Aggregate tables of the revision run (factorial, main effects, regimes), copied from the analysis output |
| `exploratory_2gpu_partial_rows.csv` | **Do not use for the paper**: an earlier partial run where two visible GPUs halved the number of updates |
| `regenerate_manifest.py` | Rebuilds `split_manifest.jsonl` (no training needed) |

## Status of the revision data
The final revision run (1,188 result rows: 1,056 fine-tuning runs with 108 optimizer steps and 132 zero-shot models with 500 steps,
identical to the original configuration) was analysed and its tables are in `printed_*.csv`. The per-run CSV of that run was lost when the
Kaggle session output expired; re-running `launch_kaggle.py` regenerates it (the runs are deterministic on the same hardware) and
`regenerate_manifest.py` regenerates the split manifest.

## Reproduce
```
pip install -r requirements.txt
python launch_kaggle.py            # two GPUs; or: python revision_experiments.py --preset core --gpu 0 --models mBERT XLM-R --out run_A
python merge_results.py run_A run_B merged
python analyze_revision.py merged/revision_results_raw.csv merged
```
Sampling: train = `shuffle(seed)` then first k; validation = `shuffle(seed)`, first 200; Naamapadam test = `shuffle(seed)`, first 500. Seeds 42, 123, 2024.

## Keep the outputs safe (Kaggle)
Use *Save Version -> Save & Run All* and download from the version's Output tab. As insurance, a last cell can print the results as text inside the notebook:
```
import gzip, base64
for f in ["merged/revision_results_raw.csv", "merged/split_manifest.jsonl"]:
    print("BEGIN", f); print(base64.b64encode(gzip.compress(open("/kaggle/working/"+f,"rb").read())).decode()); print("END", f)
```
