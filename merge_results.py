"""Merge the per-GPU outputs:  python merge_results.py run_A run_B merged/"""
import sys, os, pandas as pd, shutil
*runs, out = sys.argv[1:]; os.makedirs(out, exist_ok=True)
df = pd.concat([pd.read_csv(f"{r}/revision_results_raw.csv") for r in runs], ignore_index=True).drop_duplicates("key", keep="last")
df.to_csv(f"{out}/revision_results_raw.csv", index=False)
with open(f"{out}/split_manifest.jsonl", "w") as w:
    for r in runs: w.write(open(f"{r}/split_manifest.jsonl").read())
shutil.copy(f"{runs[0]}/environment.json", f"{out}/environment_A.json"); shutil.copy(f"{runs[-1]}/environment.json", f"{out}/environment_B.json")
print(len(df), "rows;", df.steps.value_counts().to_dict() if "steps" in df else "")
