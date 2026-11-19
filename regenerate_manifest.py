"""Regenerates split_manifest.jsonl (exact sampled sentence indices) WITHOUT any training. Needs internet (Hugging Face), no GPU.
Run in Kaggle/Colab next to revision_experiments.py:   python regenerate_manifest.py
Output: /kaggle/working/manifest_out/split_manifest.jsonl  (also gives n_train_full / n_val_full / n_test_full for Table 1)."""
import sys, os
sys.argv = [sys.argv[0], "--out", "/kaggle/working/manifest_out"]
os.makedirs("/kaggle/working/manifest_out", exist_ok=True)
import revision_experiments as R
for lang in R.CFG["languages"]:
    for k in (10, 50, 100):
        for seed in R.CFG["seeds"]:
            R.naam_splits(lang, k, seed)
print("written:", R.MANIFEST)
