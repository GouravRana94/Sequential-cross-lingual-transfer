# ==============================================================================
# Revision experiments for "Sequential Cross-Lingual Transfer for Few-Shot NER ..."
#   (1) COMMON EVALUATION: every regime (zero-shot / direct few-shot / sequential) is
#       scored on the SAME test sets: WikiAnn test AND Naamapadam test (seed-specific
#       500-example subsample, identical to the original runs, AND the full test split).
#   (2) ABLATION: full 2x2x2 factorial
#         init  in {base pre-trained, zero-shot checkpoint}
#         LALF  in {0 frozen layers, 4 frozen layers}
#         WCE   in {plain CE, inverse-frequency class-weighted CE}
#       The original "direct" run = (base, LALF4, WCE on); "sequential" = (zs, LALF4, WCE on).
#   (3) REPRODUCIBILITY: writes environment.json (GPU, library versions, dataset commit
#       SHAs) and split_manifest.jsonl (exact example indices for every train/val/test sample).
#
# Same training code/hyper-parameters as the original notebook v4.5 (transformers 5.0.0,
# datasets 5.0.0). Resumable: results are appended to CSV after every run.
# Usage (Kaggle cell or shell):   python revision_experiments.py --preset core
#   --preset core : k=50, all 4 models, 11 langs, 3 seeds, 8 cells  (= 1,056 runs + 12 zero-shot)
#   --preset full : k in {10,50,100}                                  (= 3,168 runs + 12 zero-shot)
#   --models MuRIL mBERT   restrict models (run the two extremes first if GPU time is short)
# ==============================================================================
import os, sys
# --- IMPORTANT: use exactly ONE GPU per process. With two visible GPUs the HF Trainer silently wraps the model in
# DataParallel, doubling the effective batch size and halving the number of optimizer steps (60 instead of 108 at k=50),
# which does NOT reproduce the paper's training configuration. The original notebook pinned CUDA_VISIBLE_DEVICES=0.
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json, time, shutil, random, gc, argparse, itertools, platform
import numpy as np, pandas as pd, torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, Features, ClassLabel, Sequence, Value, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments,
                          Trainer, DataCollatorForTokenClassification, set_seed)
import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="core", choices=["core", "full"])
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--langs", nargs="*", default=None)
ap.add_argument("--out", default=os.environ.get("OUT_DIR", "./revision_results"))
ap.add_argument("--gpu", default="0", help="physical GPU index for this process (one GPU per process)")
ap.add_argument("--no-full-test", action="store_true", help="skip evaluation on full Naamapadam test")
args, _ = ap.parse_known_args()

CFG = dict(
    languages=["hi", "bn", "mr", "gu", "pa", "or", "as", "ta", "te", "kn", "ml"],
    models={"mBERT": "bert-base-multilingual-cased", "XLM-R": "xlm-roberta-base",
            "MuRIL": "google/muril-base-cased", "IndicBERT": "ai4bharat/IndicBERTv2-MLM-only"},
    seeds=[42, 123, 2024],
    k_values=[50] if args.preset == "core" else [10, 50, 100],
    zs_train_cap=8000, zs_epochs=2, zs_batch=32, zs_lr=3e-5,
    fs_epochs_by_k={10: 20, 50: 12, 100: 8}, fs_batch_by_k={10: 4, 50: 6, 100: 8},
    fs_lr_by_model={"mBERT": 2e-5, "XLM-R": 2e-5, "MuRIL": 1e-5, "IndicBERT": 1.5e-5},
    max_len=96, weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0,
    collapse_f1_threshold=0.01, max_collapse_retries=1, retry_lr_factor=0.5,
    val_cap=200, test_cap=500, lalf_layers=4,
)
if args.models: CFG["models"] = {m: CFG["models"][m] for m in args.models}
if args.langs: CFG["languages"] = args.langs
FULL_TEST = not args.no_full_test

OUT = args.out; CKPT = os.path.join(OUT, "_ckpts")
os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)
RES_CSV = os.path.join(OUT, "revision_results_raw.csv")
MANIFEST = os.path.join(OUT, "split_manifest.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.device_count() <= 1, "more than one GPU visible: Trainer would use DataParallel (see note at top)"
import datasets as _ds, transformers as _tf
try:
    _ds.disable_progress_bars(); _tf.logging.set_verbosity_error()
except Exception:
    pass
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
L2I = {l: i for i, l in enumerate(LABELS)}; I2L = {i: l for l, i in L2I.items()}; NL = len(LABELS)
metric = evaluate.load("seqeval")

# ----------------------------- environment record -----------------------------
def write_environment():
    import transformers, datasets, evaluate as ev, accelerate, seqeval
    env = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
               cuda_runtime=torch.version.cuda, transformers=transformers.__version__,
               datasets=datasets.__version__, evaluate=ev.__version__, accelerate=accelerate.__version__,
               numpy=np.__version__, pandas=pd.__version__, fp16=bool(torch.cuda.is_available()),
               gpus=[dict(name=torch.cuda.get_device_name(i),
                          total_memory_gb=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
                          capability=list(torch.cuda.get_device_capability(i)))
                     for i in range(torch.cuda.device_count())],
               config={k: v for k, v in CFG.items()}, full_test_eval=FULL_TEST)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        env["dataset_commits"] = {
            "ai4bharat/naamapadam@refs/convert/parquet": api.dataset_info("ai4bharat/naamapadam", revision="refs/convert/parquet").sha,
            "unimelb-nlp/wikiann@main": api.dataset_info("unimelb-nlp/wikiann").sha}
        env["model_commits"] = {n: api.model_info(c).sha for n, c in CFG["models"].items()}
    except Exception as e:
        env["commit_lookup_error"] = repr(e)
    json.dump(env, open(os.path.join(OUT, "environment.json"), "w"), indent=2, default=str)
    print(json.dumps({k: env[k] for k in ("torch", "transformers", "datasets", "gpus")}, indent=1))

# ----------------------------- data -----------------------------
def retry_load(*a, tries=5, **kw):
    for t in range(tries):
        try: return load_dataset(*a, **(dict(kw, download_mode="force_redownload") if t >= 1 else kw))
        except Exception as e:
            print("   hub error:", repr(e)[:150]); time.sleep(8 * 2 ** t)
    raise RuntimeError("dataset load failed")

def canon(ds):
    names = list(ds.features["ner_tags"].feature.names)
    if names == LABELS: return ds
    feats = Features({"tokens": Sequence(Value("string")), "ner_tags": Sequence(ClassLabel(names=LABELS))})
    return ds.map(lambda ex: {"tokens": ex["tokens"], "ner_tags": [L2I.get(names[i], 0) for i in ex["ner_tags"]]}, features=feats)

_naam, _wiki = {}, {}
def naam(lang):
    if lang not in _naam:
        dd = retry_load("ai4bharat/naamapadam", data_dir=lang, revision="refs/convert/parquet", trust_remote_code=False)
        if "validation" not in dd:  # identical to the original notebook
            sp = dd["train"].train_test_split(test_size=0.1, seed=42); dd["train"], dd["validation"] = sp["train"], sp["test"]
        for s in dd: dd[s] = canon(dd[s]).add_column("_idx", list(range(len(dd[s]))))
        _naam[lang] = dd
    return _naam[lang]

def wiki(lang):
    if lang not in _wiki:
        dd = retry_load("unimelb-nlp/wikiann", lang)
        for s in dd: dd[s] = canon(dd[s]).add_column("_idx", list(range(len(dd[s]))))
        _wiki[lang] = dd
    return _wiki[lang]

def log_manifest(**rec):
    with open(MANIFEST, "a") as f: f.write(json.dumps(rec) + "\n")

_logged = set()
def naam_splits(lang, k, seed):
    """Same sampling as the original notebook: shuffle(seed) then take first k / 200 / 500."""
    dd = naam(lang)
    tr = dd["train"].shuffle(seed=seed).select(range(min(k, len(dd["train"]))))
    va = dd["validation"].shuffle(seed=seed); va = va.select(range(min(CFG["val_cap"], len(va))))
    te = dd["test"].shuffle(seed=seed);       te_cap = te.select(range(min(CFG["test_cap"], len(te))))
    key = (lang, k, seed)
    if key not in _logged:
        _logged.add(key)
        log_manifest(dataset="naamapadam", lang=lang, k=k, seed=seed, n_train_full=len(dd["train"]),
                     n_val_full=len(dd["validation"]), n_test_full=len(dd["test"]),
                     train_idx=[int(i) for i in tr["_idx"]], val_idx=[int(i) for i in va["_idx"]],
                     test_cap_idx=[int(i) for i in te_cap["_idx"]])
    strip = lambda d: d.remove_columns("_idx")
    return strip(tr), strip(va), strip(te_cap), strip(dd["test"])

# ----------------------------- training utils (as in notebook v4.5) -----------------------------
def tok_fn_factory(tk):
    def f(ex):
        t = tk(ex["tokens"], truncation=True, is_split_into_words=True, max_length=CFG["max_len"], padding=False)
        out = []
        for i, labs in enumerate(ex["ner_tags"]):
            prev, seq = None, []
            for w in t.word_ids(batch_index=i):
                seq.append(-100 if (w is None or w == prev) else labs[w]); prev = w
            out.append(seq)
        t["labels"] = out; return t
    return f

def metrics(p):
    preds, labels = p; preds = np.argmax(preds, axis=2); tp, tl = [], []
    for a, b in zip(preds, labels):
        tp.append([I2L[x] for x, y in zip(a, b) if y != -100]); tl.append([I2L[y] for y in b if y != -100])
    r = metric.compute(predictions=tp, references=tl, zero_division=0)
    o = dict(precision=r["overall_precision"], recall=r["overall_recall"], f1=r["overall_f1"])
    for e in ("PER", "ORG", "LOC"): o[f"f1_{e}"] = r[e]["f1"] if e in r else np.nan
    return o

def class_weights(ds):
    c = np.ones(NL)
    for ex in ds:
        for l in ex["ner_tags"]: c[l] += 1
    inv = 1.0 / c; return torch.tensor(inv / inv.sum() * NL, dtype=torch.float)

class WTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **kw):
        super().__init__(*a, **kw); self.cw = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels"); out = model(**inputs)
        w = self.cw.to(out.logits.device) if self.cw is not None else None
        loss = CrossEntropyLoss(weight=w, ignore_index=-100)(out.logits.view(-1, NL), labels.view(-1))
        return (loss, out) if return_outputs else loss

def freeze_bottom(model, n):
    if n <= 0: return 0
    base = getattr(model, "bert", None) or getattr(model, "roberta", None)
    assert base is not None, "unknown architecture: LALF could not be applied"
    for layer in base.encoder.layer[:n]:
        for p in layer.parameters(): p.requires_grad = False   # embeddings stay trainable (as in the paper)
    return n

def fit(ckpt, init, train_ds, run, epochs, batch, seed, lr, wce, lalf):
    set_seed(seed); tk = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(init or ckpt, num_labels=NL, id2label=I2L, label2id=L2I).to(DEVICE)
    nfrozen = freeze_bottom(model, lalf)
    ttr = train_ds.map(tok_fn_factory(tk), batched=True, remove_columns=train_ds.column_names)
    spe = max(1, len(ttr) // max(1, batch))
    ta = TrainingArguments(output_dir=os.path.join(CKPT, run), num_train_epochs=epochs, per_device_train_batch_size=batch,
        per_device_eval_batch_size=32, learning_rate=lr, weight_decay=CFG["weight_decay"],
        warmup_steps=max(3, int(CFG["warmup_ratio"] * spe * epochs)), max_grad_norm=CFG["max_grad_norm"],
        logging_steps=10**6, eval_strategy="no", save_strategy="no", fp16=torch.cuda.is_available(),
        report_to="none", seed=seed, data_seed=seed, disable_tqdm=True)
    kw = dict(model=model, args=ta, train_dataset=ttr, data_collator=DataCollatorForTokenClassification(tk))
    if wce: kw["class_weights"] = class_weights(train_ds)
    tr = (WTrainer if wce else Trainer)(**kw); tr.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, tk, nfrozen, n_trainable

def score(tr, tk, ds):
    t = ds.map(tok_fn_factory(tk), batched=True, remove_columns=ds.column_names)
    p = tr.predict(t); return metrics((p.predictions, p.label_ids))

def fit_guarded(ckpt, init, train_ds, val_ds, run, epochs, batch, seed, lr, wce, lalf):
    att, cur = 0, lr
    while True:
        tr, tk, nf, ntr = fit(ckpt, init, train_ds, run, epochs, batch, seed, cur, wce, lalf)
        v = score(tr, tk, val_ds)["f1"]
        if v < CFG["collapse_f1_threshold"] and att < CFG["max_collapse_retries"]:
            att += 1; cur *= CFG["retry_lr_factor"]; del tr; gc.collect(); torch.cuda.empty_cache(); continue
        return tr, tk, att, v, cur, ntr

# ----------------------------- bookkeeping -----------------------------
done = set(); rows = []
if os.path.exists(RES_CSV):
    prev = pd.read_csv(RES_CSV); rows = prev.to_dict("records")
    done = set(prev["key"])
def save(r):
    rows.append(r); pd.DataFrame(rows).to_csv(RES_CSV, index=False); done.add(r["key"])

def eval_all(tr, tk, lang, seed, k, base_meta):
    """Score one model on every test set used in the paper, for one target language."""
    out = {}
    w = score(tr, tk, wiki(lang)["test"].remove_columns("_idx"))
    out.update({f"wikiann_{a}": b for a, b in w.items()})
    _, _, te_cap, te_full = naam_splits(lang, k or 50, seed)
    n = score(tr, tk, te_cap); out.update({f"naam500_{a}": b for a, b in n.items()})
    if FULL_TEST:
        f = score(tr, tk, te_full); out.update({f"naamfull_{a}": b for a, b in f.items()})
    out.update(n_wikiann_test=len(wiki(lang)["test"]), n_naam500=len(te_cap), n_naamfull=len(te_full))
    return {**base_meta, **out}

# ----------------------------- main -----------------------------
def main():
    write_environment(); t0 = time.time()
    # ---- Phase A: zero-shot checkpoints, evaluated on BOTH datasets ----
    src = retry_load("unimelb-nlp/wikiann", "en")["train"]
    src = canon(src).shuffle(seed=42).select(range(CFG["zs_train_cap"]))   # same 8,000 EN examples as the paper
    log_manifest(dataset="wikiann-en", note="shuffle(seed=42), first 8000 of train")
    zs_path = {}
    for mname, ckpt in CFG["models"].items():
        for seed in CFG["seeds"]:
            run = f"zs_{mname}_s{seed}"; path = os.path.join(CKPT, run); zs_path[(mname, seed)] = path
            if all(f"zs|{mname}|{l}|{seed}" in done for l in CFG["languages"]) and os.path.exists(path): continue
            print(f"--- {run} ({(time.time()-t0)/3600:.2f}h)")
            tr, tk, _, ntr = fit(ckpt, None, src, run, CFG["zs_epochs"], CFG["zs_batch"], seed, CFG["zs_lr"], wce=False, lalf=0)
            tr.save_model(path); tk.save_pretrained(path)
            for lang in CFG["languages"]:
                if f"zs|{mname}|{lang}|{seed}" in done: continue   # checkpoint was lost between sessions; row already saved
                m = eval_all(tr, tk, lang, seed, 50, dict(key=f"zs|{mname}|{lang}|{seed}", regime="zero_shot", model=mname,
                              lang=lang, k=0, seed=seed, init="zs", lalf=0, wce=0, retries=0, trainable_params=ntr,
                              steps=tr.state.global_step))
                save(m)
            del tr; gc.collect(); torch.cuda.empty_cache()
    # NOTE: zero-shot Naamapadam subsample uses naam_splits(lang, 50, seed): the sampled TEST set depends only on
    # (lang, seed), not on k, so it is identical to the subsample used for the few-shot/sequential runs.

    # ---- Phase B: 2x2x2 factorial over init x LALF x WCE ----
    cells = list(itertools.product(["base", "zs"], [0, CFG["lalf_layers"]], [0, 1]))
    for mname, ckpt in CFG["models"].items():
        for lang in CFG["languages"]:
            for k in CFG["k_values"]:
                for seed in CFG["seeds"]:
                    tr_ds, va_ds, te_cap, _ = naam_splits(lang, k, seed)
                    for init, lalf, wce in cells:
                        key = f"fs|{mname}|{lang}|{k}|{seed}|{init}|{lalf}|{wce}"
                        if key in done: continue
                        run = f"ab_{mname}_{lang}_k{k}_s{seed}_{init}_L{lalf}_W{wce}"
                        try:
                            tr, tk, att, vf1, lr_used, ntr = fit_guarded(
                                ckpt, zs_path[(mname, seed)] if init == "zs" else None, tr_ds, va_ds, run,
                                CFG["fs_epochs_by_k"][k], min(CFG["fs_batch_by_k"][k], k), seed,
                                CFG["fs_lr_by_model"][mname], bool(wce), lalf)
                            m = eval_all(tr, tk, lang, seed, k, dict(key=key, regime="finetune", model=mname, lang=lang, k=k,
                                         seed=seed, init=init, lalf=lalf, wce=wce, retries=att, val_f1=vf1, lr_used=lr_used,
                                         trainable_params=ntr, steps=tr.state.global_step))
                            save(m)
                            print(f"{run}: naam500 F1={m['naam500_f1']:.3f} wikiann F1={m['wikiann_f1']:.3f} ({(time.time()-t0)/3600:.2f}h)")
                        except Exception as e:
                            print("!! FAILED", run, repr(e)[:200])
                        finally:
                            for v in ("tr",):
                                if v in locals(): del tr
                            shutil.rmtree(os.path.join(CKPT, run), ignore_errors=True); gc.collect(); torch.cuda.empty_cache()
    print("done", len(rows), "result rows")

if __name__ == "__main__":
    main()
