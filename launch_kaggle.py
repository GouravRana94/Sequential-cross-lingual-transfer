# Paste into ONE Kaggle cell (Accelerator: GPU T4 x2). Runs two processes, one per GPU, in parallel.
# Each process is stopped after 11 h (timeout) so the cells after this one still run before Kaggle's 12 h limit.
# Progress is saved after every run, so you can resume later (see README / chat instructions).
import subprocess, os
assert os.path.exists("revision_experiments.py"), "Run the %%writefile cell for revision_experiments.py first"
LIMIT = 39600  # seconds = 11 h
cmds = [
 f"timeout {LIMIT} python revision_experiments.py --preset core --gpu 0 --models mBERT XLM-R   --out /kaggle/working/run_A --no-full-test",
 f"timeout {LIMIT} python revision_experiments.py --preset core --gpu 1 --models MuRIL IndicBERT --out /kaggle/working/run_B --no-full-test",
]
procs = [subprocess.Popen(c + f" > /kaggle/working/log_{i}.txt 2>&1", shell=True) for i, c in enumerate(cmds)]
for p in procs: p.wait()
for i, p in enumerate(procs):
    if p.returncode == 0:
        print(f"process {i}: finished all runs")
    elif p.returncode == 124:
        print(f"process {i}: stopped at the 11 h limit - progress is saved, resume in a new session")
    else:
        print(f"process {i} CRASHED, exit code {p.returncode}; last log lines:")
        print("".join(open(f"/kaggle/working/log_{i}.txt").readlines()[-40:]))
