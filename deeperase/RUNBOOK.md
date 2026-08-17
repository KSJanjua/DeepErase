# RUNBOOK — Running DeepErase on your GPU server, from zero

**Written for:** someone who has never run this project before.
**Your server:** `root@dgxhnode1`, working folder `/workspace`, one GPU with 20 GB.
**Total time:** about 1 hour, most of it waiting for downloads.

Every command below can be copied and pasted exactly as written. After each one
I say what you should see, and what to do if you see something else.

---

## Contents

| Part | What it covers | Time |
|---|---|---|
| 0 | Before you start | 2 min |
| 1 | Check the GPU is really there | 2 min |
| 2 | Get the project onto the server | 10 min |
| 3 | Install the software it needs | 10 min |
| 4 | Run the built-in self-tests | 2 min |
| 5 | Preflight check | 2 min |
| 6 | **The real experiment** | 30–45 min |
| 7 | Read the results | 5 min |
| 8 | Get the results back to your laptop | 5 min |
| 9 | Troubleshooting | as needed |

---

# PART 0 — Before you start

### What we are about to do

We will check whether our depth-measurement code produces the same numbers as
the researchers who invented it. They published four expected values. If our
code reproduces them, our measurement is probably correct. If it does not, we
have found a bug — which is also useful.

**Nothing here trains a model.** We download four ready-made models and measure
them. That is why it only takes an hour.

### One habit that will save you pain

Your connection to the server can drop. If that happens in the middle of a
command, the command dies with it. So we will run the long step inside
`tmux`, which keeps things running even if you disconnect. Part 6 covers this.

### Log in

```bash
ssh root@dgxhnode1
```

You should land in a prompt like:

```
root@dgxhnode1:/workspace#
```

If you land somewhere else, move to the workspace folder:

```bash
cd /workspace
```

---

# PART 1 — Check the GPU is really there

### 1.1 Ask the machine about its graphics card

```bash
nvidia-smi
```

**What you should see:** a table showing a GPU name and memory, something like
`20480MiB` or `22GB`.

Look at the memory column. If another process is already using most of it, stop
that process first, or the experiment will run out of memory.

**If you see `command not found`:** there is no GPU driver on this machine.
Stop here — nothing else will work. Ask whoever gave you the server.

### 1.2 Ask Python about the graphics card

This matters more than the previous step. The GPU can exist while Python
cannot see it.

```bash
python -c "import torch; print('torch', torch.__version__); print('CUDA works:', torch.cuda.is_available())"
```

**What you should see:**

```
torch 2.5.1+cu121
CUDA works: True
```

The version number does not have to match mine. Anything from `2.1` upward is
fine.

**Write down the version you see.** You will need it in Part 3.

**If it says `CUDA works: False`** — Python has a CPU-only torch installed.
See Troubleshooting **T1**.

**If it says `No module named 'torch'`** — see Troubleshooting **T2**.

---

# PART 2 — Get the project onto the server

Pick **one** of the three options below. Option A is easiest if your code is on
GitHub.

### Option A — from GitHub

```bash
cd /workspace
git clone <your-repository-url> deeperase
cd deeperase
```

### Option B — upload a zip from your laptop

On **your laptop** (not the server), zip the project folder. Then:

```bash
scp Capstone.zip root@dgxhnode1:/workspace/
```

Back on the **server**:

```bash
cd /workspace
unzip Capstone.zip -d deeperase
cd deeperase
```

### Option C — copy the folder directly

On **your laptop**:

```bash
scp -r "C:/Users/janju/OneDrive/Desktop/Capstone" root@dgxhnode1:/workspace/deeperase
```

Then on the **server**:

```bash
cd /workspace/deeperase
```

### 2.1 Check the files arrived

```bash
ls
```

**What you should see** — these names must all be present:

```
deeperase   tests   requirements-gpu.txt   pyproject.toml   README.md
```

**If `deeperase` is missing**, you copied the wrong level of folder. You want
the folder that *contains* a folder called `deeperase`, not the inner one.

### 2.2 Check you are in the right place

```bash
pwd
```

**You should see:** `/workspace/deeperase`

Every command from here on assumes you are in this folder. If you get lost:

```bash
cd /workspace/deeperase
```

---

# PART 3 — Install the software it needs

### 3.1 Install everything except torch

**Do not install torch.** Your server already has a working one, and replacing
it usually breaks it.

```bash
pip install -r requirements-gpu.txt
```

**What you should see:** a lot of scrolling, ending with `Successfully installed ...`

This takes 5–10 minutes.

**If you see red `ERROR` lines**, see Troubleshooting **T3**.

### 3.2 Confirm torch still works

Installing other packages can occasionally disturb torch. Always check.

```bash
python -c "import torch; print('CUDA works:', torch.cuda.is_available())"
```

**What you should see:** `CUDA works: True`

**If this now says `False` but said `True` in Part 1**, something overwrote
torch. See Troubleshooting **T4**.

### 3.3 Confirm the other pieces loaded

```bash
python -c "import transformers, datasets, sklearn, scipy, matplotlib; print('all imports OK')"
```

**What you should see:** `all imports OK`

---

# PART 4 — Run the built-in self-tests

Before touching a real model, check the code itself is healthy. This runs
entirely on the processor and needs no GPU.

```bash
python -m pytest tests/ -q
```

**What you should see** (takes about 30 seconds):

```
342 passed, 1 skipped
```

The 1 skipped test is a GPU-only check that is skipped on some setups. That is
normal.

**If any test fails, stop.** Do not continue to Part 5. Copy the failure
message and send it to me. Running an experiment on broken code wastes an hour
and produces numbers you cannot trust.

---

# PART 5 — Preflight check

This checks everything the experiment needs, without downloading any models.
It takes seconds and catches problems early.

```bash
python -m deeperase.scripts.check_gpu --size 1B
```

**What you should see:**

```
========================================================================
DeepErase preflight check
========================================================================

1. PyTorch and GPU
------------------------------------------------------------------------
  torch version      : 2.5.1+cu121
  CUDA build         : 12.1
  [ OK ] GPU detected     : NVIDIA A100-SXM4-40GB
  total memory       : 21.47 GB
  bfloat16 supported : yes
  currently free     : 21.20 GB

2. Memory plan
------------------------------------------------------------------------
  [ OK ] 1B     7.41 GB peak  all_resident    <-- selected
  [ OK ] 3B     6.43 GB peak  sequential

3. Model repositories reachable
------------------------------------------------------------------------
  [ OK ] full      open-unlearning/tofu_Llama-3.2-1B-Instruct_full
  [ OK ] retain90  open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90
  [ OK ] retain95  open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95
  [ OK ] retain99  open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99

4. Disk space
------------------------------------------------------------------------
  needed for 1B      : 9.89 GB (all four splits)
  free               : 500.00 GB
  [ OK ] Sufficient space

Summary
------------------------------------------------------------------------
  [ OK ] READY. This machine can run the validation experiment.
```

**You need `[ OK ] READY` at the bottom.** If any section shows `[FAIL]`, fix
that before continuing — see Troubleshooting.

> **Why `--size 1B` and not 3B?** The researchers published their expected
> numbers for the 1B model, so that is the only size we can check ourselves
> against. It is also four times smaller to download. Get 1B working first.

---

# PART 6 — The real experiment

### 6.1 Start a tmux session first

This is the important habit. `tmux` keeps your command running even if your
connection drops.

```bash
tmux new -s uds
```

Your screen clears and you get a fresh prompt with a green bar at the bottom.
**You are now inside tmux.** Everything you run here survives a disconnect.

Move back to the project folder (tmux starts in your home folder):

```bash
cd /workspace/deeperase
```

### 6.2 Start the experiment

```bash
python -m deeperase.scripts.run_uds_validation --size 1B --n-examples 50 2>&1 | tee run_log.txt
```

The `| tee run_log.txt` part saves everything you see into a file, so you have
a record even if the screen scrolls away.

### 6.3 What happens, in order

**Stage 1 of 4 — downloading (10–20 minutes the first time).**
You will see progress bars. This downloads about 10 GB. It only happens once;
future runs reuse the files.

**Stage 2 — choosing the prompt format (1 minute).**

```
[1/4] Loading TOFU and selecting prompt format
  format chat      mean entity log-prob  -1.2451  OK
  format plain_qa  mean entity log-prob  -3.8820  OK
  format bare      mean entity log-prob  -6.1203  poor
  selected format: chat
```

This is the code working out how to talk to the model. **Higher (less
negative) is better.** It picks the best one automatically.

**Stage 2b — filtering the questions (seconds).**

```
  190/400 examples have a usable entity span (dropped: {'too_many_words': 149, 'suspicious_short_span': 61})
  50 examples tokenised and ready
```

This is the code discarding questions it cannot measure properly. TOFU has 400
questions in this split. Two kinds are removed:

* `too_many_words` — the answer packs several facts together, so there is no
  single thing to score.
* `suspicious_short_span` — the answer is long and rambling, and the only word
  that is not already in the question is an incidental one like "books".
  Scoring that would measure nothing.

**Seeing roughly 190 kept is correct.** If you see far fewer, tell me.

**Stage 3 — the baseline (5–10 minutes).**

```
[2/4] Stage 1: retain90 -> full (baseline, computed once)
  max dS1 = 2.1847; 47/50 examples have at least one KE layer
```

This is an important line. It says the "knows everything" model genuinely knows
things the "never learned it" model does not. **If it says `0/50`, the run
stops** — and that is correct behaviour, not a crash. See Troubleshooting **T5**.

**Stage 4 — the four measurements (15–25 minutes).**

```
[3/4] Stage 2: four source models
  full      computing...
    UDS=0.0031 over 47/50 examples ...
  retain99  computing...
    UDS=0.1602 over 47/50 examples ...
```

**Stage 5 — the results table.** Covered in Part 7.

### 6.4 Leaving it running

You can safely close your laptop. To detach from tmux without stopping the run:

**Press `Ctrl+b`, let go, then press `d`.**

You are back at the normal prompt. The experiment is still running.

### 6.5 Coming back later

```bash
ssh root@dgxhnode1
tmux attach -t uds
```

You are back watching the same run.

### 6.6 If the run died anyway

Nothing is lost. Results are saved after each model. Find your run id:

```bash
ls results/gpu_runs/
```

You will see something like `table2_1B_20260812_143022`. Restart with:

```bash
python -m deeperase.scripts.run_uds_validation --size 1B --run-id table2_1B_20260812_143022 --resume
```

It skips everything already finished and continues from where it stopped.

---

# PART 7 — Read the results

At the end you will see:

```
======================================================================
RESULTS
======================================================================
Stage-2 source   expected  observed      diff
----------------------------------------------------------------------
full                0.002     0.003     0.001
retain99            0.153     0.160     0.007
retain95            0.496     0.474     0.022
retain90            1.000     0.981     0.019

monotonic: True
within tolerance: 4/4

PASS: monotonic and every split within tolerance
```

### How to read this

There are four rows. Each is a model that saw a different amount of the
information we want forgotten:

| Row | That model saw... | So its score should be... |
|---|---|---|
| `full` | everything | near 0 (nothing forgotten) |
| `retain99` | most of it | a bit higher |
| `retain95` | half of it | about half |
| `retain90` | none of it | near 1 (fully forgotten) |

**The single most important thing is that the observed column goes UP as you
read down.** That is `monotonic: True`.

### The three possible outcomes

| Verdict | Meaning | What to do |
|---|---|---|
| **PASS** | Numbers rise, and all are close to published | Excellent. Send me the output. |
| **PARTIAL** | Numbers rise, but some are off | Still good — ordering is the real test. Send me the output. |
| **FAIL** | Numbers do not rise | There is a bug. Send me the output; this is valuable information. |

**All three are worth reporting.** A FAIL found now is far better than building
months of work on a broken measurement.

### Where the files are

```bash
ls results/gpu_runs/table2_1B_*/
```

| File | What it holds |
|---|---|
| `report.json` | The final results |
| `config.json` | Exactly what settings were used |
| `stage1_cache.json` | The baseline, reusable |
| `partial/` | One file per model |

To print the summary again:

```bash
cat results/gpu_runs/table2_1B_*/report.json | python -m json.tool | head -40
```

---

# PART 8 — Get the results back to your laptop

The result files are small (a few MB). On **your laptop**:

```bash
scp -r root@dgxhnode1:/workspace/deeperase/results ./gpu_results
```

And the log:

```bash
scp root@dgxhnode1:/workspace/deeperase/run_log.txt ./
```

**Do not copy `hf_cache`** — that is 10 GB of model weights you do not need
locally.

---

# PART 9 — Troubleshooting

### T1 — `CUDA works: False` but `nvidia-smi` shows a GPU

Python has a processor-only torch. Check which:

```bash
python -c "import torch; print(torch.__version__)"
```

If the version ends in `+cpu`, that is the problem. Install a CUDA build
matching your driver. Check the driver version at the top of `nvidia-smi`
output, then:

```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**Do not ask for a specific version like `torch==2.6.0`.** Each CUDA index
carries different versions, and asking for one that is not there gives the
error you saw earlier. Let pip pick.

### T2 — `No module named 'torch'`

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then re-check with the command in step 1.2.

### T3 — Red ERROR lines during `pip install`

**"Could not find a version that satisfies the requirement torch==..."**
You used the wrong file. Use `requirements-gpu.txt`, not `requirements.txt`.
The GPU file deliberately leaves torch alone.

**"No matching distribution found"** with connection errors — the server cannot
reach the internet. Check with:

```bash
ping -c 3 pypi.org
```

If that fails, you need a proxy or an offline install. Ask your administrator.

### T4 — torch worked before installing, broken after

Something replaced it. Reinstall the CUDA build:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

### T5 — "No example has a Knowledge-Encoding layer"

The run stopped with this message. **This is the code protecting you**, not a
crash. It means the "knows everything" model does not measurably know more than
the "never learned it" model, so the measurement would be meaningless.

Try forcing a different prompt format:

```bash
python -m deeperase.scripts.run_uds_validation --size 1B --prompt-format plain_qa
```

If that also stops, send me the output including the format log-probabilities.
That is a real finding and I need to see it.

### T6 — "CUDA out of memory"

Something else is using the GPU. Check:

```bash
nvidia-smi
```

If another process is listed, stop it. If not, reduce the load:

```bash
python -m deeperase.scripts.run_uds_validation --size 1B --n-examples 25 --max-seq-length 128
```

### T7 — Download is very slow or stalls

Resume it — completed files are kept:

```bash
python -m deeperase.scripts.run_uds_validation --size 1B --run-id <your-run-id> --resume
```

### T8 — "No space left on device"

```bash
df -h /workspace
```

You need at least 15 GB free. To remove the model cache and start over:

```bash
rm -rf hf_cache
```

### T9 — `command not found: tmux`

```bash
apt-get update && apt-get install -y tmux
```

If you cannot install it, use `nohup` instead:

```bash
nohup python -m deeperase.scripts.run_uds_validation --size 1B > run_log.txt 2>&1 &
```

Watch progress with:

```bash
tail -f run_log.txt
```

(Press `Ctrl+c` to stop watching — this does not stop the run.)

---

# Quick reference card

```bash
# Log in and go to the project
ssh root@dgxhnode1
cd /workspace/deeperase

# One-time setup
pip install -r requirements-gpu.txt

# Health checks
python -m pytest tests/ -q                          # expect: 342 passed
python -m deeperase.scripts.check_gpu --size 1B     # expect: READY

# The experiment, protected from disconnects
tmux new -s uds
cd /workspace/deeperase
python -m deeperase.scripts.run_uds_validation --size 1B --n-examples 50 2>&1 | tee run_log.txt

# Detach:  Ctrl+b then d
# Return:  tmux attach -t uds

# If it died
ls results/gpu_runs/
python -m deeperase.scripts.run_uds_validation --size 1B --run-id <ID> --resume
```

---

# What to send me afterwards

Copy and paste these three things:

1. **The prompt-format lines** from step `[1/4]` — all three numbers and which
   was selected.
2. **The `max dS1` line** from step `[2/4]`.
3. **The whole RESULTS table** and the verdict.

Or simply send the entire `run_log.txt`.

If it stopped at a safety gate instead of finishing, send me the message it
stopped with. That is not a failure — it means we caught a real problem before
building anything on top of it.

---

# Folder layout when everything is done

```
/workspace/deeperase/
├── deeperase/                  the code
│   ├── config.py               which models exist, memory planning
│   ├── models.py               loading models, managing GPU memory
│   ├── data/tofu.py            the questions, and which words matter
│   ├── eval/
│   │   ├── patching.py         swapping signals between models
│   │   └── uds.py              the depth score
│   └── scripts/
│       ├── check_gpu.py        preflight
│       └── run_uds_validation.py   the experiment
├── tests/                      342 self-tests
├── requirements-gpu.txt        what to install on a GPU server
├── run_log.txt                 your saved output
│
├── hf_cache/                   ~10 GB of downloaded models  [do not copy]
│
└── results/gpu_runs/
    └── table2_1B_<date>/
        ├── config.json         exact settings used
        ├── stage1_cache.json   the reusable baseline
        ├── partial/            one file per model, saved as it finishes
        └── report.json         the final answer
```
