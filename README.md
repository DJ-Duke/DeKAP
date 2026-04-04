# DeKAP
**Distillation-Enabled Knowledge Alignment Protocol for AI Agent Networks**

DeKAP is a research codebase for efficient semantic communication through knowledge distillation.

[中文版README](./README_CN.md)

## Quick Start

If you just want to verify that the repository works, start here:

### Option A: Allocation Benchmark

```bash
uv sync --extra allocation
bash run_script/run_allocation.sh --eval-case Time
```

What this does:

- 🚀 runs the recommended smoke test
- 📊 compares `prop` vs `greedy`
- 📝 prints a `Compare prop vs greedy` summary in the log

### Option B: Distillation Demo

After downloading the required checkpoints and dataset files:

```bash
bash run_script/run_distillation.sh
```

## What You Can Run

- 🚀 **Distillation demo**: run the end-to-end demo with released checkpoints and dataset files.
- 📊 **Allocation benchmark**: compare the exact Gurobi MIP baseline (`prop`) against the proposed fast greedy method (`greedy`).

## Which Part Should I Use?

- 🧪 Use the **allocation benchmark** if you want a lightweight, reproducible solver comparison.
- 🧠 Use the **distillation demo** if you want to run the main semantic communication pipeline with checkpoints and data.
- 🔬 Use the **default `Time` allocation smoke** if you are preparing a release, checking a fresh environment, or validating remote setup.

## News
- **2025-09-16** Our paper has been published in IEEE Communications Letters! Check it out: [Distillation-Enabled Knowledge Alignment Protocol for Semantic Communication in AI Agent Networks](https://ieeexplore.ieee.org/document/11134386)
- **2025-09-23** Released the distillation demo.
- **2026-04** Released the knowledge **allocation** benchmark demo (`process/allocation.py`); see [Allocation benchmark](#allocation-benchmark) below.

## Installation

### Quick Setup

1. Install `uv`.
2. Create the environment:
```bash
uv sync
```
3. Install PyTorch with CUDA support:
```bash
uv add --index https://download.pytorch.org/whl/cu124 "torch==2.4.*" torchvision torchaudio

uv sync

. .venv/bin/activate  # Windows: .venv\Scripts\activate

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" # Expected output: 2.4.1+cu124 12.4 True
```
4. `wandb` is required for training visualization. Please create an account at [wandb.ai](https://wandb.ai/) and follow the official setup instructions. You will be prompted to log in the first time you run the distillation demo.

> 💡 If you only want to try the allocation benchmark, you can skip the checkpoint and dataset downloads required by the distillation demo.

### Environment Notes

- `uv sync` is enough for the main repository setup.
- `uv sync --extra allocation` installs the extra dependencies needed for the allocation benchmark.
- If you are using a custom environment instead of `uv`, make sure `numpy`, `scipy`, `cython`, and any solver-specific dependencies are installed in the same Python environment.

## Distillation Demo

### Overview

To run the distillation demo, you need:

- 📦 released checkpoints under `ckpt_models/`
- 🗂️ dataset files under `datasets/distilled_dataset/`
- 🧪 a working Python environment from the setup above

### Step 1: Download Pre-trained Models
Download the checkpoints ([Google Drive Link](https://drive.google.com/drive/folders/1V-JPboJFg4PNPev0yey7XuUOMwHN7kFn?usp=sharing)) containing expert knowledge for each task, and place the `ckpt_models` folder under the project root.

### Step 2: Download Dataset
Download the dataset files `resEnhance_train.npz` and `resEnhance_test.npz` ([Google Drive Link](https://drive.google.com/drive/folders/1CkGFOv11DjfUR1FYu7_nav7BZ2yMf1fc?usp=sharing)) and place them in the `datasets/distilled_dataset` directory.

> **Note:** Currently, only the resolution enhancement task dataset is available. More datasets for other tasks will be uploaded soon. This demo is fully functional.

### Step 3: Verify File Structure
Your project structure should look like this:
```
DeKAP/
├── ckpt_models/
│   └── full_ft_low_resolution/
│       └── FT_curBest.pt
│       └── ...
├── datasets/
│   └── distilled_dataset/
│       ├── resEnhance_test.npz
│       └── resEnhance_train.npz
└── ...
```

### Step 4: Run the Demo
Execute the distillation script:
```bash
bash run_script/run_distillation.sh
```

> ✅ If the script starts training / evaluation without missing-file errors, your distillation demo setup is ready.

## Allocation benchmark

This optional demo compares solvers for the **knowledge allocation** mixed-integer formulation. Example per-task loss vectors ship under `data/example_loss_data/` as **illustrative** `.mat` inputs. You can later replace them with your own files using the same naming pattern.

### Standard comparison (recommended for releases and smoke tests)

Use the **default `Time` smoke** as the canonical baseline-vs-proposed check.

### Recommended Default

- 🔍 Mode: `--eval-case Time`
- 👥 Problem size: `N=10`, `L=2`
- 🎯 Dataset: `resEnhance`
- 🧮 Methods: `prop` (Gurobi full MIP) and `greedy` (`solve_greedy`)
- 📝 Output: one log line named `Compare prop vs greedy`
- 💾 Saved artifacts: `.mat` files with both timing and objective summaries

Both methods run on the **same** random instance, so this is the easiest way to check whether the proposed greedy solver is behaving sensibly relative to the exact MIP baseline.

Typical output looks like:

```text
Compare prop vs greedy (seed=0, N=10): time prop=...s greedy=...s; objective prop=... greedy=...
```

```bash
bash run_script/run_allocation.sh --eval-case Time
# optional: --log-dir /path/to/output
```

For objective-only sweeps (no timing focus), default **`Performance`** smoke uses the same **`N=10`** and **`prop` + `greedy`**.

### Methods

You can select any subset via `--methods` (comma-separated) or `ALLOC_METHODS`:

| Key | Description |
|-----|-------------|
| `prop` | Full mixed-integer program solved with **Gurobi** (baseline MIP). |
| `ga` | Genetic algorithm (**PyGAD**). |
| `greedy` | **Proposed** greedy allocation (`solve_greedy`); typically much faster than the Gurobi MIP at larger `N`. |
| `no_save` | Fixed “no storage” policy. |
| `all_save` | Fixed “store all” policy. |

> ⚠️ **Scaling note:** Gurobi MIP (`prop`) and GA (`ga`) slow down quickly as the number of agents `N` increases (roughly **O(N³)** in the MIP construction / solve regime). This harness therefore **enforces `N ≤ 20`** for `prop` and `ga`. Beyond that, a single run may consume a lot of memory and can easily take **more than ten minutes**.
>
> ⚡ The proposed **`greedy`** routine is **not** subject to that cap and remains fast at larger `N` (for example, on the order of seconds at `N=50` in typical settings).

### Prerequisites

1. **Python** 3.10 (see `pyproject.toml`; use `uv sync` or your own venv).
2. **Install allocation extras** (Gurobi Python API, PyGAD, Cython, build tools):
   ```bash
   uv sync --extra allocation
   ```
   Or: `pip install gurobipy pygad cython setuptools` into the same environment.
3. **Gurobi Optimizer** with a valid license (e.g. [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/)):
   - Install the full Gurobi software matching your OS.
   - Request and install the license file (`grbgetkey` / `gurobi.lic` as per Gurobi docs). Renew before expiry if you see `License expired` from `gurobipy`.
   - Verify: `python -c "import gurobipy as gp; print(gp.gurobi.version())"` (or any successful `import gurobipy` in your environment).
4. **Build Cython extensions** (once per environment, from the repository root):
   ```bash
   cd source_alloc && python setup_cython.py build_ext --inplace && cd ..
   ```
   The helper script below runs this step automatically if the extension import fails.

> ✅ In most cases you do **not** need to run the Cython build step manually, because `run_script/run_allocation.sh` will try to build it automatically if the extension is missing.

### Quick Commands

The **standard comparison** is documented above. Here is the quick reference:

```bash
bash run_script/run_allocation.sh --eval-case Time
bash run_script/run_allocation.sh --eval-case Performance
```

**Larger preset** (`--extended`):

- `Time`: uses `N=20` and runs **`prop`, `ga`, `greedy`** by default
- `Performance`: uses the `N` grid `4…20` with **`greedy`** only by default

```bash
bash run_script/run_allocation.sh --eval-case Time --extended
```

Examples with explicit methods:

```bash
bash run_script/run_allocation.sh --eval-case Time --methods prop,ga,greedy
bash run_script/run_allocation.sh --eval-case Performance --methods prop,greedy --extended
```

- `--eval-case Time` — wall-clock timing (default eval mode).
- `--eval-case Performance` — objective comparison across methods.
- `--extended` — use the larger presets above; omit for the small default.
- `--methods` — comma-separated method keys (see table); overrides the preset for that mode.
- `--log-dir PATH` — optional output directory for logs and `.mat` summaries (defaults: `logs_eval_4c_time` or `logs_eval_4c3`).

Environment variables: `ALLOC_EVAL_CASE` sets the eval mode if `--eval-case` is omitted; `ALLOC_EXTENDED=1` is equivalent to `--extended`; `ALLOC_METHODS` can list methods like `--methods`.

### Outputs

Logs and MATLAB `.mat` summaries are written under the chosen log directory.

In **`Time`** mode, each method’s `.mat` includes:

- `time_summary_*` — wall-clock time in seconds
- `objective_summary_*` — total objective value for that solve

Gurobi may also write `gurobi.log` in the current working directory depending on your license and settings.

> 📌 If you are preparing a release or checking that the repo still works end-to-end, start with:
>
> `bash run_script/run_allocation.sh --eval-case Time`

### Troubleshooting

- ❌ `License expired` or Gurobi import errors:
  check your Gurobi installation and license file first.
- ❌ Missing Cython extension:
  rerun `bash run_script/run_allocation.sh ...` once; the helper script may build it automatically.
- ⏳ `prop` or `ga` feels too slow:
  this is expected at larger `N`; try the default smoke first, or switch to `greedy`.
- 📁 Want to store outputs elsewhere:
  use `--log-dir /your/output/path`.