# DeKAP
**面向智能体网络的蒸馏式知识对齐协议（Distillation-Enabled Knowledge Alignment Protocol for AI Agent Networks）**

DeKAP 是一个研究代码库，通过知识蒸馏实现高效的语义通信。

> 英文版说明见 [README.md](README.md)。

## 快速开始

若你只想确认本仓库能否正常运行，从这里开始：

### 方案 A：知识分配基准测试（Allocation Benchmark）

```bash
uv sync --extra allocation
bash run_script/run_allocation.sh --eval-case Time
```

上述命令会：

- 🚀 运行推荐的冒烟测试
- 📊 对比 `prop` 与 `greedy`
- 📝 在日志中打印一行 `Compare prop vs greedy` 摘要

### 方案 B：蒸馏演示（Distillation Demo）

在下载好所需的检查点与数据集文件之后：

```bash
bash run_script/run_distillation.sh
```

## 可以运行什么

- 🚀 **蒸馏演示**：使用已发布的检查点与数据集文件，跑通端到端流程。
- 📊 **知识分配基准测试**：将精确的 Gurobi 混合整数规划基线（`prop`）与所提出的快速贪心方法（`greedy`）进行对比。

## 我该用哪一部分？

- 🧪 若你需要**轻量、可复现**的求解器对比，请使用**知识分配基准测试**。
- 🧠 若你要在检查点与数据齐备的前提下运行**语义通信主流程**，请使用**蒸馏演示**。
- 🔬 若你在**准备发版**、**检查全新环境**或**验证远程环境**，请使用**默认的 `Time` 分配冒烟测试**。

## 动态
- **2025-09-16** 论文已在 IEEE Communications Letters 发表，见：[Distillation-Enabled Knowledge Alignment Protocol for Semantic Communication in AI Agent Networks](https://ieeexplore.ieee.org/document/11134386)
- **2025-09-23** 已发布蒸馏演示相关代码与说明。
- **2026-04** 已发布知识**分配**基准演示（`process/allocation.py`）；详见下文 [知识分配基准测试](#allocation-benchmark)。

## 安装

### 快速配置

1. 安装 `uv`。
2. 创建环境：
```bash
uv sync
```
3. 安装带 CUDA 支持的 PyTorch：
```bash
uv add --index https://download.pytorch.org/whl/cu124 "torch==2.4.*" torchvision torchaudio

uv sync

. .venv/bin/activate  # Windows: .venv\Scripts\activate

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" # 期望输出: 2.4.1+cu124 12.4 True
```
4. 训练过程可视化需要 `wandb`。请在 [wandb.ai](https://wandb.ai/) 注册账号并按官方说明配置；首次运行蒸馏演示时会提示登录。

> 💡 若你只想试用知识分配基准测试，可以跳过蒸馏演示所需的检查点与数据集下载。

### 环境说明

- 主仓库使用 `uv sync` 即可。
- `uv sync --extra allocation` 会安装知识分配基准测试所需的额外依赖。
- 若不用 `uv` 而是自建环境，请确保在同一 Python 环境中安装 `numpy`、`scipy`、`cython` 以及各求解器相关依赖。

## 蒸馏演示

### 概览

运行蒸馏演示需要：

- 📦 位于 `ckpt_models/` 下的已发布检查点
- 🗂️ 位于 `datasets/distilled_dataset/` 的数据集文件
- 🧪 按上文配置好的 Python 环境

### 步骤 1：下载预训练模型
下载各任务专家知识对应的检查点（[Google Drive 链接](https://drive.google.com/drive/folders/1V-JPboJFg4PNPev0yey7XuUOMwHN7kFn?usp=sharing)），将 `ckpt_models` 文件夹放在项目根目录下。

### 步骤 2：下载数据集
下载数据集文件 `resEnhance_train.npz` 与 `resEnhance_test.npz`（[Google Drive 链接](https://drive.google.com/drive/folders/1CkGFOv11DjfUR1FYu7_nav7BZ2yMf1fc?usp=sharing)），放入 `datasets/distilled_dataset/` 目录。

> **说明：** 目前仅提供分辨率增强任务的数据集；其他任务的数据集将陆续上传。本演示在现有数据下可完整运行。

### 步骤 3：确认目录结构
项目结构应类似：
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

### 步骤 4：运行演示
执行蒸馏脚本：
```bash
bash run_script/run_distillation.sh
```

> ✅ 若脚本能正常开始训练/评估且未出现缺文件错误，说明蒸馏演示环境已就绪。

<a id="allocation-benchmark"></a>

## 知识分配基准测试

该可选演示针对**知识分配**混合整数规划形式，对比不同求解器。示例的逐任务损失向量以说明性 `.mat` 文件形式放在 `data/example_loss_data/`。你之后可用相同命名规则替换为自己的数据。

### 标准对比（推荐用于发版与冒烟测试）

请将**默认的 `Time` 冒烟测试**作为「基线 vs 所提出方法」的规范检查。

### 推荐默认设置

- 🔍 模式：`--eval-case Time`
- 👥 问题规模：`N=10`，`L=2`
- 🎯 数据集：`resEnhance`
- 🧮 方法：`prop`（Gurobi 完整 MIP）与 `greedy`（`solve_greedy`）
- 📝 输出：一条名为 `Compare prop vs greedy` 的日志行
- 💾 保存产物：同时包含时间与目标值摘要的 `.mat` 文件

两种方法在**同一**随机实例上运行，因此这是检查贪心求解器相对精确 MIP 基线是否合理的最简方式。

典型输出形如：

```text
Compare prop vs greedy (seed=0, N=10): time prop=...s greedy=...s; objective prop=... greedy=...
```

```bash
bash run_script/run_allocation.sh --eval-case Time
# 可选: --log-dir /path/to/output
```

若以目标值扫参为主（不强调计时），默认 **`Performance`** 冒烟同样使用 **`N=10`** 与 **`prop` + `greedy`**。

### 方法（Methods）

可通过 `--methods`（逗号分隔）或环境变量 `ALLOC_METHODS` 任选子集：

| 键 | 说明 |
|-----|------|
| `prop` | 用 **Gurobi** 求解的完整混合整数规划（MIP 基线）。 |
| `ga` | 遗传算法（**PyGAD**）。 |
| `greedy` | **所提出**的贪心分配（`solve_greedy`）；在较大 `N` 下通常远快于 Gurobi MIP。 |
| `no_save` | 固定「不存储」策略。 |
| `all_save` | 固定「全部存储」策略。 |

> ⚠️ **规模说明：** 随着智能体数量 `N` 增大，Gurobi MIP（`prop`）与 GA（`ga`）会迅速变慢（在 MIP 建模/求解量级上大致为 **O(N³)**）。因此本测试脚本对 `prop` 与 `ga` **强制 `N ≤ 20`**。超过该规模时，单次运行可能占用大量内存，且很容易超过**十分钟**。
>
> ⚡ 所提出的 **`greedy`** 流程**不受**该上限约束，在较大 `N` 下仍较快（例如在典型设置下 `N=50` 量级可在秒级完成）。

### 前置条件

1. **Python** 3.10（见 `pyproject.toml`；可用 `uv sync` 或自建虚拟环境）。
2. **安装 allocation 附加依赖**（Gurobi Python API、PyGAD、Cython、构建工具等）：
   ```bash
   uv sync --extra allocation
   ```
   或在同一环境中执行：`pip install gurobipy pygad cython setuptools`。
3. **Gurobi Optimizer** 及有效许可证（例如 [免费学术许可](https://www.gurobi.com/academia/academic-program-and-licenses/)）：
   - 安装与操作系统匹配的完整 Gurobi 软件。
   - 按 Gurobi 文档申请并安装许可证文件（`grbgetkey` / `gurobi.lic` 等）。若 `gurobipy` 报 `License expired`，请在过期前续期。
   - 验证：`python -c "import gurobipy as gp; print(gp.gurobi.version())"`（或环境中能成功 `import gurobipy` 即可）。
4. **编译 Cython 扩展**（每个环境只需一次，在仓库根目录执行）：
   ```bash
   cd source_alloc && python setup_cython.py build_ext --inplace && cd ..
   ```
   若扩展导入失败，下文的辅助脚本会自动尝试执行该步骤。

> ✅ 多数情况下**无需**手动执行 Cython 编译，因为 `run_script/run_allocation.sh` 在缺少扩展时会自动尝试构建。

### 常用命令

**标准对比**见上文；此处为速查：

```bash
bash run_script/run_allocation.sh --eval-case Time
bash run_script/run_allocation.sh --eval-case Performance
```

**更大预设**（`--extended`）：

- `Time`：默认使用 `N=20`，并默认运行 **`prop`、`ga`、`greedy`**
- `Performance`：在 `N` 网格 `4…20` 上，默认仅 **`greedy`**

```bash
bash run_script/run_allocation.sh --eval-case Time --extended
```

显式指定方法的示例：

```bash
bash run_script/run_allocation.sh --eval-case Time --methods prop,ga,greedy
bash run_script/run_allocation.sh --eval-case Performance --methods prop,greedy --extended
```

- `--eval-case Time` — 墙钟计时（默认评估模式）。
- `--eval-case Performance` — 跨方法对比目标值。
- `--extended` — 使用上述更大预设；省略则使用较小默认配置。
- `--methods` — 逗号分隔的方法键（见上表）；会覆盖该模式下的预设方法列表。
- `--log-dir PATH` — 可选：日志与 `.mat` 摘要的输出目录（默认：`logs_eval_4c_time` 或 `logs_eval_4c3`）。

环境变量：若未传 `--eval-case`，可由 `ALLOC_EVAL_CASE` 指定评估模式；`ALLOC_EXTENDED=1` 等价于 `--extended`；`ALLOC_METHODS` 的用法类似 `--methods`。

### 输出

日志与 MATLAB `.mat` 摘要写入所选日志目录。

在 **`Time`** 模式下，每个方法的 `.mat` 通常包含：

- `time_summary_*` — 墙钟时间（秒）
- `objective_summary_*` — 该次求解的总目标值

视许可证与设置而定，Gurobi 也可能在当前工作目录写入 `gurobi.log`。

> 📌 若你在准备发版或做端到端自检，建议从下面命令开始：
>
> `bash run_script/run_allocation.sh --eval-case Time`

### 故障排除

- ❌ `License expired` 或 Gurobi 导入错误：先检查 Gurobi 安装与许可证文件。
- ❌ 缺少 Cython 扩展：再运行一次 `bash run_script/run_allocation.sh ...`；辅助脚本可能会自动完成构建。
- ⏳ `prop` 或 `ga` 很慢：在较大 `N` 下属预期；可先跑默认冒烟，或改用 `greedy`。
- 📁 希望把输出放到其他位置：使用 `--log-dir /your/output/path`。
