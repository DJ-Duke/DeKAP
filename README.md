# DeKAP
**Distillation-Enabled Knowledge Alignment Protocol for AI Agent Networks**

An implementation of DeKAP that enables efficient semantic communication through knowledge distillation.

## News
- **2025-09-16** Our paper has been published in IEEE Communications Letters! Check it out: [Distillation-Enabled Knowledge Alignment Protocol for Semantic Communication in AI Agent Networks](https://ieeexplore.ieee.org/document/11134386)
- **2025-09-23** Released the distillation demo. The allocation demo will be released soon.

## Installation

### Prerequisites
1. Install `uv`
2. Set up the virtual environment:
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
4. Need to use `wandb` for training visualization. It needs to register online ([Link](https://wandb.ai/)) to get an account to use. Please follow the guideline there.

## Distillation Demo

To run the distillation demo, follow these steps:

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

