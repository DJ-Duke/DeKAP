# DeKAP
Implementation of DeKAP

We are actively preparing to release the code of DeKAP. Please stay tuned!

## News
- **2025-09-16** Our paper has been published in IEEE COMML, check it out: [Distillation-Enabled Knowledge Alignment Protocol for Semantic Communication in AI Agent Networks](https://ieeexplore.ieee.org/document/11134386)
- **2025-09-23** Released the distillation demo.

## Dependency
1. Install `uv`.
2. Install venv dependency
```bash
uv sync
```
3. Install the GPU version of torch and the corresponding CUDA dependencies
```bash
uv add --index https://download.pytorch.org/whl/cu124 "torch==2.4.*" torchvision torchaudio

uv sync

. .venv/bin/activate  # Windows: .venv\Scripts\activate

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" # Expected output： 2.4.1+cu124 12.4 True
```

## Distillation Demo
To run the distillation demo, first download the checkpoints ([Google Drive Link](https://drive.google.com/drive/folders/1V-JPboJFg4PNPev0yey7XuUOMwHN7kFn?usp=sharing)) of the fine-tuned models for each task (which contain expert knowledge), and place the folder "ckpt_models" under the project root.

Then, download the dataset files "resEnhance_train.npz" and "resEnhance_test.npz" ([Google Drive Link](https://drive.google.com/drive/folders/1CkGFOv11DjfUR1FYu7_nav7BZ2yMf1fc?usp=sharing)) and move them into the folder "datasets/distilled_dataset" under the project root.

*Currently, only the dataset for the resolution enhancement task is available, but we will upload those of other tasks soon. Still, it is sufficient for running the demo.*

The correct file structure should look like this:
```
DEKAP
-ckpt_models
--full_ft_low_resolution
---FT_curBest.pt
---...
-datasets
--distilled_dataset
---resEnhance_test.npz
---resEnhance_train.npz
-...
```

To run the distillation demo, simply run the `run_distillation.sh` script by terminal command under the project root:
```bash
bash run_script/run_distillation.sh
```