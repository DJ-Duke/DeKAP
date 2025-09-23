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
To run the distillation demo, 