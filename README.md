# MatForge — PBR Map Prediction Tool

Local Streamlit application for predicting PBR material maps (Normal, Roughness, Metallic)
from a single RGB surface image.

## Model

- **Encoder**: PVT-v2-B1 (~13M parameters)
- **Decoder**: FPN-style with three independent refine heads
- **SR module**: RRDBNet ×4, 23 RRDB blocks (Real-ESRGAN)
- **Material classifier**: DINOv2-small + PCA-50 + KNN

Final checkpoint: `best_gan.pt` (GAN fine-tuning epoch 11)
— MAE Normal: 10.37° | Roughness MAE: 0.1117 | LPIPS: 0.0976

## Hardware requirements

- NVIDIA GPU with ≥4 GB VRAM (tested on GTX 1650 Max-Q)
- 16 GB RAM
- Python 3.11.9

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
streamlit run app.py
```

## Large files (Git LFS)

Checkpoints are tracked via Git LFS. After cloning, run:

```bash
git lfs pull
```

## Project structure

```
checkpoints/matforge/   — MatForge trained weights
checkpoints/sr/         — SR module weights
artifacts/              — KNN classifier, PCA model, label encoder
scripts/                — Diagnostic scripts (not part of the app)
src/                    — Application source modules
assets/                 — Static UI resources
```