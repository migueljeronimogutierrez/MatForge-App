"""
Standalone diagnostic — verifies the full local inference pipeline before
Streamlit development begins. Run once from the project root with the venv
activated:

    python scripts/matforge_app_00_inference_check.py
"""

import gc
import logging
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_MATFORGE   = Path("checkpoints/matforge/best_gan.pt")
CHECKPOINT_SR_PRIMARY = Path("checkpoints/sr/sr_ft_phase1_best_lpips.pt")
CHECKPOINT_SR_FALLBACK = Path("checkpoints/sr/RealESRGAN_x4plus.pth")
ARTIFACTS_DIR         = Path("artifacts/")

# ImageNet normalisation constants used at training time
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Model definitions (verbatim from architecture spec)
# ---------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    def __init__(self, in_channels=(64, 128, 320, 512), out_channels=256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(64,  256, 1, bias=False),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.Conv2d(320, 256, 1, bias=False),
            nn.Conv2d(512, 256, 1, bias=False),
        ])
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)
        ])

    def forward(self, features):
        # features: [L1, L2, L3, L4] — shallow to deep
        projected = [proj(f) for proj, f in zip(self.proj, features)]
        x = projected[-1]                                         # L4, deepest
        for i in range(len(self.merge) - 1, -1, -1):
            x = F.interpolate(x, size=projected[i].shape[-2:], mode="nearest")
            x = self.merge[i](torch.cat([x, projected[i]], dim=1))
        return x                                                  # (B, 256, H/4, W/4)


class RefineHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample layers carry no parameters and are not in the state dict;
        # they live in forward() so the Sequential index positions match the checkpoint.
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(64, out_channels, 1)  # bias confirmed in checkpoint

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)
        return self.out(x)


class MatForgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder       = timm.create_model("pvt_v2_b1", pretrained=False, features_only=True)
        self.fpn           = FPNDecoder(in_channels=(64, 128, 320, 512), out_channels=256)
        self.head_normal    = RefineHead(256, 3)
        self.head_roughness = RefineHead(256, 1)
        self.head_metallic  = RefineHead(256, 1)

    def forward(self, x):
        features    = self.encoder(x)
        fpn_out     = self.fpn(features)
        raw_normal    = self.head_normal(fpn_out)
        raw_roughness = self.head_roughness(fpn_out)
        raw_metallic  = self.head_metallic(fpn_out)
        # normal is unit-length in [-1, 1]; roughness is [0, 1];
        # metallic logits are left raw — sigmoid applied at inference
        normal    = F.normalize(torch.tanh(raw_normal), dim=1, eps=1e-6)
        roughness = torch.sigmoid(raw_roughness)
        return {"normal": normal, "roughness": roughness, "metallic": raw_metallic}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0


def _hann_window(n: int, device, dtype) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype)
    w1d = torch.sin(math.pi * k / (n - 1)) ** 2
    return torch.outer(w1d, w1d)  # (N, N)


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def step1_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    log.info("Step 1 — Device detection")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("  GPU: %s | Total VRAM: %.0f MB", props.name, props.total_memory / 1024 ** 2)
    else:
        log.info("  Running on CPU")
    log.info("  dtype: %s", dtype)
    return device, dtype


def step2_vram_baseline(device: str) -> float:
    log.info("Step 2 — VRAM baseline")
    mb = _vram_mb()
    log.info("  Allocated before loading: %.1f MB", mb)
    return mb


def step3_load_matforgenet(device: str, dtype: torch.dtype) -> MatForgeNet:
    log.info("Step 3 — Load MatForgeNet")
    ckpt = torch.load(CHECKPOINT_MATFORGE, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    model = MatForgeNet()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device).to(dtype)
    log.info("  Loaded. VRAM after: %.1f MB", _vram_mb())
    return model


def step4_build_synthetic(device: str, dtype: torch.dtype) -> torch.Tensor:
    log.info("Step 4 — Synthetic test image")
    img  = torch.rand(1, 3, 512, 512)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(STD).view(1, 3, 1, 1)
    img  = (img - mean) / std
    img  = img.to(device).to(dtype)
    log.info("  Image shape: %s", tuple(img.shape))
    return img


def step5_tiled_inference(
    model: MatForgeNet,
    img: torch.Tensor,
    device: str,
    dtype: torch.dtype,
):
    log.info("Step 5 — Tile-and-merge inference")
    TILE   = 256
    STRIDE = 128

    _, _, H, W = img.shape
    pad_h = (STRIDE - H % STRIDE) % STRIDE
    pad_w = (STRIDE - W % STRIDE) % STRIDE
    img_p = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_p, W_p = img_p.shape

    hann = _hann_window(TILE, device=device, dtype=torch.float32)

    acc_n = torch.zeros(1, 3, H_p, W_p, device=device, dtype=torch.float32)
    acc_r = torch.zeros(1, 1, H_p, W_p, device=device, dtype=torch.float32)
    acc_m = torch.zeros(1, 1, H_p, W_p, device=device, dtype=torch.float32)
    acc_w = torch.zeros(1, 1, H_p, W_p, device=device, dtype=torch.float32)

    # AMP is enabled only on CUDA; on CPU the context is a no-op
    autocast_ctx = torch.amp.autocast("cuda", enabled=(device == "cuda"))

    with torch.no_grad(), autocast_ctx:
        for y in range(0, H_p - TILE + 1, STRIDE):
            for x in range(0, W_p - TILE + 1, STRIDE):
                tile = img_p[:, :, y : y + TILE, x : x + TILE]
                out  = model(tile)
                w    = hann.unsqueeze(0).unsqueeze(0)             # (1,1,T,T)
                acc_n[:, :, y : y + TILE, x : x + TILE] += out["normal"].float()    * w
                acc_r[:, :, y : y + TILE, x : x + TILE] += out["roughness"].float() * w
                acc_m[:, :, y : y + TILE, x : x + TILE] += out["metallic"].float()  * w
                acc_w[:, :, y : y + TILE, x : x + TILE] += w

    normal    = F.normalize(acc_n / acc_w, dim=1, eps=1e-6)
    roughness = acc_r / acc_w
    metallic  = torch.sigmoid(acc_m / acc_w)

    # Crop back to original spatial dimensions
    normal    = normal[:, :, :H, :W]
    roughness = roughness[:, :, :H, :W]
    metallic  = metallic[:, :, :H, :W]

    log.info("  normal    shape: %s", tuple(normal.shape))
    log.info("  roughness shape: %s", tuple(roughness.shape))
    log.info("  metallic  shape: %s", tuple(metallic.shape))
    return normal, roughness, metallic


def step6_verify_outputs(normal, roughness, metallic) -> dict:
    log.info("Step 6 — Output verification")
    results = {}

    def _check(label, ok, detail):
        tag = "PASS" if ok else "FAIL"
        log.info("  %-36s %s  %s", label, tag, detail)
        return ok

    # Normal map
    ok = _check("normal shape (1,3,512,512)",
                tuple(normal.shape) == (1, 3, 512, 512),
                str(tuple(normal.shape)))
    results["normal_shape"] = ok

    n_min, n_max = normal.min().item(), normal.max().item()
    ok = _check("normal range [-1, 1]",
                -1.0 <= n_min and n_max <= 1.0,
                f"[{n_min:.4f}, {n_max:.4f}]")
    results["normal_range"] = ok

    norm_mean = normal.norm(dim=1).mean().item()
    ok = _check("normal per-pixel unit norm ≈ 1.0",
                abs(norm_mean - 1.0) < 0.05,
                f"mean={norm_mean:.4f}")
    results["normal_unit"] = ok

    # Roughness map
    ok = _check("roughness shape (1,1,512,512)",
                tuple(roughness.shape) == (1, 1, 512, 512),
                str(tuple(roughness.shape)))
    results["roughness_shape"] = ok

    r_min, r_max = roughness.min().item(), roughness.max().item()
    ok = _check("roughness range [0, 1]",
                0.0 <= r_min and r_max <= 1.0,
                f"[{r_min:.4f}, {r_max:.4f}]")
    results["roughness_range"] = ok

    # Metallic map
    ok = _check("metallic shape (1,1,512,512)",
                tuple(metallic.shape) == (1, 1, 512, 512),
                str(tuple(metallic.shape)))
    results["metallic_shape"] = ok

    m_min, m_max = metallic.min().item(), metallic.max().item()
    ok = _check("metallic range [0, 1]",
                0.0 <= m_min and m_max <= 1.0,
                f"[{m_min:.4f}, {m_max:.4f}]")
    results["metallic_range"] = ok

    if not all(results.values()):
        raise AssertionError("One or more output checks failed — see log above.")
    return results


def step7_free_model(model: MatForgeNet, device: str) -> None:
    log.info("Step 7 — Free MatForgeNet from GPU")
    if model is None:
        raise RuntimeError("MatForgeNet was not loaded — nothing to free.")
    before = _vram_mb()
    model.to("cpu")          # transfers all parameters off the GPU
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    after = _vram_mb()
    log.info("  VRAM: %.1f MB → %.1f MB", before, after)
    if device == "cuda" and after >= before:
        raise RuntimeError(f"VRAM did not drop after release: {before:.1f} → {after:.1f} MB")


def step8_knn_pipeline() -> None:
    log.info("Step 8 — KNN artifacts")
    pca = joblib.load(ARTIFACTS_DIR / "pca_model.pkl")
    knn = joblib.load(ARTIFACTS_DIR / "knn_classifier.pkl")
    le  = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    log.info("  Classes: %s", list(le.classes_[:8]))
    dummy   = np.random.randn(1, 384)
    reduced = pca.transform(dummy)           # → (1, 50)
    label   = le.inverse_transform(knn.predict(reduced))[0]
    log.info("  KNN pipeline: OK — dummy prediction: %s", label)


def step9_sr_checkpoint() -> None:
    log.info("Step 9 — SR checkpoint existence")
    for path in (CHECKPOINT_SR_PRIMARY, CHECKPOINT_SR_FALLBACK):
        if path.exists():
            size_mb = path.stat().st_size / 1024 ** 2
            log.info("  Found: %s (%.1f MB)", path, size_mb)
            return
    raise FileNotFoundError(
        f"No SR checkpoint found at {CHECKPOINT_SR_PRIMARY} or {CHECKPOINT_SR_FALLBACK}"
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(summary: dict) -> None:
    col = 38
    sep = "=" * (col + 10)
    print(f"\n{sep}")
    print(f"  {'Step':<{col}} Result")
    print("-" * (col + 10))
    for step, result in summary.items():
        print(f"  {step:<{col}} {result}")
    passed = sum(1 for v in summary.values() if v == "PASS")
    total  = len(summary)
    print(sep)
    print(f"  {passed}/{total} steps passed")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    summary = {
        "1  Device detection"    : "FAIL",
        "2  VRAM baseline"       : "FAIL",
        "3  Load MatForgeNet"    : "FAIL",
        "4  Synthetic image"     : "FAIL",
        "5  Tiled inference"     : "FAIL",
        "6  Output verification" : "FAIL",
        "7  Free GPU memory"     : "FAIL",
        "8  KNN pipeline"        : "FAIL",
        "9  SR checkpoint"       : "FAIL",
    }

    device = dtype = None
    model  = None
    img    = None
    normal = roughness = metallic = None

    try:
        device, dtype = step1_device()
        summary["1  Device detection"] = "PASS"
    except Exception as exc:
        log.error("Step 1 failed: %s", exc)
        _print_summary(summary)
        sys.exit(1)

    try:
        step2_vram_baseline(device)
        summary["2  VRAM baseline"] = "PASS"
    except Exception as exc:
        log.error("Step 2 failed: %s", exc)

    try:
        model = step3_load_matforgenet(device, dtype)
        summary["3  Load MatForgeNet"] = "PASS"
    except Exception as exc:
        log.error("Step 3 failed: %s", exc)

    try:
        img = step4_build_synthetic(device, dtype)
        summary["4  Synthetic image"] = "PASS"
    except Exception as exc:
        log.error("Step 4 failed: %s", exc)

    if model is not None and img is not None:
        try:
            normal, roughness, metallic = step5_tiled_inference(model, img, device, dtype)
            summary["5  Tiled inference"] = "PASS"
        except Exception as exc:
            log.error("Step 5 failed: %s", exc)

        if normal is not None:
            try:
                step6_verify_outputs(normal, roughness, metallic)
                summary["6  Output verification"] = "PASS"
            except Exception as exc:
                log.error("Step 6 failed: %s", exc)

    try:
        step7_free_model(model, device)
        model = None
        summary["7  Free GPU memory"] = "PASS"
    except Exception as exc:
        log.error("Step 7 failed: %s", exc)
        model = None

    try:
        step8_knn_pipeline()
        summary["8  KNN pipeline"] = "PASS"
    except Exception as exc:
        log.error("Step 8 failed: %s", exc)

    try:
        step9_sr_checkpoint()
        summary["9  SR checkpoint"] = "PASS"
    except Exception as exc:
        log.error("Step 9 failed: %s", exc)

    _print_summary(summary)


if __name__ == "__main__":
    main()
