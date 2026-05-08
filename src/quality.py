"""
src/quality.py
Heuristic quality evaluation for AI-generated PBR normal maps.

Pure logic – only NumPy / SciPy, no Streamlit, no GPU, no state.
All functions accept (H, W, 3) float32 arrays with unit vectors in [-1, 1]
and return diagnostic scores along with a heatmap overlay.
"""

import numpy as np
from scipy.ndimage import sobel, generic_filter


def evaluate_normal_quality(normal_map: np.ndarray) -> dict:
    """Assess the quality of a normal map using three complementary heuristics.

    Args:
        normal_map: Normal map as (H, W, 3) float32, vectors in [-1, 1].
                   They are expected to be approximately unit-length.

    Returns:
        Dictionary with keys:
            - "coherence_score": float in [0, 1] – how many vectors are unit-length.
            - "continuity_score": float in [0, 1] – smoothness of the normal field.
            - "blockiness_score": float in [0, 1] – absence of block/streak artifacts.
            - "overall_score": float in [0, 1] – weighted average of the three.
            - "heatmap": np.ndarray (H, W, 4) uint8 – RGBA overlay for problematic areas.
            - "warnings": list[str] – messages triggered by low scores.

    Notes:
        Weights: 0.40 * coherence + 0.35 * continuity + 0.25 * blockiness.
        Warning thresholds (hard-coded):
            - coherence < 0.95
            - continuity < 0.80
            - blockiness < 0.85
    """
    # --- compute individual metrics ---
    coh_score, coh_map = _compute_coherence(normal_map)
    cont_score, cont_map = _compute_continuity(normal_map)
    block_score, block_map = _compute_blockiness(normal_map)

    # --- aggregated score ---
    overall = (
        0.40 * coh_score
        + 0.35 * cont_score
        + 0.25 * block_score
    )

    # --- warnings ---
    warnings = []
    if coh_score < 0.95:
        warnings.append("Low coherence: many vectors deviate from unit length.")
    if cont_score < 0.80:
        warnings.append("Low continuity: sharp gradients suggest seam artifacts.")
    if block_score < 0.85:
        warnings.append("Low blockiness: possible patch-blocking pattern detected.")

    # --- diagnostic heatmap ---
    heatmap = _build_heatmap_overlay(coh_map, cont_map, block_map)

    return {
        "coherence_score": float(coh_score),
        "continuity_score": float(cont_score),
        "blockiness_score": float(block_score),
        "overall_score": float(overall),
        "heatmap": heatmap,
        "warnings": warnings,
    }


def _compute_coherence(normal_map: np.ndarray) -> tuple[float, np.ndarray]:
    """Measure how many per-pixel vectors have unit length.

    Args:
        normal_map: (H, W, 3) float32, values in [-1, 1].

    Returns:
        coherence_score: fraction of pixels with norm ∈ [0.95, 1.05].
        error_map: (H, W) float32 in [0, 1], |norm - 1| / 0.5 clipped to [0, 1].
                   Higher values indicate degraded vectors.
    """
    norms = np.linalg.norm(normal_map, axis=2)  # (H, W)
    coherent = (norms >= 0.95) & (norms <= 1.05)
    score = np.mean(coherent)

    error = np.abs(norms - 1.0)
    error_map = np.clip(error / 0.5, 0.0, 1.0)
    return float(score), error_map.astype(np.float32)


def _compute_continuity(normal_map: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate the smoothness of the normal field by gradient magnitude.

    High gradient magnitude indicates abrupt changes (possible seams between
    tiles or other artificial borders).

    Args:
        normal_map: (H, W, 3) float32, values in [-1, 1].

    Returns:
        continuity_score: 1 - (mean_gradient / 0.5), clamped to [0, 1].
                          Lower when the mean gradient is high.
        grad_map: (H, W) float32 in [0, 1], gradient magnitude normalised by 0.5.

    Notes:
        The threshold 0.5 was calibrated on PBR textures in 256–1024 px range.
    """
    grad_sq = np.zeros(normal_map.shape[:2], dtype=np.float64)
    for c in range(3):
        gx = sobel(normal_map[..., c], axis=1)  # horizontal
        gy = sobel(normal_map[..., c], axis=0)  # vertical
        grad_sq += gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2
    grad_mag = np.sqrt(grad_sq).astype(np.float32)

    mean_grad = np.mean(grad_mag)
    score = 1.0 - min(mean_grad / 0.5, 1.0)
    score = max(0.0, min(1.0, score))

    # gradient map normalised; saturate at 0.5 for visualisation
    grad_map = np.clip(grad_mag / 0.5, 0.0, 1.0)
    return float(score), grad_map


def _compute_blockiness(normal_map: np.ndarray) -> tuple[float, np.ndarray]:
    """Detect blocky/streaky artifacts via local entropy on a 16×16 window.

    Patch-based models can leave visible grid patterns when blending is
    imperfect. Low local entropy inside a block and high entropy at borders
    increase the standard deviation of the entropy map.

    Args:
        normal_map: (H, W, 3) float32, values in [-1, 1].

    Returns:
        blockiness_score: 1 - (std(entropy_map) / 0.5), clamped to [0, 1].
                          Low values indicate likely blocking.
        blockiness_map: (H, W) float32 in [0, 1], inverted normalised entropy.
                        Bright areas = suspiciously homogeneous (blocky).

    Notes:
        Entropy is computed over 8 bins in [-1, 1] using a 16×16 footprint.
        The threshold 0.5 was tuned for PVT-v2-B1 typical outputs.
    """
    # simple luminance from normal directions
    gray = np.mean(normal_map, axis=2)  # (H, W) in [-1, 1]

    # internal entropy function for generic_filter
    def _entropy_16x16(window: np.ndarray) -> float:
        hist, _ = np.histogram(window, bins=8, range=(-1.0, 1.0))
        hist = hist.astype(np.float64) / window.size
        hist = hist[hist > 0]  # only non-empty bins
        return -np.sum(hist * np.log2(hist + 1e-8))

    footprint = np.ones((16, 16), dtype=bool)
    # generic_filter needs float64 input to avoid precision issues
    entropy_map = generic_filter(
        gray.astype(np.float64),
        _entropy_16x16,
        footprint=footprint,
        mode="reflect",
    ).astype(np.float32)

    std_entropy = float(np.std(entropy_map))
    score = 1.0 - min(std_entropy / 0.5, 1.0)
    score = max(0.0, min(1.0, score))

    # invert entropy: low entropy → high blockiness suspicion
    ent_min = np.min(entropy_map)
    ent_max = np.max(entropy_map)
    if ent_max - ent_min < 1e-8:
        norm_ent = np.zeros_like(entropy_map)
    else:
        norm_ent = (entropy_map - ent_min) / (ent_max - ent_min)
    blockiness_map = 1.0 - norm_ent  # "inverted": low entropy = bright
    return float(score), blockiness_map.astype(np.float32)


def _build_heatmap_overlay(
    coherence_map: np.ndarray,
    continuity_map: np.ndarray,
    blockiness_map: np.ndarray,
) -> np.ndarray:
    """Combine the three error maps into a translucent RGBA overlay.

    Args:
        coherence_map:  (H, W) float32 in [0, 1] – high = problematic.
        continuity_map: (H, W) float32 in [0, 1] – high = problematic.
        blockiness_map: (H, W) float32 in [0, 1] – high = problematic.

    Returns:
        (H, W, 4) uint8 array, values in [0, 255].
        R: continuity, G: blockiness, B: coherence, A: max(R,G,B) * 0.6.
    """
    h, w = coherence_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = continuity_map  # R
    overlay[..., 1] = blockiness_map  # G
    overlay[..., 2] = coherence_map   # B
    overlay[..., 3] = np.maximum(
        np.maximum(continuity_map, blockiness_map), coherence_map
    ) * 0.6  # A scaled to 60%

    overlay_uint8 = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay_uint8