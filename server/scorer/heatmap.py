"""Region failure attribution — generate heatmap data for frontend visualization."""
from __future__ import annotations

from .scorer import ForgeScore


def generate_heatmap_data(score: ForgeScore, image_width: int = 1024, image_height: int = 1024) -> dict:
    """Convert region scores to heatmap overlay data for the frontend canvas."""
    regions = []
    for r in score.regions:
        # Color based on composite score
        if r.composite >= 0.8:
            color = "#22c55e"  # green — good
        elif r.composite >= 0.6:
            color = "#eab308"  # yellow — partial
        elif r.composite >= 0.4:
            color = "#f97316"  # orange — needs work
        else:
            color = "#ef4444"  # red — failed
        
        regions.append({
            "id": r.element_id,
            "label": r.label,
            "color": color,
            "score": round(r.composite, 2),
            "opacity": 0.3 + (1.0 - r.composite) * 0.4,  # more opaque = worse
            "diagnosis": _region_diagnosis(r),
        })
    
    return {
        "regions": regions,
        "overall_score": round(score.overall, 2),
        "converged": score.converged,
    }


def _region_diagnosis(region) -> str:
    if not region.present:
        return "Missing from output"
    if region.bbox_iou < 0.5:
        return f"Misplaced (IoU: {region.bbox_iou:.2f})"
    if region.dino_similarity < 0.6:
        return f"Visual mismatch (sim: {region.dino_similarity:.2f})"
    return "Good match"
