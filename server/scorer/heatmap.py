"""Region failure attribution — generate heatmap data for frontend visualization."""
from __future__ import annotations

from .scorer import ForgeScore


def generate_heatmap_data(score: ForgeScore, image_width: int = 1024, image_height: int = 1024) -> dict:
    """Convert region scores to heatmap overlay data for the frontend canvas."""
    regions = []
    for i, r in enumerate(score.regions):
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
            "id": str(i),
            "label": r.label,
            "bbox": r.bbox,
            "color": color,
            "score": round(r.composite, 2),
            "opacity": 0.3 + (1.0 - r.composite) * 0.4,  # more opaque = worse
            "diagnosis": r.diagnosis or _region_diagnosis(r),
        })
    
    return {
        "regions": regions,
        "overall_score": round(score.overall, 2),
        "converged": score.converged,
    }


def _region_diagnosis(region) -> str:
    if region.composite < 0.3:
        return f"Low match ({region.composite:.2f})"
    if region.composite < 0.6:
        return f"Partial match ({region.composite:.2f})"
    return "Good match"
