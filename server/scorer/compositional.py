"""Bbox IoU scoring — spatial accuracy per element."""
from __future__ import annotations


def compute_bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection over Union for two normalized bboxes [x1,y1,x2,y2]."""
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    
    # Intersection
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union
