"""DINOv2 per-region visual similarity scoring."""
from __future__ import annotations

import torch
from PIL import Image
from typing import Optional


_dino_model = None


def _load_dino():
    """Lazy-load DINOv2 model."""
    global _dino_model
    if _dino_model is not None:
        return _dino_model
    
    try:
        _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        _dino_model.eval()
    except Exception:
        pass
    return _dino_model


@torch.no_grad()
def compute_dino_similarity(image_a: Image.Image, image_b: Image.Image) -> float:
    """Compute DINOv2 cosine similarity between two image crops."""
    model = _load_dino()
    if model is None:
        return 0.5  # fallback
    
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor_a = transform(image_a.convert("RGB")).unsqueeze(0)
    tensor_b = transform(image_b.convert("RGB")).unsqueeze(0)
    
    features_a = model(tensor_a)
    features_b = model(tensor_b)
    
    # Normalize and cosine similarity
    features_a = torch.nn.functional.normalize(features_a, dim=1)
    features_b = torch.nn.functional.normalize(features_b, dim=1)
    
    similarity = (features_a * features_b).sum(dim=1).item()
    return max(0.0, min(1.0, similarity))
