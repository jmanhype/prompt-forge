"""Style extraction — CLIP-interrogator vocab lookup + optional Qwen-VL."""
from __future__ import annotations

from PIL import Image
from typing import Optional


# Curated style vocabulary (subset of CLIP-interrogator)
STYLE_VOCABULARY = {
    "aesthetics": [
        "cinematic", "dramatic", "moody", "vibrant", "muted",
        "desolate", "industrial", "stark", "ethereal", "gritty",
        "vintage", "retro", "futuristic", "minimal", "ornate",
    ],
    "lighting": [
        "natural daylight", "golden hour", "overcast", "harsh shadows",
        "neon", "studio", "backlit", "volumetric", "flat",
        "bright direct overhead sunlight", "diffused",
    ],
    "medium": [
        "photograph", "oil painting", "watercolor", "digital art",
        "3D render", "pencil sketch", "film still", "anime",
    ],
    "art_style": [
        "ektachrome", "kodachrome", "film noir", "pop art",
        "surrealism", "photorealism", "impressionism", "cyberpunk",
    ],
}


class StyleExtractor:
    """Extract style_description from images using CLIP vocabulary matching."""
    
    def __init__(self, qwen_vl_enabled: bool = False, qwen_vl_url: str = ""):
        self.qwen_vl_enabled = qwen_vl_enabled
        self.qwen_vl_url = qwen_vl_url
        self._clip_model = None
        self._clip_preprocess = None
    
    def _load_clip(self):
        if self._clip_model is not None:
            return
        try:
            import open_clip
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        except ImportError:
            pass  # graceful degradation
    
    def extract(self, image: Image.Image) -> dict:
        """Extract style_description dict from image."""
        # Try Qwen-VL first if available
        if self.qwen_vl_enabled and self.qwen_vl_url:
            result = self._extract_qwen_vl(image)
            if result:
                return result
        
        # Fall back to CLIP vocabulary matching
        return self._extract_clip_vocab(image)
    
    def _extract_clip_vocab(self, image: Image.Image) -> dict:
        """Match image against style vocabulary using CLIP embeddings."""
        self._load_clip()
        
        if self._clip_model is None:
            return self._default_style()
        
        import torch
        img_tensor = self._clip_preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            image_features = self._clip_model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        style_desc = {}
        for field, candidates in STYLE_VOCABULARY.items():
            text_tokens = self._clip_tokenizer(candidates)
            with torch.no_grad():
                text_features = self._clip_model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).squeeze(0)
            best_idx = similarity.argmax().item()
            style_desc[field] = candidates[best_idx]
        
        return style_desc
    
    def _extract_qwen_vl(self, image: Image.Image) -> Optional[dict]:
        """Use Qwen-VL API for nuanced style extraction."""
        # TODO: implement API call to Qwen-VL endpoint
        return None
    
    def _default_style(self) -> dict:
        return {
            "aesthetics": "cinematic",
            "lighting": "natural daylight",
            "medium": "photograph",
            "art_style": "",
        }
