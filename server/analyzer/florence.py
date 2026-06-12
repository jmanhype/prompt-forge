"""Florence-2 scene analyzer — captioning, bbox detection, OCR."""
from __future__ import annotations

import torch
from PIL import Image
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Element:
    id: str
    type: str  # "obj" | "text"
    label: str
    description: str
    bbox: list[float]  # normalized [x1, y1, x2, y2] 0-1
    confidence: float = 0.0


@dataclass
class AnalysisResult:
    caption: str = ""
    background: str = ""
    palette: list[str] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)
    style_description: dict = field(default_factory=dict)
    raw_json: dict = field(default_factory=dict)
    model_used: str = ""


class FlorenceAnalyzer:
    """Wraps Florence-2 for scene analysis. Lazy-loads model on first use."""
    
    def __init__(self, model_id: str = "microsoft/Florence-2-base-ft"):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = None
    
    def _load_model(self):
        if self._model is not None:
            return
        
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="eager",
        ).to(self._device)
        self._model.eval()
    
    @torch.no_grad()
    def _run_task(self, image: Image.Image, task: str, text: str = "") -> str:
        self._load_model()
        # Florence-2 uses the task token as the text prompt
        prompt_text = text if text else task
        inputs = self._processor(text=prompt_text, images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        return self._processor.batch_decode(ids, skip_special_tokens=False)[0]
    
    def _parse_response(self, response: str, task: str, image_size: tuple) -> dict:
        self._load_model()
        parsed = self._processor.post_process_generation(
            response, task=task, image_size=image_size
        )
        return parsed.get(task, {})
    
    def analyze(self, image: Image.Image) -> AnalysisResult:
        """Run full analysis pipeline on an image."""
        w, h = image.size
        result = AnalysisResult(model_used=self.model_id)
        
        # 1. Detailed caption
        caption_resp = self._run_task(image, "<DETAILED_CAPTION>")
        caption_data = self._parse_response(caption_resp, "<DETAILED_CAPTION>", (w, h))
        result.caption = caption_data if isinstance(caption_data, str) else str(caption_data)
        
        # 2. Object detection (bboxes)
        od_resp = self._run_task(image, "<OD>")
        od_data = self._parse_response(od_resp, "<OD>", (w, h))
        
        bboxes = od_data.get("bboxes", [])
        labels = od_data.get("labels", [])  # Key is "labels", not "bboxes_labels"
        
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # Normalize to 0-1
            x1, y1, x2, y2 = bbox
            norm_bbox = [x1/w, y1/h, x2/w, y2/h]
            result.elements.append(Element(
                id=f"e{i+1}",
                type="obj",
                label=label.lower(),
                description=label.lower(),
                bbox=norm_bbox,
                confidence=0.9,  # Florence doesn't return confidence
            ))
        
        # 3. OCR (text regions)
        ocr_resp = self._run_task(image, "<OCR>")
        ocr_data = self._parse_response(ocr_resp, "<OCR>", (w, h))
        if isinstance(ocr_data, dict):
            texts = ocr_data.get("text", [])
            for i, text in enumerate(texts):
                if text.strip():
                    result.elements.append(Element(
                        id=f"t{i+1}",
                        type="text",
                        label="text",
                        description=text.strip(),
                        bbox=[0.0, 0.0, 0.1, 0.1],  # OCR doesn't give bboxes in base
                        confidence=0.7,
                    ))
        
        # 4. Background from caption
        if result.caption:
            result.background = result.caption
        
        return result
    
    def to_ideogram_json(self, analysis: AnalysisResult) -> dict:
        """Convert analysis to Ideogram 4 JSON prompt format.
        
        Ideogram 4 expects:
        - Bbox coordinates in 0-1000 range (not pixels)
        - Coordinate order: [y_min, x_min, y_max, x_max]
        - Structured schema: high_level_description, style_description, compositional_deconstruction
        """
        elements = []
        for elem in analysis.elements:
            entry = {
                "type": elem.type,
                "desc": elem.description,
            }
            if elem.type == "obj":
                # Convert normalized 0-1 bbox to Ideogram's 0-1000 coordinate system
                # Input: [x1, y1, x2, y2] normalized
                # Output: [y_min, x_min, y_max, x_max] in 0-1000
                x1, y1, x2, y2 = elem.bbox
                entry["bbox"] = [
                    round(y1 * 1000),  # y_min
                    round(x1 * 1000),  # x_min
                    round(y2 * 1000),  # y_max
                    round(x2 * 1000),  # x_max
                ]
            elif elem.type == "text":
                entry["text"] = elem.description
            elements.append(entry)
        
        prompt = {
            "high_level_description": analysis.caption,
            "style_description": analysis.style_description or {
                "aesthetics": "professional photography, sharp detail, natural tones",
                "lighting": "soft ambient lighting",
                "medium": "photograph",
                "color_palette": ["#FFFFFF", "#333333", "#666666", "#999999"],
            },
            "compositional_decomposition": {
                "background": analysis.background + " No text, no watermark, no logo, no clutter.",
                "elements": elements,
            },
        }
        
        return prompt
