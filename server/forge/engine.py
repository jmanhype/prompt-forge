"""Main convergence loop — the heart of Prompt Forge."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

from PIL import Image

from ..analyzer.florence import FlorenceAnalyzer, AnalysisResult
from ..analyzer.style import StyleExtractor
from ..analyzer.palette import extract_palette
from ..compiler.compiler import WorkflowCompiler
from ..connector.comfyui import ComfyUIConnector, GenerationResult, ProgressUpdate
from ..scorer.scorer import Scorer, ForgeScore
from ..mutator.mutator import Mutator
from ..mutator.llm import llm_mutate
from ..store.database import Database
from ..lora.detector import LoRADetector
from ..config import config
from ..patterns.registry import CalibrationRegistry
from ..patterns.observer import (
    ObserverManager, DiagnosisObserver, HeatmapObserver, MutationTriggerObserver
)


@dataclass
class Iteration:
    number: int
    prompt: dict
    images: list[str] = field(default_factory=list)
    score: Optional[ForgeScore] = None
    diagnosis: list[str] = field(default_factory=list)
    heatmap: dict = field(default_factory=dict)
    mutations: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class ForgeResult:
    session_id: str
    description: str
    iterations: list[Iteration] = field(default_factory=list)
    converged: bool = False
    final_score: float = 0.0
    total_duration_ms: int = 0


@dataclass
class ForgeEvent:
    type: str  # "analyzing" | "generating" | "scoring" | "mutating" | "iteration" | "converged" | "error"
    data: dict = field(default_factory=dict)


class ForgeEngine:
    """Orchestrates the full converge loop: analyze → compile → generate → score → mutate."""
    
    def __init__(self):
        self.analyzer = FlorenceAnalyzer(model_id=config.FLORENCE_MODEL)
        self.style_extractor = StyleExtractor(
            qwen_vl_enabled=config.QWEN_VL_ENABLED,
            qwen_vl_url=config.QWEN_VL_URL,
        )
        self.connector = ComfyUIConnector(url=config.COMFYUI_URL)
        self.scorer = Scorer(
            threshold=config.CONVERGENCE_THRESHOLD,
            model_name=config.CLIP_MODEL,
            pretrained=config.CLIP_PRETRAINED,
            calibration_registry=CalibrationRegistry(config.CALIBRATION_CONFIG)
        )
        self.mutator = Mutator(convergence_threshold=config.CONVERGENCE_THRESHOLD)
        self.db = Database(config.DB_PATH)
        self.lora_detector = LoRADetector(config.loras_dir)
        self._compiler: Optional[WorkflowCompiler] = None
    
    async def initialize(self):
        """Load models and probe ComfyUI capabilities."""
        from ..compiler.capability import probe_capabilities
        
        caps = await probe_capabilities(config.COMFYUI_URL)
        self._compiler = WorkflowCompiler(caps, config.TEMPLATES_DIR)
        # Pass API-detected LoRAs to detector
        self.lora_detector.scan(api_loras=caps.available_loras)
        self.db.initialize()
    
    async def run(
        self,
        description: str,
        image: Optional[Image.Image] = None,
        max_iterations: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> AsyncGenerator[ForgeEvent, None]:
        """Run the full forge loop, yielding events as they happen."""
        import sys
        print(f"\n[ENGINE] Starting forge with description: '{description}'", file=sys.stderr)
        
        session_id = str(uuid.uuid4())[:8]
        max_iter = max_iterations or config.MAX_ITERATIONS
        thresh = threshold or config.CONVERGENCE_THRESHOLD
        self.scorer.threshold = thresh
        
        # Set up observers for score feedback
        observer_manager = ObserverManager()
        diagnosis_observer = DiagnosisObserver(convergence_threshold=thresh)
        heatmap_observer = HeatmapObserver()
        mutation_observer = MutationTriggerObserver(convergence_threshold=thresh)
        
        observer_manager.register(diagnosis_observer)
        observer_manager.register(heatmap_observer)
        observer_manager.register(mutation_observer)
        
        iterations = []
        start_time = time.time()
        prev_score = None
        
        try:
            # ── PHASE 1: Analyze ──
            yield ForgeEvent(type="analyzing", data={"message": "Analyzing input..."})
            
            if image:
                analysis = self.analyzer.analyze(image)
                prompt_data = self.analyzer.to_ideogram_json(analysis)
                palette = extract_palette(image)
                prompt_data["palette"] = palette
            else:
                # Text mode — parse description into structured prompt
                prompt_data = await self._parse_text_description(description)
                print(f"[ENGINE] prompt_data after _parse_text_description: {prompt_data}", file=sys.stderr)
            
            # Inject style if available
            if image:
                style = self.style_extractor.extract(image)
                prompt_data["style_description"] = style
            
            # Auto-inject LoRAs
            lora_config = self.lora_detector.match(prompt_data)
            if lora_config:
                yield ForgeEvent(type="analyzing", data={
                    "message": f"Auto-injected LoRA: {lora_config['lora_name']}"
                })
            
            yield ForgeEvent(type="analyzing", data={
                "message": "Analysis complete",
                "elements": len(prompt_data.get("composition", {}).get("elements", [])),
            })
            
            # ── PHASE 2-6: Convergence Loop ──
            current_prompt = prompt_data
            print(f"[ENGINE] current_prompt before loop: {current_prompt}", file=sys.stderr)
            
            for i in range(max_iter):
                iteration = Iteration(number=i + 1, prompt=current_prompt)
                iter_start = time.time()
                
                # Compile
                print(f"[ENGINE] Iteration {i+1}: compiling with prompt: {current_prompt}", file=sys.stderr)
                yield ForgeEvent(type="generating", data={
                    "iteration": i + 1,
                    "max_iterations": max_iter,
                    "message": f"Compiling workflow (iteration {i+1}/{max_iter})..."
                })
                
                workflow = self._compiler.compile(current_prompt, lora_config)
                
                # Generate
                yield ForgeEvent(type="generating", data={
                    "iteration": i + 1,
                    "message": "Generating image..."
                })
                
                gen_result = await self._generate(workflow)
                if gen_result.error:
                    yield ForgeEvent(type="error", data={"message": gen_result.error})
                    return
                
                iteration.images = gen_result.images
                iteration.duration_ms = gen_result.duration_ms
                
                # Score
                yield ForgeEvent(type="scoring", data={
                    "iteration": i + 1,
                    "message": "Scoring output..."
                })
                
                if gen_result.images:
                    gen_image = Image.open(gen_result.images[0])
                    score = self.scorer.score(gen_image, prompt_data)
                else:
                    score = ForgeScore(overall=0.0, converged=False)
                
                iteration.score = score
                
                # Notify observers of score update
                observer_manager.notify_all(score, i + 1)
                
                # Get data from observers
                iteration.diagnosis = diagnosis_observer.get_messages()
                iteration.heatmap = heatmap_observer.get_heatmap_data()
                
                # Record
                iterations.append(iteration)
                
                yield ForgeEvent(type="iteration", data={
                    "number": i + 1,
                    "score": score.to_dict(),
                    "images": gen_result.images,
                    "diagnosis": iteration.diagnosis,
                    "heatmap": iteration.heatmap,
                    "duration_ms": iteration.duration_ms,
                })
                
                # Check convergence
                if score.converged:
                    break
                
                # Check plateau
                if self.mutator.should_use_llm(score, i + 1, prev_score):
                    yield ForgeEvent(type="mutating", data={
                        "message": "Rules plateaued — trying LLM-assisted mutation..."
                    })
                    llm_result = await llm_mutate(
                        current_prompt, score, score.diagnosis(),
                        llm_url=config.QWEN_VL_URL,
                        llm_model=config.QWEN_VL_MODEL,
                    )
                    if llm_result:
                        current_prompt = llm_result
                        iteration.mutations = ["LLM rewrote prompt based on scoring feedback"]
                        prev_score = score
                        continue
                
                # Mutate
                yield ForgeEvent(type="mutating", data={
                    "iteration": i + 1,
                    "message": "Applying targeted mutations..."
                })
                
                current_prompt, changes = self.mutator.mutate(current_prompt, score, lora_config=lora_config)
                print(f"[ENGINE] After mutation: {current_prompt}", file=sys.stderr)
                iteration.mutations = changes
                prev_score = score
            
            # ── PHASE 7: Save ──
            total_duration = int((time.time() - start_time) * 1000)
            final = iterations[-1] if iterations else None
            
            result = ForgeResult(
                session_id=session_id,
                description=description,
                iterations=iterations,
                converged=final.score.converged if final and final.score else False,
                final_score=final.score.overall if final and final.score else 0.0,
                total_duration_ms=total_duration,
            )
            
            # Save to composition library
            if final and final.score and final.score.overall > 0.5:
                self.db.save_composition(result)
            
            yield ForgeEvent(type="converged", data={
                "session_id": session_id,
                "iterations": len(iterations),
                "final_score": result.final_score,
                "converged": result.converged,
                "total_duration_ms": total_duration,
                "final_image": final.images[0] if final and final.images else None,
            })
        
        except Exception as e:
            print(f"[ENGINE] ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            yield ForgeEvent(type="error", data={"message": str(e)})
    
    async def _generate(self, workflow: dict) -> GenerationResult:
        """Run generation and return result."""
        config.ensure_dirs()
        last_result = GenerationResult(error="No output")
        
        async for event in self.connector.generate(workflow, config.OUTPUTS_DIR):
            if isinstance(event, GenerationResult):
                last_result = event
        return last_result
    
    async def _parse_text_description(self, description: str) -> dict:
        """Parse natural language into structured prompt with elements.
        
        Produces caption + composition with elements that have descriptions
        suitable for JSON prompts with bounding boxes.
        """
        # Extract the core subject from the description
        subject = description.strip()
        
        # If already detailed enough, use as-is
        if len(subject) > 100:
            caption = subject
            # Try to extract elements from the detailed description
            elements = self._extract_elements_from_text(caption)
        else:
            # Short descriptions need enrichment with CONCRETE details
            # Map common subjects to specific visual settings with TANGIBLE objects
            setting_map = {
                "wolfman": ("standing in a moonlit forest clearing with tall pine trees and silver light",
                           [{"desc": "Wolfman creature with detailed fur, glowing eyes, and sharp claws", "type": "subject"}]),
                "werewolf": ("standing in a moonlit forest clearing with tall pine trees and silver light",
                            [{"desc": "Werewolf with muscular form, detailed fur, and sharp features", "type": "subject"}]),
                "dog": ("sitting on green grass in a sunny suburban backyard with a wooden fence", 
                        [{"desc": "Golden retriever dog with detailed fur", "type": "subject"}]),
                "retriever": ("sitting on green grass in a sunny suburban backyard with a wooden fence",
                             [{"desc": "Golden retriever dog with detailed fur", "type": "subject"}]),
                "puppy": ("sitting on green grass in a sunny suburban backyard with a wooden fence",
                         [{"desc": "Cute puppy with playful expression", "type": "subject"}]),
                "cat": ("sitting on a wooden windowsill with lace curtains and sunlight streaming in",
                       [{"desc": "Cat with detailed whiskers and eyes", "type": "subject"}]),
                "kitten": ("sitting on a wooden windowsill with lace curtains and sunlight streaming in",
                          [{"desc": "Adorable kitten with fluffy fur", "type": "subject"}]),
                "car": ("parked on a tree-lined street next to red brick row houses",
                       [{"desc": "Car with polished surface and chrome details", "type": "subject"}]),
                "person": ("standing on a cobblestone sidewalk next to a cafe with potted plants",
                          [{"desc": "Person with natural pose and detailed clothing", "type": "subject"}]),
                "woman": ("standing on a cobblestone sidewalk next to a cafe with potted plants",
                         [{"desc": "Woman with natural pose and detailed clothing", "type": "subject"}]),
                "man": ("standing on a cobblestone sidewalk next to a cafe with potted plants",
                       [{"desc": "Man with natural pose and detailed clothing", "type": "subject"}]),
                "flower": ("in a terracotta pot on a rustic wooden garden table with sunflowers behind",
                          [{"desc": "Flower with delicate petals and vibrant colors", "type": "subject"}]),
                "rose": ("in a terracotta pot on a rustic wooden garden table with green leaves behind",
                        [{"desc": "Rose with layered petals and thorns", "type": "subject"}]),
                "building": ("a red brick warehouse with large iron-framed windows on a gravel lot",
                            [{"desc": "Building with architectural details and weathered facade", "type": "subject"}]),
                "house": ("a charming cottage with a stone path and blooming garden",
                         [{"desc": "House with detailed architecture and surroundings", "type": "subject"}]),
                "mountain": ("snow-capped peaks with tall pine trees and a crystal clear lake in front",
                            [{"desc": "Mountain peaks with snow and rocky texture", "type": "subject"}]),
                "ocean": ("rocky coastline with crashing blue waves and white seabirds flying overhead",
                         [{"desc": "Ocean waves with foam and spray", "type": "subject"}]),
                "beach": ("sandy shore with turquoise water and palm trees swaying in the breeze",
                         [{"desc": "Beach with golden sand and gentle waves", "type": "subject"}]),
                "food": ("on a white marble countertop with fresh basil leaves and olive oil bottle",
                        [{"desc": "Food with natural colors and textures", "type": "subject"}]),
                "bird": ("perched on a mossy oak branch with soft green forest bokeh behind",
                        [{"desc": "Bird with detailed feathers and sharp eyes", "type": "subject"}]),
                "tree": ("a large oak tree in a wildflower meadow with daisies and blue sky above",
                        [{"desc": "Tree with textured bark and detailed leaves", "type": "subject"}]),
                "horse": ("standing in a green paddock with white wooden fence and rolling hills behind",
                         [{"desc": "Horse with muscular form and flowing mane", "type": "subject"}]),
                "boat": ("moored at a wooden dock in a calm blue harbor with colorful houses on shore",
                        [{"desc": "Boat with detailed hull and rigging", "type": "subject"}]),
                "train": ("at a vintage railway platform with wrought iron columns and glass roof",
                         [{"desc": "Train with metallic details and steam", "type": "subject"}]),
                "robot": ("standing on a polished concrete floor with server racks and LED panels behind",
                         [{"desc": "Robot with metallic surface and glowing elements", "type": "subject"}]),
                "castle": ("on a hilltop surrounded by stone walls and medieval towers against a cloudy sky",
                          [{"desc": "Castle with detailed stonework and battlements", "type": "subject"}]),
                "bridge": ("a stone arch bridge over a river with autumn trees reflecting in the water",
                          [{"desc": "Bridge with architectural details and reflections", "type": "subject"}]),
            }
            
            # Try to match a setting
            setting = "in a bright outdoor scene with trees and natural light"
            elements = [{"desc": subject, "type": "subject"}]
            for key, (val, elems) in setting_map.items():
                if key in subject.lower():
                    setting = val
                    elements = elems
                    break
            
            caption = (
                f"{subject} {setting}. "
                f"Professional photography, sharp focus, natural daylight."
            )
        
        return {
            "caption": caption,
            "composition": {
                "background": f"Background and environment surrounding the subject.",
                "elements": elements,
            },
            "style_description": {},
            "negative_prompt": ["ugly", "blurry", "low quality", "watermark", "text"],
        }
    
    def _extract_elements_from_text(self, text: str) -> list[dict]:
        """Extract element descriptions from detailed text."""
        # Simple extraction: treat the whole text as one main subject
        # More sophisticated NLP parsing could be added later
        return [{"desc": text, "type": "subject"}]
