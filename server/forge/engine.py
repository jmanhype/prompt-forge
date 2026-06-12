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
from ..scorer.heatmap import generate_heatmap_data
from ..mutator.mutator import Mutator
from ..mutator.llm import llm_mutate
from ..store.database import Database
from ..lora.detector import LoRADetector
from ..config import config


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
        self.scorer = Scorer(threshold=config.CONVERGENCE_THRESHOLD)
        self.mutator = Mutator()
        self.db = Database(config.DB_PATH)
        self.lora_detector = LoRADetector(config.loras_dir)
        self._compiler: Optional[WorkflowCompiler] = None
    
    async def initialize(self):
        """Load models and probe ComfyUI capabilities."""
        from ..compiler.capability import probe_capabilities
        
        caps = await probe_capabilities(config.COMFYUI_URL)
        self._compiler = WorkflowCompiler(caps, config.TEMPLATES_DIR)
        self.lora_detector.scan()
        self.db.initialize()
    
    async def run(
        self,
        description: str,
        image: Optional[Image.Image] = None,
        max_iterations: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> AsyncGenerator[ForgeEvent, None]:
        """Run the full forge loop, yielding events as they happen."""
        session_id = str(uuid.uuid4())[:8]
        max_iter = max_iterations or config.MAX_ITERATIONS
        thresh = threshold or config.CONVERGENCE_THRESHOLD
        self.scorer.threshold = thresh
        
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
            
            for i in range(max_iter):
                iteration = Iteration(number=i + 1, prompt=current_prompt)
                iter_start = time.time()
                
                # Compile
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
                iteration.diagnosis = score.diagnosis()
                iteration.heatmap = generate_heatmap_data(score)
                
                # Record
                iterations.append(iteration)
                
                yield ForgeEvent(type="iteration", data={
                    "number": i + 1,
                    "score": score.to_dict(),
                    "images": gen_result.images,
                    "diagnosis": score.diagnosis(),
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
                
                current_prompt, changes = self.mutator.mutate(current_prompt, score)
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
        """Parse natural language into structured prompt JSON.
        Uses simple heuristics — full NLP parsing is TODO.
        """
        # Simple structured prompt from text
        return {
            "caption": description,
            "composition": {
                "background": description,
                "elements": [],
            },
            "style_description": {},
            "negative_prompt": ["ugly", "blurry", "low quality", "watermark", "text"],
        }
