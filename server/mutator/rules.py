"""Rule-based prompt mutations — deterministic, instant, no LLM needed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Mutation:
    description: str
    apply_fn: Callable[[dict], dict]
    
    def apply(self, prompt: dict) -> dict:
        return self.apply_fn(prompt)
    
    def describe(self) -> str:
        return self.description


class RuleMutator:
    """Generates rule-based mutations based on failure modes."""
    
    def get_mutations(self, region, prompt: dict) -> list[Mutation]:
        """Determine what mutations to apply for a failing region."""
        mutations = []
        
        if not region.present:
            mutations.append(self._increase_weight(region))
            mutations.append(self._add_negative(region))
        elif region.bbox_iou < 0.5:
            mutations.append(self._increase_weight(region))
            mutations.append(self._refine_description(region))
        elif region.dino_similarity < 0.6:
            mutations.append(self._add_detail(region))
        elif region.clip_score < 0.6:
            mutations.append(self._increase_weight(region))
        
        return mutations
    
    def _increase_weight(self, region) -> Mutation:
        label = region.label
        def apply(prompt):
            elements = prompt.get("composition", {}).get("elements", [])
            for elem in elements:
                if elem.get("label", "") == label or elem.get("desc", "").startswith(label):
                    current_weight = elem.get("weight", 1.0)
                    elem["weight"] = min(current_weight + 0.2, 2.0)
                    break
            return prompt
        return Mutation(
            description=f"increased '{label}' weight by 0.2",
            apply_fn=apply,
        )
    
    def _add_negative(self, region) -> Mutation:
        label = region.label
        def apply(prompt):
            negatives = prompt.setdefault("negative_prompt", [])
            neg = f"no {label}, missing {label}"
            if neg not in negatives:
                negatives.append(neg)
            return prompt
        return Mutation(
            description=f"added 'no {label}' to negative prompt",
            apply_fn=apply,
        )
    
    def _refine_description(self, region) -> Mutation:
        label = region.label
        def apply(prompt):
            elements = prompt.get("composition", {}).get("elements", [])
            for elem in elements:
                if elem.get("label", "") == label:
                    # Add positional context
                    bbox = elem.get("bbox", [0, 0, 0, 0])
                    cx = (bbox[0] + bbox[2]) / 2 if bbox else 0.5
                    position = "center" if 0.3 < cx < 0.7 else ("left" if cx <= 0.3 else "right")
                    elem["desc"] = f"{elem['desc']}, positioned on the {position}"
                    break
            return prompt
        return Mutation(
            description=f"added positional context to '{label}'",
            apply_fn=apply,
        )
    
    def _add_detail(self, region) -> Mutation:
        label = region.label
        def apply(prompt):
            elements = prompt.get("composition", {}).get("elements", [])
            for elem in elements:
                if elem.get("label", "") == label:
                    elem["desc"] = f"{elem['desc']}, highly detailed, sharp focus"
                    break
            return prompt
        return Mutation(
            description=f"added detail emphasis to '{label}'",
            apply_fn=apply,
        )
