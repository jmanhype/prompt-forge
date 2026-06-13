"""Rule-based prompt mutations — using Strategy pattern for contextual enrichment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List

from ..patterns.factory import EnrichmentFactory
from ..patterns.strategy import EnrichmentStrategy, NoEnrichment


@dataclass
class Mutation:
    description: str
    apply_fn: Callable[[dict], dict]

    def apply(self, prompt: dict) -> dict:
        return self.apply_fn(prompt)

    def describe(self) -> str:
        return self.description


class RuleMutator:
    """Generates rule-based mutations using contextual enrichment strategies."""

    def __init__(self, convergence_threshold: float = 0.85):
        self.convergence_threshold = convergence_threshold
        # Derive thresholds from convergence threshold
        self.mutation_threshold = convergence_threshold * 0.7
        self.low_score_threshold = convergence_threshold * 0.6

    def get_mutations(self, score, prompt: dict, lora_config: Optional[dict] = None) -> list[Mutation]:
        """Determine what mutations to apply based on scoring results.
        
        Uses Strategy pattern to select context-appropriate enrichment.
        Only enriches if relevant to the prompt's aesthetic.
        """
        mutations = []
        caption = prompt.get("caption", "")

        # Select enrichment strategy based on prompt content and LoRA
        strategy = EnrichmentFactory.create(prompt, lora_config)
        
        # If strategy should apply, use it for enrichment
        if strategy.should_apply(prompt, lora_config) and not isinstance(strategy, NoEnrichment):
            # 1. If overall score is low, enrich caption with strategy
            if score.overall < self.mutation_threshold:
                mutations.extend(self._enrich_caption_with_strategy(caption, prompt, strategy))

            # 2. If style score is low, add style details from strategy
            if score.style < self.mutation_threshold:
                mutations.append(self._add_style_detail(prompt, strategy))

            # 3. If subject score is low, enrich element descriptions
            if score.subject < self.mutation_threshold:
                mutations.append(self._enrich_subjects_with_strategy(prompt, strategy))
        else:
            # No enrichment strategy applies - just return minimal mutations
            if score.overall < self.mutation_threshold:
                mutations.append(self._add_generic_detail(prompt))

        # 4. If we have regions, enrich the worst one
        failing_regions = [r for r in score.regions if r.composite < self.low_score_threshold]
        if failing_regions:
            worst = min(failing_regions, key=lambda r: r.composite)
            mutations.append(self._enrich_element(worst, prompt))

        if not mutations:
            mutations.append(self._add_generic_detail(prompt))

        return mutations

    def _enrich_caption_with_strategy(self, caption: str, prompt: dict, strategy: EnrichmentStrategy) -> list[Mutation]:
        """Enrich caption using the selected strategy."""
        enriched_caption = strategy.enrich_caption(caption)
        if enriched_caption != caption:
            return [Mutation(
                description=f"enriched caption with {strategy.get_name()} style",
                apply_fn=lambda p: _update_caption(p, lambda c: enriched_caption),
            )]
        return []

    def _add_style_detail(self, prompt: dict, strategy: EnrichmentStrategy) -> Mutation:
        """Add style details from strategy."""
        phrases = strategy.get_environment_phrases()
        if phrases:
            phrase = phrases[0]  # Use first phrase
            return Mutation(
                description=f"added {strategy.get_name()} style: '{phrase[:40]}...'",
                apply_fn=lambda p: _update_caption(p, lambda c: f"{c}, {phrase}"),
            )
        else:
            return self._add_generic_detail(prompt)

    def _enrich_subjects_with_strategy(self, prompt: dict, strategy: EnrichmentStrategy) -> Mutation:
        """Enrich element descriptions using strategy."""
        def apply(p):
            elements = p.get("composition", {}).get("elements", [])
            enriched_elements = strategy.enrich_elements(elements)
            if "composition" in p:
                p["composition"]["elements"] = enriched_elements
            return p
        
        return Mutation(
            description=f"enriched elements with {strategy.get_name()} details",
            apply_fn=apply,
        )

    def _add_generic_detail(self, prompt: dict) -> Mutation:
        """Add a generic detail when no strategy applies."""
        generic_details = [
            "highly detailed",
            "sharp focus",
            "professional quality",
            "carefully composed",
        ]
        # Pick one that's not already in the caption
        caption = prompt.get("caption", "")
        for detail in generic_details:
            if detail not in caption.lower():
                return Mutation(
                    description=f"added generic detail: '{detail}'",
                    apply_fn=lambda p: _update_caption(p, lambda c: f"{c}, {detail}"),
                )
        # All generic details already present
        return Mutation(
            description="no generic details to add",
            apply_fn=lambda p: p,
        )

    def _enrich_element(self, region, prompt: dict) -> Mutation:
        """Add detail to a specific failing element."""
        label = region.label
        def apply(p):
            elements = p.get("composition", {}).get("elements", [])
            for elem in elements:
                desc = elem.get("desc", "")
                if label.lower() in desc.lower():
                    if "highly detailed" not in desc:
                        elem["desc"] = f"{desc}, highly detailed, sharp focus, clearly defined"
                    break
            return p
        return Mutation(
            description=f"enriched '{label}' with detail emphasis",
            apply_fn=apply,
        )


def _update_caption(prompt: dict, fn) -> dict:
    """Apply a transformation function to the caption."""
    prompt["caption"] = fn(prompt.get("caption", ""))
    # Also update background to match
    if prompt.get("composition"):
        prompt["composition"]["background"] = prompt["caption"]
    return prompt
