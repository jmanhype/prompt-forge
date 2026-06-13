"""Mutation orchestrator — decides how to fix failed regions."""
from __future__ import annotations

from .rules import RuleMutator, Mutation
from typing import Optional


class Mutator:
    """Orchestrates prompt mutations based on scoring results."""

    def __init__(self):
        self.rule_mutator = RuleMutator()

    def mutate(self, prompt: dict, score) -> tuple[dict, list[str]]:
        """Apply targeted mutations to fix lowest-scoring regions.
        Returns (mutated_prompt, list_of_changes).
        
        CRITICAL: Never strip or replace the original caption.
        Only APPEND detail to it.
        """
        changes = []
        mutated = _deep_copy(prompt)
        
        # Store the original caption so we can protect it
        original_caption = mutated.get("caption", "")

        # Get mutations from rule engine based on score
        mutations = self.rule_mutator.get_mutations(score, mutated)

        # Apply up to 3 mutations per iteration (don't over-mutate)
        for mutation in mutations[:3]:
            mutated = mutation.apply(mutated)
            changes.append(mutation.describe())

        if not changes:
            changes = ["No mutations needed — all scores above threshold"]

        # PROTECT THE CAPTION: ensure the original subject is still present
        new_caption = mutated.get("caption", "")
        # Extract the core subject (first sentence of original)
        core_subject = original_caption.split(".")[0].strip()
        if core_subject and core_subject.lower() not in new_caption.lower():
            # The mutator destroyed the subject — restore it
            mutated["caption"] = original_caption
            changes.append(f"RESTORED original subject (was lost during mutation)")

        return mutated, changes

    def should_use_llm(self, score, iteration: int, prev_score=None) -> bool:
        """Determine if rule-based mutations have plateaued."""
        if iteration < 3:
            return False
        if prev_score and score.overall - prev_score.overall < 0.03:
            return True  # Less than 3% improvement = plateau
        return False


def _deep_copy(d: dict) -> dict:
    import json
    return json.loads(json.dumps(d))
