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
        """
        changes = []
        mutated = _deep_copy(prompt)

        # Get mutations from rule engine based on score
        mutations = self.rule_mutator.get_mutations(score, mutated)

        # Apply up to 3 mutations per iteration (don't over-mutate)
        for mutation in mutations[:3]:
            mutated = mutation.apply(mutated)
            changes.append(mutation.describe())

        if not changes:
            changes = ["No mutations needed — all scores above threshold"]

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
