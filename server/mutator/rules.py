"""Rule-based prompt mutations — enriched for Ideogram 4 based on calibration data.

Key finding: longer, more detailed prompts scored significantly higher in CLIP.
Mutations ADD detail rather than swapping words randomly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Mutation:
    description: str
    apply_fn: Callable[[dict], dict]

    def apply(self, prompt: dict) -> dict:
        return self.apply_fn(prompt)

    def describe(self) -> str:
        return self.description


# ── Enrichment libraries ──

# Film/medium detail phrases that Ideogram 4 responds to
FILM_DETAILS = [
    "visible luminance film grain throughout the frame",
    "cyan color shift in the shadow areas",
    "slight optical gate weave at the frame edges",
    "faded archival quality with subtle color desaturation",
    "natural lens vignetting at corners",
    "soft halation around highlights",
    "fine chromatic aberration at high contrast edges",
]

# Camera/lighting detail phrases
CAMERA_DETAILS = [
    "shot from a static tripod at human eye level",
    "deep depth of field with everything in sharp focus",
    "flat even lighting with no dramatic shadows",
    "natural ambient lighting from the environment",
    "slightly underexposed for rich shadow detail",
    "medium format camera perspective with natural compression",
]

# Setting/environment detail phrases
SETTING_DETAILS = [
    "empty and undisturbed environment",
    "signs of age and weathering on surfaces",
    "atmospheric haze in the background",
    "muted earth tones in the surroundings",
    "documented as if for scientific observation",
    "archival government documentation aesthetic",
]

# Subject enrichments — adjectives that make subjects more specific
SUBJECT_ENRICHMENTS = {
    "dog": ["sitting upright", "alert posture", "fur catching the light"],
    "cat": ["perched", "watchful eyes", "tail curled"],
    "car": ["polished surface reflecting light", "chrome details gleaming", "parked stationary"],
    "person": ["standing naturally", "relaxed posture", "facing the camera"],
    "building": ["weathered facade", "architectural details visible", "casting long shadows"],
    "flower": ["in full bloom", "delicate petals", "backlit by soft light"],
    "food": ["freshly prepared", "natural colors", "artfully arranged"],
    "default": ["clearly visible", "in sharp focus", "centered in frame"],
}


class RuleMutator:
    """Generates rule-based mutations calibrated for Ideogram 4."""

    def __init__(self):
        self._film_idx = 0
        self._camera_idx = 0
        self._setting_idx = 0

    def get_mutations(self, score, prompt: dict) -> list[Mutation]:
        """Determine what mutations to apply based on scoring results.
        
        Strategy: enrich the caption with more detail.
        Calibration showed longer prompts score 0.06+ higher in CLIP.
        """
        mutations = []
        caption = prompt.get("caption", "")

        # 1. If overall score is low, enrich the caption
        if score.overall < 0.4:
            mutations.extend(self._enrich_caption(caption, prompt))

        # 2. If style score is low, add film/camera details
        if score.style < 0.4:
            mutations.append(self._add_film_detail(prompt))
            mutations.append(self._add_camera_detail(prompt))

        # 3. If subject score is low, enrich element descriptions
        if score.subject < 0.4:
            mutations.append(self._enrich_subjects(prompt))

        # 4. If we have regions, enrich the worst one
        failing_regions = [r for r in score.regions if r.composite < 0.3]
        if failing_regions:
            worst = min(failing_regions, key=lambda r: r.composite)
            mutations.append(self._enrich_element(worst, prompt))

        # 5. Always add a setting detail (calibration showed these help)
        if len(caption.split()) < 40:
            mutations.append(self._add_setting_detail(prompt))

        if not mutations:
            mutations.append(self._add_film_detail(prompt))

        return mutations

    def _enrich_caption(self, caption: str, prompt: dict) -> list[Mutation]:
        """Add descriptive detail to the main caption."""
        mutations = []

        # Add medium description if missing
        if "film" not in caption.lower() and "photograph" not in caption.lower():
            mutations.append(Mutation(
                description="added medium description (archival photograph)",
                apply_fn=lambda p: _update_caption(p, lambda c: c + ", archival documentary photograph"),
            ))

        # Add lighting if missing
        if "lighting" not in caption.lower() and "light" not in caption.lower():
            mutations.append(Mutation(
                description="added lighting description (flat even lighting)",
                apply_fn=lambda p: _update_caption(p, lambda c: c + ", flat even lighting with no dramatic shadows"),
            ))

        return mutations

    def _add_film_detail(self, prompt: dict) -> Mutation:
        detail = FILM_DETAILS[self._film_idx % len(FILM_DETAILS)]
        self._film_idx += 1
        return Mutation(
            description=f"added film detail: '{detail[:40]}...'",
            apply_fn=lambda p: _update_caption(p, lambda c: c + f", {detail}"),
        )

    def _add_camera_detail(self, prompt: dict) -> Mutation:
        detail = CAMERA_DETAILS[self._camera_idx % len(CAMERA_DETAILS)]
        self._camera_idx += 1
        return Mutation(
            description=f"added camera detail: '{detail[:40]}...'",
            apply_fn=lambda p: _update_caption(p, lambda c: c + f", {detail}"),
        )

    def _add_setting_detail(self, prompt: dict) -> Mutation:
        detail = SETTING_DETAILS[self._setting_idx % len(SETTING_DETAILS)]
        self._setting_idx += 1
        return Mutation(
            description=f"added setting detail: '{detail[:40]}...'",
            apply_fn=lambda p: _update_caption(p, lambda c: c + f", {detail}"),
        )

    def _enrich_subjects(self, prompt: dict) -> Mutation:
        """Enrich element descriptions with specific adjectives."""
        def apply(p):
            elements = p.get("composition", {}).get("elements", [])
            enriched = 0
            for elem in elements:
                desc = elem.get("desc", "")
                if not desc or "clearly visible" in desc:
                    continue
                # Find matching enrichment
                key = "default"
                for k in SUBJECT_ENRICHMENTS:
                    if k != "default" and k in desc.lower():
                        key = k
                        break
                enrichments = SUBJECT_ENRICHMENTS[key]
                extra = ", ".join(enrichments)
                if extra not in desc:
                    elem["desc"] = f"{desc}, {extra}"
                    enriched += 1
                    if enriched >= 3:
                        break
            return p
        return Mutation(
            description="enriched subject descriptions with specific adjectives",
            apply_fn=apply,
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
