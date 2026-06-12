# PROMPT FORGE — CONTENT PACKAGE
# Created: June 12, 2026
# Brand: BatmanOsama / StraughterG
# Product: Prompt Forge (github.com/jmanhype/prompt-forge)

---

## TWITTER/X THREAD (for StraughterG-os)

**Tweet 1 (Hook):**
I built something that doesn't exist yet.

You type what you want. The AI generates an image. Then another AI looks at it and says "that's not right — the dog should be bigger and the lighting is wrong." Then it tries again. And again. Until it's right.

No manual prompting. No guessing.

🧵

**Tweet 2 (Problem):**
Every image gen tool works the same way:

1. You write a prompt
2. You get an image
3. It's wrong
4. You rewrite the prompt
5. Repeat 20 times

That's not automation. That's you doing the work while the AI watches.

**Tweet 3 (Solution):**
Prompt Forge closes the loop:

Describe → Generate → Analyze (Florence-2) → Score (CLIP) → Mutate → Regenerate

It keeps going until the image actually matches what you described. Or until you tell it to stop.

No more "close enough." No more 20 prompt rewrites.

**Tweet 4 (How it works):**
Under the hood:

• Florence-2 captions + detects objects in the output
• CLIP scores how well the image matches your description
• If score < threshold → rule-based mutator enriches the prompt
• Adds film grain, camera angles, lighting details, environment context
• Regenerates with the improved prompt
• Repeats until convergence

**Tweet 5 (The mutation part):**
The mutator doesn't just swap random words.

It adds the kind of detail that makes images better:
- "visible luminance film grain throughout the frame"
- "shot from a static tripod at human eye level"
- "deep depth of field with everything in sharp focus"
- "empty and undisturbed environment"

These aren't random. They're calibrated from testing what actually improves CLIP scores.

**Tweet 6 (LoRA support):**
It also auto-detects your LoRAs.

Type "ektachrome" anywhere in your prompt → it finds your Ektachrome Style LoRA, injects it at the right strength (0.8 model, 0.8 clip), and generates with it.

No manual LoRA selection. No strength guessing.

**Tweet 7 (Image mode):**
Or drop a reference image.

Florence-2 analyzes it — detects objects, draws bounding boxes, captions the scene — then generates a new image matching that layout with your style.

Reference → Analysis → Generation → Convergence.

**Tweet 8 (Stack):**
Built on:
• FastAPI backend (Python)
• Florence-2-base-ft (Microsoft's vision model)
• CLIP ViT-B-32 (OpenAI's image-text model)
• ComfyUI (generation engine)
• Vanilla JS frontend (dark terminal aesthetic, no React bloat)

100% local. No API keys needed for core functionality.

**Tweet 9 (Comparison):**
cocktailpeanut's image-to-prompt: image → JSON (no generation)
A1111/Forge: prompt → image (no feedback loop)
Midjourney: prompt → image (no control)

Prompt Forge: prompt → image → analysis → mutation → better image → repeat

The feedback loop is the whole point.

**Tweet 10 (CTA):**
It's open source. 13 commits. Docker install or bash script.

https://github.com/jmanhype/prompt-forge

Requirements: ComfyUI + GPU (24GB VRAM recommended)

Star it if closed-loop generation is something you've been waiting for.

---

## REDDIT POST (r/comfyui)

**Title:** I built a closed-loop generation tool — describe what you want, it keeps regenerating until it matches

**Body:**

Been frustrated with the prompt → generate → "nah" → rewrite prompt → generate cycle. So I built something that closes the loop.

**Prompt Forge** — you describe what you want, it generates an image, then Florence-2 analyzes the output and CLIP scores how well it matches your description. If the score is too low, it mutates the prompt (adds detail, adjusts composition descriptions) and tries again. Repeats until converged.

**How it works:**

1. Type a description (or drop a reference image)
2. It compiles a ComfyUI workflow and generates
3. Florence-2 captions the output, detects objects, runs OCR
4. CLIP scores image-vs-description (calibrated with 65 real measurements)
5. If score < threshold → mutator enriches the prompt with film/camera/setting details
6. Regenerates → repeats up to 5 iterations

**What the mutator actually does:**

Instead of random word swaps, it adds the kind of descriptive detail that consistently improves image quality:
- Film characteristics (grain, cyan shift, gate weave)
- Camera details (tripod, depth of field, lighting)
- Environment context (empty spaces, weathering, haze)
- Subject enrichments (specific adjectives per object type)

**LoRA auto-injection:**

Scans your ComfyUI LoRAs via API. Type "ektachrome" in your prompt → it finds `ektachrome_style_v1.safetensors`, injects at model:0.8/clip:0.8 automatically. Works with any LoRA that has trigger words.

**The scoring calibration:**

I ran 65 CLIP measurements across 5 images and 13 prompts at different match levels:
- Exact match prompts: raw CLIP 0.13-0.33
- Close match: 0.19-0.27
- Partial match: 0.09-0.23
- Wrong match: 0.02-0.19

Built a piecewise normalization from this. Threshold defaults to 0.55.

**Browser UI:**

Dark terminal aesthetic. No React, no build step. Vanilla JS. Shows real-time scores (overall, composition, style, subject), diagnosis messages, and iteration history.

**Requirements:**
- ComfyUI running (I use it with Ideogram 4 + Ektachrome LoRA)
- Python 3.10+
- Florence-2 auto-downloads on first run

**Install:**
```
git clone https://github.com/jmanhype/prompt-forge
cd prompt-forge
bash install.sh
```

Or Docker: `docker-compose up`

Open source, 13 commits, no API keys needed for core. Would love feedback on the approach — especially if anyone has ideas for better scoring than CLIP or smarter mutation strategies.

GitHub: https://github.com/jmanhype/prompt-forge

---

## REDDIT POST (r/StableDiffusion) — SHORTER VERSION

**Title:** Prompt Forge — closed-loop image generation that keeps regenerating until it matches your description

**Body:**

Built a tool that closes the feedback loop in image generation:

1. You describe what you want (text or reference image)
2. It generates via ComfyUI
3. Florence-2 analyzes the output, CLIP scores the match
4. If too low → mutator enriches the prompt (adds film grain, camera details, environment context)
5. Regenerates → repeats until converged

Auto-detects your LoRAs from ComfyUI's API. Type a trigger word and it injects automatically.

The mutation isn't random — it adds descriptive detail that consistently improves CLIP scores (calibrated from 65 measurements). Not random word swaps.

Dark terminal UI, vanilla JS, no build step. Docker or bash install.

https://github.com/jmanhype/prompt-forge

---

## HUGGING FACE POST / DISCUSSION

**Title:** Prompt Forge — Closed-loop composition engine for ComfyUI + LoRA

**Body:**

Sharing a tool I built for automating the prompt refinement cycle:

**Prompt Forge** generates an image, scores it against your description using CLIP, and if it doesn't match — mutates the prompt and regenerates. Repeats until the output actually matches what you described.

Key features:
- Florence-2 for scene analysis (captioning, object detection, OCR)
- CLIP ViT-B-32 for scoring (piecewise normalization calibrated on real data)
- Rule-based mutator (adds film/camera/setting details instead of random word swaps)
- Auto LoRA detection via ComfyUI API
- Reference image mode (drop image → Florence analyzes → generates matching layout)
- Ideogram 4 + LoRA workflow built-in

Works with any ComfyUI setup. Designed for LoRA workflows — type a trigger word and it auto-injects the matching LoRA.

https://github.com/jmanhype/prompt-forge

---

## YOUTUBE SCRIPT OUTLINE (for StraughterG-os)

**Title:** "I Built an AI That Criticizes Its Own Images Until They're Perfect"

**Hook (0:00-0:15):**
"What if the AI didn't just generate your image — what if it looked at what it made, decided it wasn't good enough, and tried again? That's what I built. It's called Prompt Forge."

**Problem (0:15-0:45):**
Show the manual prompting cycle. Type prompt → get image → "nah" → rewrite → get image → "still wrong" → rewrite again. "This isn't automation. This is you doing the work."

**Solution (0:45-1:30):**
Show the loop: Describe → Generate → Florence-2 analyzes → CLIP scores → Mutator enriches → Regenerate. "It keeps going until the score is high enough. No human in the loop."

**Demo (1:30-3:00):**
Show 3 examples:
1. Detailed prompt → converges in 1 iteration
2. Vague prompt ("a cat") → 3 iterations, prompt gets enriched each time
3. Reference image → Florence-2 detects objects → generates matching layout

**Technical (3:00-4:00):**
Show the scoring calibration data. Show the mutation rules. Show the LoRA auto-injection.

**CTA (4:00-4:15):**
"Open source. Docker or one-line install. Link in description."
