# Prompt Forge

> Closed-loop composition engine for local image generation.
> Describe what you want. It generates, grades itself, and iterates until it matches.

![Prompt Forge](https://img.shields.io/badge/Prompt%20Forge-v0.1-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What is this?

Every image-to-prompt tool is open-loop. You get a prompt, generate manually, evaluate by eye, tweak by feel. Repeat.

**Prompt Forge closes the loop:**

```
Description → Analyze → Generate → Score → Diagnose → Mutate → Regenerate → ... → Converged Result
```

It doesn't give you a prompt. It gives you a **result that matches your intent**, with full iteration log showing how it got there.

## Features

- **Closed-loop convergence** — generates, scores against your description, fixes failures, retries
- **Per-region scoring** — not one global CLIP number. Heatmap showing exactly what worked and what didn't
- **LoRA-aware** — scans your ComfyUI loras/ folder, auto-injects trigger words
- **ComfyUI native** — connects to your existing install. No bundled models. Works with any checkpoint
- **Graceful degradation** — GLIGEN → Attention Couple → IPAdapter → Mega-Prompt fallback
- **Composition library** — every successful run saved with FTS5 search
- **Dark terminal UI** — JetBrains Mono, minimal, no build step

## Quick Start

```bash
git clone https://github.com/jmanhype/prompt-forge.git
cd prompt-forge

pip install -r requirements.txt
cp .env.example .env
# Edit .env — set COMFYUI_URL and COMFYUI_ROOT

python scripts/check_setup.py
python -m server.main
```

Open http://localhost:7861

## How It Works

1. **Analyze** — Drop an image or type a description. Florence-2 detects objects, bboxes, captions
2. **Compile** — Structured JSON → ComfyUI workflow (selects best regional strategy for your install)
3. **Generate** — Queues workflow in your ComfyUI, retrieves output
4. **Score** — Florence re-detects output. Bbox IoU + DINOv2 per-region similarity
5. **Diagnose** — Heatmap shows exactly which elements failed and why
6. **Mutate** — Rule-based fixes targeting lowest-scoring regions only
7. **Repeat** — Until convergence (≥85% overall) or max iterations

## Requirements

- Python 3.10+
- ComfyUI running locally (any checkpoint)
- Optional: NVIDIA GPU (CPU works but slower)

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design.

```
Frontend (Vanilla JS) ←→ FastAPI Server
                              ├─ Analyzer (Florence-2 + CLIP)
                              ├─ Compiler (JSON → Workflow)
                              ├─ Connector (ComfyUI API)
                              ├─ Scorer (IoU + DINOv2)
                              ├─ Mutator (rules + LLM)
                              ├─ Forge Engine (convergence loop)
                              ├─ Store (SQLite + FTS5)
                              └─ LoRA Detector
                                    ↕
                              Your ComfyUI Instance
```

## Real-World Benchmark: Atomic Breath on Ideogram 4

The workflow in `workflows/templates/ideogram4_lora.json` was validated on June 10, 2026 using the Ektachrome Style LoRA v1 trained on Ideogram 4 FP8. We tested 3 approaches to generate Godzilla with atomic breath:

| Approach | Denoise | Result |
|----------|---------|--------|
| txt2img | 1.0 | ✅ All elements present |
| img2img | 0.85 | ✅ All elements, composition anchored to reference |
| No LoRA (control) | 1.0 | ✅ Atomic breath works natively |
| img2img | 0.65 | ❌ Too conservative, no beam generated |

**Key finding:** For adding new elements (effects, beams, objects), use txt2img or img2img @ 0.85+. For style transfer only, img2img @ 0.4-0.6 works.

Full benchmark: [docs/atomic-breath-benchmark.md](docs/atomic-breath-benchmark.md)

## Acknowledgments

- Inspired by [cocktailpeanut/image-to-prompt](https://github.com/cocktailpeanut/image-to-prompt) — showed that local image-to-prompt tools can go viral
- Architecture patterns from [ai2764/Camera-lab](https://github.com/ai2764/Camera-lab) — connector pattern, capability detection, graceful degradation
- Content engine design from [StraughterG-os](https://github.com/jmanhype/StraughterG-os) — scoring UI, dark aesthetic

## License

MIT. Do whatever you want with it.

---

*Built by [@StraughterG](https://x.com/StraughterG)*
