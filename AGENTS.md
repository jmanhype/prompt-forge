# AGENTS.md — Prompt Forge

## Quick Start for Coding Agents

1. Install dependencies: `pip install -r requirements.txt`
2. Copy config: `cp .env.example .env` (edit COMFYUI_URL and COMFYUI_ROOT)
3. Run setup check: `python scripts/check_setup.py`
4. Start server: `python -m server.main`
5. Open: http://localhost:7861

## Architecture

- **server/analyzer/** — Florence-2 image analysis + CLIP style extraction
- **server/compiler/** — JSON prompt → ComfyUI workflow conversion
- **server/connector/** — ComfyUI REST + WebSocket client
- **server/scorer/** — Per-region composition scoring (bbox IoU + DINOv2)
- **server/mutator/** — Rule-based prompt mutations + optional LLM
- **server/forge/** — Convergence loop orchestrator
- **server/store/** — SQLite composition library + FTS5 search
- **server/lora/** — LoRA scanning + trigger word extraction
- **frontend/** — Vanilla JS (no build step, no npm)

## Key Patterns

- **Connector pattern**: Never vendor ComfyUI. Speak to it via REST/WS API.
- **Capability probe**: Query /object_info to detect available nodes at runtime.
- **Graceful degradation**: Value never hits zero. Mega-prompt fallback when no regional nodes.
- **Lazy model loading**: Florence-2 and DINOv2 load on first use, not at startup.
- **Template registry**: Workflow JSONs in workflows/templates/, deep-copied per use.

## Testing

```bash
python -m pytest tests/ -v
```

Tests run without GPU or ComfyUI (unit tests only).

## Dependency Manifest

| Package | Required | Purpose |
|---------|----------|---------|
| fastapi | Yes | HTTP server |
| uvicorn | Yes | ASGI server |
| websockets | Yes | ComfyUI progress |
| transformers | Yes | Florence-2 model |
| torch | Yes | ML inference |
| Pillow | Yes | Image handling |
| open-clip-torch | Yes | Style extraction |
| aiohttp | Yes | HTTP client |
| python-dotenv | Yes | Config |

## Without ComfyUI

The server starts without ComfyUI but generation will fail.
Analysis, scoring, and library browsing work standalone.
The frontend shows "ComfyUI: offline" status.
