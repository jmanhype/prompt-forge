# PROMPT FORGE — System Architecture

> Closed-loop composition engine for local image generation.
> Describe what you want. It generates, grades itself, and iterates until it matches.

**Version:** 0.1.0-design
**Author:** StraughterG
**Date:** 2026-06-12

---

## 1. Problem Statement

Every image-to-prompt tool in existence is open-loop:

```
User → Prompt Tool → JSON/Text Prompt → ??? (user manually generates, evaluates, tweaks)
```

The user gets a prompt and prays it works. Feedback is entirely human. Iteration is manual.

**Prompt Forge closes the loop:**

```
User → Description → Generate → Score → Diagnose → Mutate → Regenerate → Score → ... → Converged Result
```

The system doesn't give you a prompt. It gives you a **result that matches your intent**, with full iteration log showing how it got there.

---

## 2. Core Principles

1. **Closed loop over open loop** — generation without evaluation is guessing
2. **Per-region scoring over global scoring** — one CLIP number lies; region heatmaps reveal truth
3. **Connector over vendor** — speak to user's ComfyUI, never bundle models
4. **Graceful degradation** — value never hits zero; scales with user's hardware
5. **Composition as asset** — every successful run becomes reusable knowledge
6. **LoRA-aware** — scan installed LoRAs, inject trigger words automatically

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Vanilla JS)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Canvas   │  │ Iteration│  │  Heatmap │  │  Composition │   │
│  │  Editor   │  │ Timeline │  │  Overlay │  │   Library    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                     ↕ WebSocket + REST                          │
├─────────────────────────────────────────────────────────────────┤
│                     FASTAPI SERVER                              │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │  ANALYZER    │    │  COMPILER   │    │   CONNECTOR      │    │
│  │  Florence-2  │───→│  JSON → WF  │───→│  ComfyUI API    │    │
│  │  + CLIP-vocab│    │  capability │    │  queue + WS     │    │
│  │  + palette   │    │  aware      │    │  + progress     │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                                      │               │
│         ↓                                      ↓               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │  SCORER      │    │  MUTATOR    │    │   LORA DETECT   │    │
│  │  Florence    │    │  rule-based │    │  scan loras/    │    │
│  │  re-detect   │    │  + LLM opt  │    │  extract trig.  │    │
│  │  + DINOv2    │    │  targeted   │    │  auto-inject    │    │
│  │  per-region  │    │  mutations  │    │                 │    │
│  └─────────────┘    └─────────────┘    └─────────────────┘    │
│         │                  │                                    │
│         └──────┬───────────┘                                    │
│                ↓                                                │
│         ┌─────────────┐    ┌─────────────┐                     │
│         │  FORGE       │    │   STORE     │                     │
│         │  ENGINE      │    │  SQLite     │                     │
│         │  convergence │    │  + FTS5     │                     │
│         │  loop        │    │  + images   │                     │
│         └─────────────┘    └─────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────────────────────────┐
│              USER'S COMFYUI INSTANCE (external)                 │
│  /object_info  /prompt  /queue  ws://history                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 Analyzer

**Purpose:** Turn an image or text description into structured composition data.

**Modes:**
- **Image mode:** Florence-2-base detects objects, bboxes, captions, OCR regions
- **Text mode:** LLM (Qwen/GLM) parses natural language into structured JSON

**Florence-2 Ladder:**

| Model | Size | VRAM | Speed | Use Case |
|-------|------|------|-------|----------|
| Florence-2-base-ft | 230MB | CPU OK | ~2s | Default — scene caption + bbox |
| Florence-2-large-ft | 770MB | ~4GB | ~4s | Higher accuracy, complex scenes |
| Qwen2.5-VL-7B (opt) | 14GB | 3090 | ~8s | Style extraction, complex prompts |

**Style Extraction Pipeline:**
1. CLIP-interrogator vocab lookup (zero-LLM, instant)
2. Optional: Qwen-VL for nuanced style fields
3. Merge into Ideogram JSON `style_description`

**Output Schema:**
```json
{
  "caption": "A woman standing in front of a brutalist building...",
  "background": "Overcast sky, concrete plaza...",
  "palette": ["#8B8680", "#A0937D", "#C4B7A6"],
  "elements": [
    {
      "id": "e1",
      "type": "obj",
      "label": "woman",
      "description": "Woman holding clipboard, facing camera",
      "bbox": [0.3, 0.2, 0.5, 0.9],
      "confidence": 0.94
    },
    {
      "id": "e2",
      "type": "obj",
      "label": "building",
      "description": "Brutalist concrete building with geometric repetition",
      "bbox": [0.0, 0.0, 1.0, 0.6],
      "confidence": 0.91
    }
  ],
  "style_description": {
    "aesthetics": "Desolate, industrial, stark",
    "lighting": "Overcast, diffused, flat",
    "medium": "Photograph",
    "art_style": "ektachrome"
  }
}
```

### 4.2 Compiler

**Purpose:** Convert structured composition JSON into a ComfyUI workflow.

**Capability Detection:**
On startup, query ComfyUI `/object_info` and build a capability map:

```python
capabilities = {
    "gligen": "GLIGENLoader" in node_info,      # SD1.5 native bbox
    "attention_couple": "AttentionCouple" in node_info,  # SDXL regional
    "ipadapter_region": "IPAdapterRegional" in node_info,  # IPAdapter regions
    "flux_guidance": "FluxGuidance" in node_info,  # Flux native
    "regional_condition": "RegionalConditioning" in node_info,
}
```

**Template Selection (priority order):**

| Priority | Strategy | Requires | Models |
|----------|----------|----------|--------|
| 1 | GLIGEN | GLIGENLoader | SD1.5 |
| 2 | Attention Couple | AttentionCouple | SDXL |
| 3 | IPAdapter Regional | IPAdapterRegional | SDXL/Flux |
| 4 | Mega-Prompt | Nothing | Any |

**Mega-Prompt Fallback:**
When no regional nodes are available, compile JSON into a structured long prompt:
```
[woman:1.3] standing in front of [brutalist concrete building:1.2], 
holding clipboard, facing camera, overcast sky, diffused lighting.
Style: ektachrome, vintage 1960s Kodak Ektachrome film, faded colors, 
visible grain, desolate industrial.
```

This works well with Flux, SDXL (with prompt scheduling), and Qwen-Image.

**LoRA Injection:**
- Scan user's `loras/` folder on startup
- Parse filenames for trigger words (e.g., `ektachrome_style_v1` → trigger: "ektachrome")
- Match style_description keywords to installed LoRAs
- Auto-inject LoRA loader nodes + trigger words into prompt

### 4.3 Connector

**Purpose:** Communicate with user's ComfyUI instance.

**Pattern (from Camera Lab):**
- `COMFYUI_URL` in .env (default: http://127.0.0.1:8188)
- Never vendor ComfyUI code or models
- `check_setup.py` validates connectivity + model availability
- Graceful degradation banner when ComfyUI unreachable

**API Endpoints Used:**
```
GET  /object_info          → capability detection
POST /prompt               → queue workflow
GET  /queue                → check queue status
WS   /ws                   → real-time progress
GET  /view?filename=X      → retrieve generated image
GET  /history/{prompt_id}  → get outputs after completion
```

**Generation Flow:**
```python
async def generate(workflow: dict) -> GenerationResult:
    # 1. Queue the prompt
    resp = await client.post(f"{url}/prompt", json={"prompt": workflow})
    prompt_id = resp["prompt_id"]
    
    # 2. Connect WebSocket for progress
    async with websockets.connect(ws_url) as ws:
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "executed" and msg["data"]["prompt_id"] == prompt_id:
                break
            yield_progress(msg)  # forward to frontend
    
    # 3. Retrieve outputs
    history = await client.get(f"{url}/history/{prompt_id}")
    images = extract_images(history[prompt_id])
    return GenerationResult(images=images, prompt_id=prompt_id)
```

### 4.4 Scorer

**Purpose:** Evaluate generated output against user's intent. Per-region, not global.

**Scoring Pipeline:**

```
Generated Image
       │
       ↓
  Florence-2 Re-detect
  (same model as analyzer for consistency)
       │
       ├──→ Bbox IoU per element (spatial accuracy)
       │    target bbox vs detected bbox
       │    IoU > 0.5 = pass, IoU > 0.7 = good
       │
       ├──→ DINOv2 Per-Region Similarity (visual match)
       │    crop each target region from source + output
       │    compute cosine similarity of embeddings
       │    > 0.8 = match, > 0.6 = partial
       │
       ├──→ CLIP Global Score (overall vibe check)
       │    text prompt vs image embedding
       │    secondary signal only
       │
       └──→ Element Presence Check
            did all requested elements appear?
            missing elements = hard failure on that region
```

**Composite Score:**
```python
def composite_score(regions: list[RegionScore]) -> ForgeScore:
    return ForgeScore(
        composition=weighted_mean([r.bbox_iou for r in regions]),
        style=clip_global_score,
        subject=weighted_mean([r.dino_sim for r in regions if r.is_primary]),
        overall=0.4 * composition + 0.3 * style + 0.3 * subject,
        regions=regions,  # per-region detail for heatmap
        converged=overall >= threshold,  # default 0.85
    )
```

**Why DINOv2 over CLIP for scoring:**
- CLIP saturates: "both images contain a dog" → high score even if composition is wrong
- DINOv2 captures structural/visual similarity: same pose, same framing, same lighting
- DINOv2-vit-base is 330MB, runs on CPU in ~0.5s per crop
- CLIP kept as secondary "vibe check" only

### 4.5 Mutator

**Purpose:** Given scoring results, modify the prompt to fix what failed.

**Rule-Based Mutations (primary, no LLM needed):**

| Failure Mode | Mutation |
|---|---|
| Low bbox IoU (element misplaced) | Shift bbox toward detected position, add negative prompt for wrong area |
| Missing element | Increase element weight [element:1.5], add to negative: "no [element]" |
| Wrong style | Strengthen style_description keywords, add style LoRA if available |
| Low DINOv2 similarity | Add more descriptive detail to element description |
| Over-triggered LoRA | Reduce LoRA strength by 0.1 |
| Element too large/small | Adjust bbox size toward detected scale |

**Targeted Mutation:**
Only mutate the lowest-scoring regions. Don't touch what works.

```python
def mutate(prompt: ForgePrompt, score: ForgeScore) -> ForgePrompt:
    # Sort regions by score ascending (worst first)
    failures = sorted(score.regions, key=lambda r: r.composite)[:3]
    
    mutated = prompt.copy()
    for region in failures:
        mutation = select_mutation(region.failure_mode)
        mutated.apply(region.id, mutation)
    
    return mutated
```

**LLM-Assisted Mutations (optional):**
When rule-based mutations plateau (3+ iterations with <5% improvement):
- Send current prompt + score + failure descriptions to LLM
- Ask for 3 alternative prompt rewrites targeting specific failures
- Pick the one with highest predicted improvement

### 4.6 Forge Engine

**Purpose:** Orchestrate the convergence loop.

**Session Lifecycle:**
```python
class ForgeSession:
    async def run(self, description: str, max_iterations: int = 5) -> ForgeResult:
        # 1. Analyze input
        analysis = await self.analyzer.analyze(description)
        prompt = self.compiler.build_prompt(analysis)
        prompt = self.lora_detector.inject(prompt)
        
        iterations = []
        for i in range(max_iterations):
            # 2. Compile to workflow
            workflow = self.compiler.compile(prompt, self.capabilities)
            
            # 3. Generate
            result = await self.connector.generate(workflow)
            
            # 4. Score
            score = await self.scorer.score(result.images, analysis)
            
            # 5. Record iteration
            iteration = Iteration(
                number=i+1,
                prompt=prompt,
                images=result.images,
                score=score,
                diagnosis=score.diagnosis(),
            )
            iterations.append(iteration)
            yield iteration  # stream to frontend
            
            # 6. Check convergence
            if score.converged:
                break
            
            # 7. Mutate
            if score.improvement_rate < 0.05 and i >= 2:
                prompt = await self.mutator.mutate_llm(prompt, score)
            else:
                prompt = self.mutator.mutate(prompt, score)
        
        # 8. Save to composition library
        await self.store.save_composition(iterations[-1], description)
        
        return ForgeResult(iterations=iterations, final=iterations[-1])
```

**Convergence Criteria:**
- Overall score >= 0.85 (configurable threshold)
- No region below 0.6
- Max 5 iterations (configurable, prevents infinite loops)
- Improvement rate < 2% for 2 consecutive iterations (plateau detection)

### 4.7 Store

**Purpose:** Persist composition library, iteration logs, user preferences.

**SQLite Schema:**
```sql
-- Composition library
CREATE TABLE compositions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    style TEXT,
    final_prompt JSON,
    final_score REAL,
    iteration_count INTEGER,
    tags TEXT,  -- comma-separated for FTS
    image_path TEXT
);

-- FTS5 for searching compositions by description/style
CREATE VIRTUAL TABLE compositions_fts USING fts5(
    description, style, tags, content=compositions
);

-- Full iteration logs (taste model training data)
CREATE TABLE iterations (
    id TEXT PRIMARY KEY,
    composition_id TEXT REFERENCES compositions(id),
    iteration_number INTEGER,
    prompt JSON,
    score JSON,
    diagnosis JSON,
    image_path TEXT,
    duration_ms INTEGER
);

-- User preferences
CREATE TABLE preferences (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- LoRA registry (cached scan results)
CREATE TABLE loras (
    filename TEXT PRIMARY KEY,
    trigger_words TEXT,
    style_tags TEXT,
    last_scanned TIMESTAMP
);
```

**Taste Model Intake:**
Every iteration (including failed ones) is logged with:
- Input prompt (JSON)
- Generated image (path)
- Per-region scores
- Composite score
- Diagnosis
- Whether this was the final/selected result

This is labeled preference data: "this prompt produced this result with this score." Over time, this trains a taste model that predicts scores from prompts alone — enabling pre-filtering before generation.

### 4.8 LoRA Detector

**Purpose:** Scan user's installed LoRAs, extract trigger words, match to style requests.

**Scanning:**
```python
def scan_loras(loras_dir: str) -> list[LoRA]:
    loras = []
    for f in Path(loras_dir).glob("*.safetensors"):
        # Extract trigger from filename convention
        trigger = extract_trigger_from_name(f.stem)
        # Try reading metadata from safetensors header
        metadata = read_safetensors_metadata(f)
        loras.append(LoRA(
            filename=f.name,
            trigger_words=metadata.get("trigger_words", [trigger]),
            style_tags=metadata.get("tags", []),
            path=str(f),
        ))
    return loras
```

**Auto-Injection:**
When user's description mentions a style that matches an installed LoRA:
1. Add LoRA loader node to workflow
2. Insert trigger word into style_description
3. Set default strength (0.8 for style LoRAs, 0.6 for character LoRAs)

---

## 5. API Design

### REST Endpoints

```
POST /api/analyze
  Body: multipart/form-data { file: image } OR { text: description }
  Response: { elements, style, palette, caption }

POST /api/forge
  Body: { description, options: { max_iterations, threshold, model } }
  Response: SSE stream of iterations

GET  /api/forge/{session_id}
  Response: { iterations[], status, final_score }

GET  /api/library
  Query: ?q=brutalist&style=ektachrome
  Response: { compositions[] }

GET  /api/loras
  Response: { loras: [{ filename, triggers, tags }] }

POST /api/rescan-loras
  Response: { count, loras[] }

GET  /api/capabilities
  Response: { comfyui: bool, nodes: {}, models: [] }

WS   /ws/forge/{session_id}
  Messages: { type: "iteration"|"progress"|"converged"|"error", data: {} }
```

### WebSocket Message Types

```json
{"type": "connected", "data": {"session_id": "abc123"}}
{"type": "analyzing", "data": {"status": "Florence-2 detecting..."}}
{"type": "generating", "data": {"iteration": 1, "progress": 45}}
{"type": "scoring", "data": {"iteration": 1}}
{"type": "iteration", "data": {"number": 1, "score": {...}, "image_url": "...", "diagnosis": [...]}}
{"type": "mutating", "data": {"changes": ["shifted woman bbox right", "added 'raw concrete' to building"]}}
{"type": "converged", "data": {"iterations": 3, "final_score": 0.89, "final_image": "..."}}
{"type": "error", "data": {"message": "ComfyUI unreachable", "fix": "Start ComfyUI at http://127.0.0.1:8188"}}
```

---

## 6. Frontend Architecture

**Stack:** Vanilla JS (ES Modules) + CSS Grid + Canvas API

**No framework.** No build step. No npm install. Open index.html.

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER: Logo + New Forge + Library + Settings              │
├────────────┬─────────────────────────────────┬───────────────┤
│  INPUT     │        CANVAS                   │   SCORES      │
│  PANEL     │                                 │   PANEL       │
│            │  ┌─────────────────────────┐    │               │
│  Text      │  │                         │    │  Overall: 89% │
│  description│  │   Generated Image       │    │  ████████░░   │
│  or        │  │   with bbox overlay     │    │               │
│  image     │  │   and heatmap colors    │    │  Composition  │
│  drop zone │  │                         │    │  Style        │
│            │  └─────────────────────────┘    │  Subject      │
│  Elements  │                                 │               │
│  list with │  ITERATION TIMELINE             │  Region       │
│  bbox      │  [iter1] [iter2] [iter3*]       │  Details      │
│  editor    │  ←→ scrub between iterations    │  (heatmap)    │
│            │                                 │               │
├────────────┴─────────────────────────────────┴───────────────┤
│  STATUS BAR: "Iteration 2/5 — Mutating: shifted bbox right" │
└──────────────────────────────────────────────────────────────┘
```

**Key Interactions:**
- Type description or drop image → auto-analyze → show elements
- Edit elements (drag bboxes, rename, add/remove) → live JSON preview
- Click "Forge" → WebSocket connects → iterations stream in
- Each iteration shows: image, score bars, diagnosis text
- Click iteration in timeline → canvas updates with that iteration's image + heatmap
- Converged state → save to library, copy final prompt, download image

---

## 7. Alternatives Evaluated

### 7.1 Why not bundle a diffusion model?

**Evaluated:** Ship with SDXL/Flux/SD1.5 baked in.
**Rejected:** 
- Median r/StableDiffusion user has 8-12GB VRAM
- Adding 6-12GB model download kills adoption
- Everyone in that community ALREADY runs ComfyUI with their own models
- Connector pattern: zero VRAM overhead, works with any model

### 7.2 Why not use CLIP for scoring?

**Evaluated:** CLIP text-image similarity as primary score.
**Rejected:**
- CLIP saturates on semantic match ("both have a dog" = high score)
- Completely blind to composition (subject left vs right, framing)
- DINOv2 captures structural/visual similarity much better
- CLIP kept as secondary signal only

### 7.3 Why not LLM-only mutation?

**Evaluated:** Send everything to Qwen/GLM for prompt rewriting.
**Rejected:**
- Adds latency (2-5s per mutation)
- Requires API credits or local LLM VRAM
- Rule-based mutations are deterministic and instant
- LLM used as fallback when rules plateau, not primary

### 7.4 Why not React/Next.js frontend?

**Evaluated:** Full React app with component library.
**Rejected:**
- Adds build step, npm install, bundler complexity
- cocktailpeanut's tool went viral BECAUSE it was minimal
- Vanilla JS + ES modules: open index.html, it works
- Matches the "local tool" ethos of the target community

### 7.5 Why SQLite over PostgreSQL?

**Evaluated:** PostgreSQL (like InsForge DB on ZimaBoard).
**Rejected:**
- This is a single-user local tool, not a multi-tenant service
- SQLite: zero-config, single file, fast FTS5 search
- Users shouldn't need to install a database
- Can export to PostgreSQL later if taste model needs scale

---

## 8. Security Considerations

- **No outbound network calls** unless user configures API (Qwen-VL, OpenAI)
- **ComfyUI connector is localhost-only** by default (configurable for Tailscale)
- **No user data collected** — everything stays on local disk
- **No telemetry** — no analytics, no crash reporting, no phone home
- **Safetensors only** — never loads pickle files (deserialization risk)

---

## 9. Scalability Path

**Current:** Single-user local tool (v1)
**Future paths:**

1. **Multi-GPU:** Distribute iterations across GPUs (iter 1 on GPU0, iter 2 on GPU1)
2. **Taste Model Service:** Train on accumulated iteration data, serve as local API
3. **Composition Cloud:** Optional sync of anonymized compositions to shared library
4. **Plugin System:** Third-party scorers, mutators, analyzers via Python entry points
5. **ComfyUI Node Pack:** Embed as a native ComfyUI node (Stage 3)

---

## 10. Dependencies

### Python (server)
```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
websockets>=13.0
transformers>=4.45.0
torch>=2.7.0
torchvision>=0.22.0
Pillow>=10.0.0
numpy>=1.26.0
scikit-learn>=1.5.0  # for DINOv2 embeddings
aiohttp>=3.10.0
python-dotenv>=1.0.0
open-clip-torch>=2.26.0  # CLIP-interrogator vocab
```

### Optional
```
qwen-vl-utils  # for Qwen-VL style extraction
diffusers      # for safetensors metadata reading
```

### Frontend
```
None. Vanilla JS. No npm. No build step.
```

---

## 11. Project Structure

```
prompt-forge/
├── server/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings from .env
│   ├── analyzer/
│   │   ├── florence.py      # Florence-2 scene analysis
│   │   ├── style.py         # CLIP-vocab style extraction
│   │   └── palette.py       # Color palette sampling
│   ├── compiler/
│   │   ├── compiler.py      # JSON → workflow converter
│   │   ├── templates.py     # Workflow template registry
│   │   └── capability.py    # ComfyUI capability probe
│   ├── connector/
│   │   ├── comfyui.py       # ComfyUI REST + WebSocket client
│   │   └── progress.py      # Progress forwarding
│   ├── scorer/
│   │   ├── scorer.py        # Composite scoring engine
│   │   ├── compositional.py # Bbox IoU scoring
│   │   ├── visual.py        # DINOv2 per-region similarity
│   │   └── heatmap.py       # Region failure attribution
│   ├── mutator/
│   │   ├── mutator.py       # Mutation orchestrator
│   │   ├── rules.py         # Rule-based mutations
│   │   └── llm.py           # LLM-assisted mutations
│   ├── forge/
│   │   ├── engine.py        # Main convergence loop
│   │   └── session.py       # Session state management
│   ├── store/
│   │   ├── database.py      # SQLite operations
│   │   └── library.py       # Composition library CRUD
│   └── lora/
│       └── detector.py      # LoRA scanning + trigger extraction
├── frontend/
│   ├── index.html           # Single page
│   ├── css/
│   │   └── forge.css        # Dark theme, JetBrains Mono
│   └── js/
│       ├── app.js           # Main controller
│       ├── canvas.js        # Bbox editor + heatmap overlay
│       ├── iterations.js    # Timeline scrubbing
│       ├── library.js       # Composition library browser
│       └── api.js           # REST + WebSocket client
├── workflows/
│   └── templates/           # ComfyUI workflow JSONs
├── scripts/
│   ├── check_setup.py       # Pre-flight checks
│   └── install_workflows.py # Install templates to ComfyUI
├── tests/
│   ├── test_analyzer.py
│   ├── test_compiler.py
│   ├── test_scorer.py
│   └── test_forge.py
├── data/                    # Runtime data (gitignored)
│   ├── forge.db            # SQLite database
│   ├── outputs/            # Generated images
│   └── cache/              # Model cache
├── .env.example
├── .gitignore
├── requirements.txt
├── AGENTS.md
├── ARCHITECTURE.md
└── README.md
```

---

## 12. Launch Strategy

### Stage 1: Analyzer + Connector (days)
- Florence-2 image analysis → structured JSON
- ComfyUI connector with capability detection
- LoRA auto-detection and injection
- Mega-prompt fallback for all models
- Side-by-side: source image vs generated output
- **Post:** "Image-to-Prompt, but it generates AND knows your LoRAs"

### Stage 2: Convergence Loop (1-2 weeks)
- Closed-loop generate → score → mutate → regenerate
- Per-region heatmap visualization
- Convergence GIF as hero asset
- Composition library with search
- **Post:** "I made it grade its own output and retry until it matches"

### Stage 3: Distribution (month 1)
- ComfyUI custom node pack (ComfyUI Manager)
- Pinokio 1-click installer
- Premium features ($29): advanced mutations, batch forge, export
- **Product:** "Prompt Forge Pro" on Gumroad

---

*This document is the single source of truth for Prompt Forge architecture.*
*All implementation must conform to these decisions unless explicitly amended.*
