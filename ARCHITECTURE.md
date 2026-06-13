# Prompt Forge Architecture: Design Patterns

## Overview

Prompt Forge had 18 hardcoded elements causing bugs (safety filter triggers, wrong
subject matching, forced aesthetics). This document maps each to the design pattern
that solves it, explains trade-offs, and provides implementation guidance.

---

## Pattern Summary

| Pattern | Purpose | Solves Elements | Key Benefit |
|---------|---------|-----------------|-------------|
| **Strategy** | Swappable enrichment approaches | #2, #3, #11, #12, #13, #14 | Context-aware, no forced aesthetics |
| **Factory** | Component creation | #4, #8, #17 | Configurable, auto-discovery |
| **Observer** | Score feedback decoupling | #10, #15, #18 | Derived thresholds, frontend colors |
| **Registry** | Configurable values | #6, #9 | External config, no code edits |
| **Template Method** | Workflow/prompt structure | #2, #4, #5, #17 | Parameterized skeleton |
| **Null Object** | Safe defaults | All patterns | No null checks, graceful degradation |

---

## 1. STRATEGY PATTERN

**File:** `server/patterns/strategy.py`

### Problem
Hardcoded enrichment lists forced Ektachrome film aesthetics on everything:
- `FILM_DETAILS` → "cyan color shift", "film grain" on a spaceship
- `CAMERA_DETAILS` → "flat even lighting" on dramatic scenes
- `SETTING_DETAILS` → "signs of age and weathering" triggering safety filter
- `SUBJECT_ENRICHMENTS` → "dog" matched "dogman", "man" matched "wolfman"

### Solution
Define `EnrichmentStrategy` interface with concrete implementations:

```
EnrichmentStrategy (interface)
├── NoEnrichment       (Null Object - pass through)
├── FilmEnrichment     (only when user asks for film/archival)
├── DigitalEnrichment  (when user asks for photo/digital)
├── IllustrationEnrichment (when user asks for art/painting)
└── ConceptArtEnrichment (when user asks for fantasy/sci-fi)
```

### How It Works

```python
# Factory selects strategy based on context
strategy = EnrichmentFactory.create(prompt, lora_config)

# Strategy enriches only if applicable
if strategy.should_apply(prompt, lora_config):
    caption = strategy.enrich_caption(caption)
    elements = strategy.enrich_elements(elements)
```

### Key Method: `should_apply()`
Each strategy checks if it's relevant:
- `FilmEnrichment.should_apply()` → True if prompt mentions "film", "archival", "vintage", or LoRA contains "ektachrome"
- `ConceptArtEnrichment.should_apply()` → True if prompt mentions "fantasy", "dragon", "robot", "creature"
- `NoEnrichment.should_apply()` → Always True (fallback)

### Trade-offs
| Pro | Con |
|-----|-----|
| No forced aesthetics | More code (interface + implementations) |
| Context-aware enrichment | Need to determine which strategy to use |
| Easy to add new aesthetics | Strategies must be well-designed |
| User intent preserved | Testing requires multiple strategy mocks |

### Implementation Notes
- Each strategy file is self-contained (can be added/removed independently)
- `NoEnrichment` is the Null Object default
- Strategies are stateless (can be reused across requests)
- `should_apply()` uses keyword matching — could be enhanced with LLM classification

---

## 2. FACTORY PATTERN

**File:** `server/patterns/factory.py`

### Problem
Hardcoded component creation:
- `open_clip.create_model_and_transforms("ViT-B-32", ...)` — model locked in code
- LoRA detection forced Ektachrome suffix on prompts
- Workflow resolution hardcoded to 768×768

### Solution
Factory classes that create components from configuration:

```
EnrichmentFactory.create(prompt, lora_config, preferred) → EnrichmentStrategy
ScorerFactory.create_scorer(model_name, pretrained, threshold) → Scorer
ScorerFactory.create_calibrated_scorer(model, calibration_data) → Scorer
WorkflowFactory.create_compiler(comfyui_url, templates_dir) → WorkflowCompiler
```

### How It Works

```python
# Enrichment: auto-select based on context
strategy = EnrichmentFactory.create(prompt, lora_config)

# Scorer: user can specify model
scorer = ScorerFactory.create_scorer(
    model_name=os.getenv("CLIP_MODEL", "ViT-B-32"),
    pretrained=os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k"),
    threshold=config.CONVERGENCE_THRESHOLD
)

# Workflow: auto-detect capabilities
compiler = await WorkflowFactory.create_compiler(
    comfyui_url=config.COMFYUI_URL,
    templates_dir=str(config.TEMPLATES_DIR)
)
```

### Trade-offs
| Pro | Con |
|-----|-----|
| Centralized creation logic | Extra abstraction layer |
| Configuration-driven | Factory can become complex |
| Easy to add new variants | Need to document factory methods |
| Testable (inject mock factories) | Runtime overhead for factory lookup |

---

## 3. OBSERVER PATTERN

**File:** `server/patterns/observer.py`

### Problem
Hardcoded score feedback:
- Diagnosis thresholds: `if score.style < 0.3` — magic numbers
- Heatmap colors in backend: `color = "#22c55e"` — UI concern in server code
- Mutation triggers: `if score.overall < 0.4` — not derived from convergence threshold

### Solution
Observers subscribe to score updates and react independently:

```
ObserverManager
├── DiagnosisObserver    → generates diagnosis messages (thresholds from config)
├── HeatmapObserver      → generates heatmap data (NO colors — frontend handles)
├── MutationTriggerObserver → decides when to mutate (derived from threshold)
└── LoggingObserver      → logs score history
```

### How It Works

```python
# Set up observers
manager = ObserverManager()
diagnosis = DiagnosisObserver(convergence_threshold=0.85)
heatmap = HeatmapObserver()
mutation = MutationTriggerObserver(convergence_threshold=0.85)

manager.register(diagnosis)
manager.register(heatmap)
manager.register(mutation)

# When score is computed
manager.notify_all(score, iteration)

# Each observer has its own data
messages = diagnosis.get_messages()
heatmap_data = heatmap.get_heatmap_data()
decision = mutation.get_mutation_decision()
```

### Key Insight: Derived Thresholds
```python
# DiagnosisObserver derives thresholds from convergence threshold
self.style_threshold = convergence_threshold * 0.6    # 0.85 * 0.6 = 0.51
self.subject_threshold = convergence_threshold * 0.6  # 0.51
self.overall_threshold = convergence_threshold * 0.4  # 0.34
```

Change the convergence threshold → all diagnosis thresholds update automatically.

### HeatmapObserver: No Colors
```python
# Backend sends raw scores
{
    "regions": [
        {"id": "0", "label": "subject", "score": 0.42, "bbox": [...]}
    ]
}

# Frontend CSS handles coloring:
# .heatmap-region[data-score-low] { background: #ef4444; }
# .heatmap-region[data-score-mid] { background: #eab308; }
# .heatmap-region[data-score-high] { background: #22c55e; }
```

### Trade-offs
| Pro | Con |
|-----|-----|
| Decoupled scoring from feedback | Event ordering can matter |
| Add observers without changing scorer | Harder to debug event chains |
| Thresholds derived, not hardcoded | Slight performance overhead |
| Frontend handles presentation | Observers must be stateless or careful with state |

---

## 4. REGISTRY PATTERN

**File:** `server/patterns/registry.py`

### Problem
Hardcoded values that require code edits to change:
- `NODE_SIGNATURES` dict — new ComfyUI nodes = code change
- Normalization breakpoints — new CLIP model = code change
- Model filenames — different installation = code change

### Solution
Load from config files, provide defaults, allow runtime registration:

```
NodeRegistry          → ComfyUI node detection signatures
CalibrationRegistry   → Per-model CLIP normalization breakpoints
ThresholdRegistry     → Derived scoring/mutation thresholds
ModelRegistry         → Model names, paths, architectures
```

### How It Works

```python
# Load from config file
node_reg = NodeRegistry(config_path=Path("config/node_signatures.json"))
cal_reg = CalibrationRegistry(config_path=Path("config/calibration.json"))

# Auto-detect capabilities using registry
available_nodes = set(object_info.keys())
has_ideogram4 = node_reg.check_capability("has_ideogram4", available_nodes)

# Normalize score using registry
normalized = cal_reg.normalize(raw_score, "ViT-B-32")

# Register new model at runtime
model_reg.register("clip", "ViT-L-14-336", {
    "pretrained": "openai",
    "description": "Higher resolution, slower"
})
```

### Config File Example: `config/node_signatures.json`
```json
{
  "has_ideogram4": ["IdeogramV4", "Ideogram4Scheduler", "CLIPLoader"],
  "has_flux": ["CLIPTextEncodeFlux", "ModelSamplingFlux"],
  "has_gligen": ["GLIGENLoader", "GLIGENTextBoxApply"]
}
```

### Config File Example: `config/calibration.json`
```json
{
  "ViT-B-32": {
    "breakpoints": [[0.15, 0.0, 0.15], [0.22, 0.15, 0.35], [0.30, 0.50, 0.35], [0.35, 0.85, 0.15]],
    "description": "65 measurements, Ideogram 4 outputs"
  },
  "ViT-L-14": {
    "breakpoints": [[0.20, 0.0, 0.15], [0.28, 0.15, 0.35], [0.36, 0.50, 0.35], [0.42, 0.85, 0.15]],
    "description": "Needs calibration study"
  }
}
```

### Trade-offs
| Pro | Con |
|-----|-----|
| No code edits for new configs | Config loading overhead |
| Version-controlled configuration | Need to handle missing/invalid configs |
| Runtime updates possible | Config file management |
| Defaults when config missing | Testing requires config fixtures |

---

## 5. TEMPLATE METHOD PATTERN

**File:** `server/patterns/template.py`

### Problem
Hardcoded workflow structure:
- Resolution: 768×768 locked in template JSON
- Steps: 28 locked in template
- Caption template: `f"{subject} {setting}. Professional photography..."` forces photography
- Caption prefix: `f"A photograph of {caption}"` forces medium
- LoRA suffix: hardcoded Ektachrome text appended

### Solution
`WorkflowTemplate` defines the skeleton, parameters fill in specifics:

```
WorkflowTemplate.compile(prompt_text, seed)
├── _build_model_loaders()     → UNET, CLIP, VAE, unconditional
├── _build_lora_loader()       → LoRA injection (if enabled)
├── _build_prompt_encoding()   → CLIPTextEncode + ConditioningZeroOut
├── _build_noise_schedule()    → AuraFlow + CFGOverride
├── _build_guider()            → DualModelGuider
├── _build_sampling()          → SamplerCustomAdvanced
└── _build_decode_and_save()   → VAEDecode + SaveImage
```

`PromptTemplate` builds prompts without forcing a medium:

```python
# OLD: Forces "photograph" on everything
f"A photograph of {caption}. Professional photography, sharp focus, natural daylight."

# NEW: User's text passes through, context added only if missing
PromptTemplate()
    .with_subject("a wolfman in a dark castle")  # User's exact words
    .build()  # → "a wolfman in a dark castle"
```

### How It Works

```python
# Create template with overrides
template = WorkflowTemplate(overrides={
    'width': 1024,        # Override resolution
    'height': 768,
    'steps': 40,          # Override steps
    'cfg': 5.0,           # Override CFG
    'lora_enabled': True,
    'lora_name': 'ektachrome_style_v1.safetensors',
})

# Compile workflow
workflow = template.compile(prompt_text=json_prompt, seed=42)

# Build prompt without forcing medium
prompt = PromptTemplate()
    .with_subject(user_input)
    .with_setting("in a moonlit forest clearing")
    .with_lora_triggers(["ektachrome"])  # Only if LoRA loaded
    .build()
```

### Trade-offs
| Pro | Con |
|-----|-----|
| Clear structure: fixed skeleton, variable parts | Can become complex with many parameters |
| Per-request configuration | Radical departures need new templates |
| User's text passes through unchanged | Need to document all parameters |
| No forced medium/aesthetic | Template bloat risk |

---

## 6. NULL OBJECT PATTERN

**File:** `server/patterns/null.py`

### Problem
Components might be unavailable (Florence-2 not installed, ComfyUI down, no LoRAs).
Code needs null checks everywhere or crashes.

### Solution
Null implementations that are safe to call but do nothing:

```
NullEnrichment     → enrich_caption() returns input unchanged
NullScorer         → score() returns neutral 0.5 scores
NullMutator        → mutate() returns prompt unchanged
NullConnector      → generate() returns mock results
NullAnalyzer       → analyze() returns minimal analysis
NullLoRADetector   → match() always returns None
```

### How It Works

```python
# Instead of checking if components exist:
if analyzer is not None:
    analysis = analyzer.analyze(image)
else:
    analysis = default_analysis

# Just use the component (might be Null):
analysis = analyzer.analyze(image)  # Works even if NullAnalyzer
```

### Trade-offs
| Pro | Con |
|-----|-----|
| No null checks needed | Can hide real errors |
| Safe defaults | Null objects must be truly safe |
| Simplifies client code | Need null for every interface |
| Graceful degradation | Testing needs both real and null |

---

## MAPPING: All 18 Hardcoded Elements → Patterns

| # | Element | Severity | Pattern | File |
|---|---------|----------|---------|------|
| 1 | setting_map (30 entries) | **Critical** | Strategy + Template | engine.py |
| 2 | Caption template | **High** | Template | engine.py |
| 3 | Fallback setting | **High** | Null + Strategy | engine.py |
| 4 | LoRA film suffix | **High** | Strategy | compiler.py |
| 5 | Caption prefix | **Medium** | Template | compiler.py |
| 6 | NODE_SIGNATURES | **Low** | Registry | capability.py |
| 7 | strategy_description | **Low** | Registry | capability.py |
| 8 | CLIP model selection | **Medium** | Factory + Registry | scorer.py |
| 9 | Normalization breakpoints | **Medium** | Registry | scorer.py |
| 10 | Diagnosis thresholds | **Low** | Observer | scorer.py |
| 11 | FILM_DETAILS | **Critical** | Strategy | rules.py |
| 12 | CAMERA_DETAILS | **High** | Strategy | rules.py |
| 13 | SETTING_DETAILS | **Critical** | Strategy (delete) | rules.py |
| 14 | SUBJECT_ENRICHMENTS | **High** | Strategy + Null | rules.py |
| 15 | Mutation thresholds | **Medium** | Observer | rules.py |
| 16 | LLM parameters | **Low** | Registry | llm.py |
| 17 | Workflow template values | **Medium** | Template + Registry | template JSON |
| 18 | Heatmap colors | **Low** | Observer (remove) | heatmap.py |

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Day 1)
1. **Delete** `setting_map` entirely → pass user text through as-is
2. **Delete** `SETTING_DETAILS` → triggers safety filter
3. **Delete** LoRA film suffix in compiler → LoRA weights handle style
4. **Replace** `SUBJECT_ENRICHMENTS` with `NullEnrichment` default

### Phase 2: Strategy Pattern (Day 2)
1. Create `EnrichmentStrategy` interface
2. Implement `NoEnrichment`, `FilmEnrichment`, `ConceptArtEnrichment`
3. Wire `EnrichmentFactory` into mutator
4. Mutator calls `strategy.should_apply()` before enriching

### Phase 3: Observer Pattern (Day 3)
1. Create `ObserverManager` in engine
2. Implement `DiagnosisObserver` with derived thresholds
3. Implement `HeatmapObserver` (no colors)
4. Implement `MutationTriggerObserver`
5. Move heatmap colors to frontend CSS

### Phase 4: Registry Pattern (Day 4)
1. Create `config/` directory with JSON files
2. Implement `NodeRegistry` loading from `config/node_signatures.json`
3. Implement `CalibrationRegistry` loading from `config/calibration.json`
4. Implement `ModelRegistry` for model discovery

### Phase 5: Template Pattern (Day 5)
1. Create `WorkflowTemplate` class
2. Create `PromptTemplate` builder
3. Replace hardcoded compiler with template-based compilation
4. Add per-request parameter overrides

### Phase 6: Factory + Null (Day 6)
1. Implement all factories
2. Implement all null objects
3. Wire factories into engine initialization
4. Add graceful degradation for missing components

---

## Architecture Diagram

```
User Input: "a wolfman"
    │
    ▼
┌─────────────────────┐
│   ForgeEngine       │
│                     │
│  1. Parse input     │──→ PromptTemplate.with_subject("a wolfman").build()
│     (no rewriting)  │    → "a wolfman"  (untouched!)
│                     │
│  2. Select strategy │──→ EnrichmentFactory.create(prompt, lora)
│                     │    → ConceptArtEnrichment (matched "wolfman")
│                     │
│  3. Build JSON      │──→ PromptTemplate.build_json()
│     with bbox       │    → {"high_level_description": "a wolfman", ...}
│                     │
│  4. Compile         │──→ WorkflowTemplate.compile(json_prompt)
│     workflow        │    → Complete ComfyUI workflow dict
│                     │
│  5. Generate        │──→ ComfyUIConnector (or NullConnector)
│                     │    → Image or error
│                     │
│  6. Score           │──→ ScorerFactory.create_scorer()
│                     │    → ForgeScore
│                     │
│  7. Observe         │──→ ObserverManager.notify_all(score)
│                     │    → Diagnosis, Heatmap, MutationTrigger
│                     │
│  8. Mutate          │──→ Strategy.enrich_caption() (if applicable)
│     (if needed)     │    → Enhanced prompt for next iteration
└─────────────────────┘
```

---

## Key Principles

1. **User's text is sacred.** Never rewrite it. Only append context the user hasn't specified.

2. **JSON format is the bypass.** The structured JSON with bounding boxes is what
   bypasses the safety filter. Enrichment is optional and contextual.

3. **LoRA weights handle style.** Don't duplicate the LoRA's aesthetic in text.
   If a LoRA is loaded, its weights apply the style. Text triggers are separate.

4. **Thresholds derive from convergence.** All scoring thresholds should be
   calculated from the convergence threshold, not hardcoded independently.

5. **Frontend handles presentation.** Colors, fonts, layouts are CSS. Backend
   sends data. Frontend renders it.

6. **Configuration over code.** Anything that might change between installations
   or use cases should be in a config file, not in source code.

7. **Null-safe by default.** Every component has a null implementation. The system
   degrades gracefully when components are missing.
