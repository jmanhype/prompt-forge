# Atomic Breath Benchmark — June 10, 2026

> Testing 3 approaches to generate Godzilla with atomic breath using the Ektachrome Style LoRA v1 on Ideogram 4 FP8.

## Context

After training the Ektachrome Style LoRA v1 on Ideogram 4 FP8, we needed to verify it could handle complex effects (atomic breath = blue energy beam from Godzilla's mouth + glowing dorsal plates + overexposure).

**Session:** "Ideogram 4 LoRA Training Setup #18"
**Session ID:** `20260610_112537_aca1f9`
**Date:** June 10, 2026
**Tool:** Hermes CLI with Qwen model
**Method:** `execute_code` to build ComfyUI workflow JSONs programmatically and POST to `http://127.0.0.1:8188/prompt`

## First Attempt: img2img @ 0.65 — FAILED

Tried img2img at 0.65 denoise from a reference image. Z.ai Vision confirmed:

```
PRESENT:
  ✓ Godzilla - bipedal, dark grey, scaly skin, long tail
  ✓ Blue glowing dorsal plates along back
  ✓ Industrial complex - buildings, metal towers, utility poles
  ✓ Vintage 16mm film aesthetic - grain, faded colors, rounded corners
  ✓ Blue sky visible

MISSING:
  ✗ NO atomic breath beam from the mouth
  ✗ Sky is clear blue (not overexposed white)
  ✗ Not high-contrast "panic" shot

VERDICT: The img2img at 0.65 denoise barely changed anything.
The blue dorsal plates are there because they were ALREADY BLUE
in the reference image. The Ektachrome LoRA added grain/film look
but did NOT add the atomic breath or the extreme overexposure.
```

**Root cause:** Denoise 0.65 is too conservative for ADDING new elements. The reference image dominated, and the model didn't have enough creative freedom to generate the beam.

## Second Attempt: 3 Parallel Approaches

Queued 3 approaches simultaneously via ComfyUI API:

### Approach 1: txt2img (denoise 1.0) + LoRA

```json
{
  "high_level_description": "A wide 1966 classified military 16mm Kodak Ektachrome observation frame showing Godzilla firing a massive blue atomic breath beam across an industrial waterfront, full body visible, dorsal plates glowing bright blue from tail to neck.",
  "style_description": {
    "aesthetics": "Declassified 1960s military archive footage, observational and unsentimental, scientific documentation framing, aged but clean Ektachrome color",
    "lighting": "Hard late-afternoon industrial daylight filtered through smoke haze, bright clipped highlights on metal roofs, deep cyan-tinted shadows, overexposed sky from energy beam",
    "photo": "16mm Kodak Ektachrome observation reel frame, 1966, visible organic film grain, subtle dust specks, slight emulsion texture, faint projector gate weave",
    "medium": "photograph",
    "color_palette": ["#1D2730", "#3D5F6B", "#7EA6B8", "#C47A3C", "#B7A46A", "#4A90E2"]
  },
  "compositional_deconstruction": {
    "background": "A wide industrial military-adjacent waterfront under smoky haze, low horizon at the lower third, faded Ektachrome cyan shadows. No modern vehicles, no contemporary architecture, no digital sharpness, no extra monsters, no commercial logos, no watermark.",
    "elements": [
      {
        "type": "obj",
        "bbox": [118, 325, 865, 690],
        "desc": "Godzilla standing in a wide scientific-observation pose, full body visible, massive reptilian silhouette towering over industrial buildings, charcoal-scaled skin with practical-suit texture, dorsal plates glowing bright blue from tail to neck",
        "color_palette": ["#111417", "#232B2F", "#3E4B4E", "#6D7A78", "#4A90E2"]
      },
      {
        "type": "obj",
        "bbox": [200, 100, 400, 250],
        "desc": "Massive blue-white energy beam firing from Godzilla's open mouth across the frame, overexposed sky behind beam, air distortion warping edges of buildings",
        "color_palette": ["#4A90E2", "#FFFFFF", "#E0F0FF"]
      }
    ]
  }
}
```

**Workflow:**
- UNETLoader (ideogram4_fp8_scaled, fp8_e4m3fn)
- CLIPLoader (qwen3vl_8b_fp8_scaled, type=ideogram4)
- LoraLoader (ektachrome_style_v1, model:0.8, clip:0.8)
- CLIPTextEncode (JSON prompt above)
- ConditioningZeroOut (negative)
- EmptyFlux2LatentImage (1024x768)
- KSampler (30 steps, cfg 3.5, euler, simple, denoise 1.0)
- VAELoader (flux2-vae)
- VAEDecode
- SaveImage

**Result:** ✅ SUCCESS

Z.ai Vision confirmed:
- Blue energy beam from mouth: YES
- Dorsal plates glowing: YES
- Overexposure/blown highlights: YES
- 16mm Ektachrome aesthetic: YES
- Godzilla subject fidelity: YES

**File:** `atomic_txt2img.png` (1024x768, 1.2MB)

### Approach 2: img2img @ 0.85 + LoRA

Same prompt as above, but:
- LoadImage (reference: magnitude_godzilla from kaiju bible)
- VAEEncode (pixels → latent)
- KSampler denoise: 0.85

**Result:** ✅ SUCCESS

All elements present. Composition influenced by reference (similar layout to magnitude_godzilla), but atomic breath and overexposure both rendered.

**File:** `atomic_img2img_85.png` (1536x1024, 2.1MB)

### Approach 3: No LoRA (Control)

Same prompt and workflow as Approach 1, but **without** the Ektachrome LoRA.

**Result:** ✅ SUCCESS

Atomic breath works without LoRA — Ideogram 4 knows kaiju natively. But lacks the specific Ektachrome 16mm film grain, cyan shadows, and "1966 observation reel" framing.

**File:** `atomic_no_lora.png` (1024x768, 1.2MB)

## Results Summary

| Approach | Godzilla | Beam | Plates | Film | Exposure |
|----------|----------|------|--------|------|----------|
| 1. txt2img (scratch + LoRA) | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2. img2img @ 0.85 + LoRA | ✅ | ✅ | ✅ | ✅ | ✅ |
| 3. No LoRA (control) | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| 4. img2img @ 0.65 + LoRA | ✅ | ❌ | ✅ | ✅ | ❌ |

**Key findings:**

1. **Denoise level is critical for adding new elements.** 0.65 is too conservative — the reference dominates and the model can't invent new content. Use 0.85+ for creative effects.

2. **txt2img is best for creative freedom.** When you want the model to invent the composition (not just apply a style), use denoise 1.0 from scratch.

3. **Ideogram 4 knows kaiju natively.** Atomic breath works without the LoRA, but the LoRA adds the specific 16mm Ektachrome aesthetic (grain, cyan shadows, film artifacts).

4. **The LoRA adds style, not content.** It doesn't force Godzilla or atomic breath — it just applies the visual treatment (film grain, color grading, framing).

## Denoise Level Playbook

| Use Case | Denoise | Method | When to Use |
|----------|---------|--------|-------------|
| **Style transfer only** | 0.4-0.6 | img2img | Same subject, new look |
| **Heavy modification** | 0.7-0.85 | img2img | Major changes, keep layout |
| **Creative content** | 1.0 | txt2img | New elements, effects, inventions |
| **Near-complete regen** | 0.85+ | img2img | Reference is just starting point |

**Critical lesson:** For ADDING new elements (beams, effects, objects): use txt2img or img2img @ 0.85+. For STYLE TRANSFER only: img2img @ 0.4-0.6 is fine. The model needs creative freedom (high denoise) to generate novel content.

## img2img Chaining

After the atomic breath test, we discovered you can chain img2img passes for cumulative style application:

1. Generate base image (txt2img or upload reference)
2. Save to ComfyUI input folder: `/home/straughter/ComfyUI/input/`
3. LoadImage node → VAEEncode → KSampler (denoise) → VAEDecode
4. Output saves to: `/home/straughter/ComfyUI/output/`
5. Copy output back to input for next pass if needed

**Use cases:**
- Progressive style application (each pass adds more Ektachrome)
- Multi-stage refinement (composition → style → effects)
- Iterative prompting (adjust prompt between passes)

**Warning:** Multiple passes can over-process (too much grain, muddy colors).

## Quality Control with Z.ai Vision MCP

All images were analyzed with Z.ai Vision MCP (`~/.hermes/bin/zai-vision-mcp-wrapper`), which proved reliable for:
- Subject identification (Godzilla, industrial scenes, etc.)
- Color analysis (cyan shadows, blue glow detection)
- Aesthetic verification (16mm film, grain, vintage look)
- Effect detection (atomic breath, energy beams, overexposure)
- Detailed compositional breakdown

**Usage pattern:**

```python
import subprocess, json

def zai_analyze_image(image_path, prompt):
    init_msg = json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize",
        "params":{"protocolVersion":"2024-11-05","capabilities":{},
        "clientInfo":{"name":"hermes","version":"1.0"}}})
    init_done = json.dumps({"jsonrpc":"2.0","method":"notifications/initialized"})
    tool_call = json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/call",
        "params":{"name":"analyze_image","arguments":{
            "image_source":image_path, "prompt":prompt}}})
    
    input_data = init_msg + "\n" + init_done + "\n" + tool_call + "\n"
    proc = subprocess.run(["/Users/speed/.hermes/bin/zai-vision-mcp-wrapper"],
        input=input_data, capture_output=True, text=True, timeout=180)
    
    for line in proc.stdout.strip().split("\n"):
        try:
            msg = json.loads(line.strip())
            if msg.get("id") == 2 and "result" in msg:
                content = msg["result"].get("content", [])
                return "\n".join(i["text"] for i in content if i.get("type") == "text")
        except: continue
    return "ERROR"

# Example usage
result = zai_analyze_image(
    "/Users/speed/sglx_lora_results/ektachrome_v1/atomic_txt2img.png",
    "Does this image show Godzilla with atomic breath (blue energy beam)? Describe the style, lighting, film grain."
)
print(result)
```

## Key Learnings

1. **Z.ai Vision MCP is reliable** — use it for all image analysis, not built-in vision tools
2. **JSON prompting matters** — proper spec gives 3-5x more detail than plain text
3. **Denoise level is critical** — 0.65 too low for new elements, 0.85+ for creative work
4. **txt2img for creativity** — when you want the model to invent, use denoise 1.0
5. **Style LoRA adds aesthetic** — doesn't force subjects, just visual treatment
6. **Block-weighting prevents distortion** — use Inspire Pack for fine control
7. **img2img chaining works** — multiple passes for cumulative style application
8. **File size indicates quality** — 1.5MB+ PNGs are real images, <200K often safety blocks

## File Locations

**Benchmark outputs:**
- `/Users/speed/sglx_lora_results/ektachrome_v1/atomic_txt2img.png`
- `/Users/speed/sglx_lora_results/ektachrome_v1/atomic_img2img_85.png`
- `/Users/speed/sglx_lora_results/ektachrome_v1/atomic_no_lora.png`
- `/Users/speed/sglx_lora_results/ektachrome_v1/magnitude_atomic.png`

**Trained LoRA:**
- `/home/straughter/ai-toolkit/output/ektachrome_style_v1/ektachrome_style_v1.safetensors` (202MB)
- `/home/straughter/ComfyUI/models/loras/ektachrome_style_v1.safetensors` (installed)

**ComfyUI Models:**
- `/home/straughter/ComfyUI/models/diffusion_models/ideogram4_fp8_scaled.safetensors`
- `/home/straughter/ComfyUI/models/text_encoders/qwen3vl_8b_fp8_scaled.safetensors`
- `/home/straughter/ComfyUI/models/vae/flux2-vae.safetensors`

**Configs:**
- `/home/straughter/ai-toolkit/config/train_ideogram4_style_lora.yaml`

**Vision MCP:**
- `/Users/speed/.hermes/bin/zai-vision-mcp-wrapper`
- `/opt/homebrew/bin/zai-mcp-server`

**Prompt Forge repo:**
- `/Users/speed/prompt-forge/workflows/templates/ideogram4_lora.json`

## Related Documentation

- [[Ektachrome-LoRA-v1-Ideogram4-Style-Training]] — Full training config and pipeline (Mac Mini zettel)
- [[BATMANOSAMA-Brand-Color-Palette-2026]] — Brand colors (Rebel Red, Cinema Black, Warm White)
- Mega Gist v10.0 — Full technical details + session transcript
- Prompt Forge repo — `/Users/speed/prompt-forge/` (closed-loop composition engine)
