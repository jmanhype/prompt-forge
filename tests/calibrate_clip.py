#!/usr/bin/env python3
"""CLIP calibration test — score 5 images against 13 prompts at different match levels."""
import torch
import open_clip
from PIL import Image
import os

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Test images (known good Ideogram 4 + Ektachrome outputs)
images = {
    'apple': 'data/outputs/forge_test_1.png',
    'lora_apple': 'data/outputs/forge_test_2_lora.png',
    'verified': 'data/outputs/ideogram4_verified.png',
    'ektachrome': 'data/outputs/ideogram4_ektachrome.png',
    'forge_dog': 'data/outputs/3da0b1b2-b94c-4589-a4f4-24fe6c54cac3_forge_ideo4_00005_.png',
}

# Test prompts at different match levels
prompts = [
    ('EXACT', 'ektachrome vintage 1960s Kodak film photograph of a golden retriever in a concrete facility'),
    ('EXACT', 'ektachrome vintage 1960s Kodak film photograph of a red apple on a white table'),
    ('CLOSE', 'vintage film photograph of a dog in a building'),
    ('CLOSE', 'ektachrome style photograph'),
    ('PARTIAL', 'a dog sitting in a room'),
    ('PARTIAL', 'vintage photograph'),
    ('WRONG', 'a spaceship landing on mars'),
    ('WRONG', 'anime girl with pink hair'),
    ('WRONG', 'watercolor painting of mountains'),
    ('SHORT', 'a photograph'),
    ('SHORT', 'dog'),
    ('SHORT', 'vintage'),
    ('LONG', 'A high-resolution archival 16mm Ektachrome film photograph documented by a government agency, showing a golden retriever dog sitting upright on a concrete floor inside an empty industrial testing facility. Flat overcast lighting from overhead fluorescent panels.'),
]

results = []
for img_name, img_path in images.items():
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert('RGB')
    img_input = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        img_feat = model.encode_image(img_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    
    print(f"\n=== {img_name} ===")
    for category, prompt in prompts:
        text_input = tokenizer([prompt])
        with torch.no_grad():
            txt_feat = model.encode_text(text_input)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).item()
        results.append({'img': img_name, 'cat': category, 'prompt': prompt[:60], 'raw': sim})
        print(f"  [{category:7s}] raw={sim:.4f}  {prompt[:55]}")

print("\n\n=== RAW SCORE DISTRIBUTION ===")
for cat in ['EXACT', 'CLOSE', 'PARTIAL', 'WRONG', 'SHORT', 'LONG']:
    scores = [r['raw'] for r in results if r['cat'] == cat]
    if scores:
        print(f"  {cat:7s}: min={min(scores):.4f} max={max(scores):.4f} avg={sum(scores)/len(scores):.4f} n={len(scores)}")

print("\n=== RECOMMENDED THRESHOLDS ===")
exact = sorted([r['raw'] for r in results if r['cat'] == 'EXACT'])
close = sorted([r['raw'] for r in results if r['cat'] == 'CLOSE'])
partial = sorted([r['raw'] for r in results if r['cat'] == 'PARTIAL'])
wrong = sorted([r['raw'] for r in results if r['cat'] == 'WRONG'])

if exact and close and partial and wrong:
    converge_min = min(min(exact), min(close))
    mutate_min = min(partial)
    bad_max = max(wrong)
    print(f"  Converge threshold: raw >= {converge_min:.4f}")
    print(f"  Mutate range:       raw {mutate_min:.4f} to {converge_min:.4f}")
    print(f"  Bad match:         raw < {bad_max:.4f}")
    
    # Convert to normalized (current formula: (raw - 0.13) / 0.17)
    norm_converge = max(0, min(1, (converge_min - 0.13) / 0.17))
    norm_bad = max(0, min(1, (bad_max - 0.13) / 0.17))
    print(f"\n  With current normalization (raw-0.13)/0.17:")
    print(f"    Converge: {norm_converge:.3f}")
    print(f"    Bad:      {norm_bad:.3f}")
    
    # What if we use a better normalization?
    # Map: wrong_max -> 0.2, close_min -> 0.6, exact_max -> 1.0
    low = max(wrong)
    mid = min(close)
    high = max(exact)
    print(f"\n  Recommended normalization: map wrong_max({low:.4f})→0.15, close_min({mid:.4f})→0.50, exact_max({high:.4f})→0.85")
