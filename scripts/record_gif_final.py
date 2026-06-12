"""Record UI demo - wait for full render before capturing."""
import time
import os
import shutil
from playwright.sync_api import sync_playwright
from PIL import Image
import numpy as np

FRAMES_DIR = "/Users/speed/prompt-forge/content/frames"
OUTPUT_GIF = "/Users/speed/prompt-forge/content/demo.gif"

if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)
os.makedirs(FRAMES_DIR)

frame_count = 0

def capture(page, hold=0.5):
    global frame_count
    temp_path = os.path.join(FRAMES_DIR, "temp.png")
    page.screenshot(path=temp_path)
    
    img = Image.open(temp_path)
    arr = np.array(img)
    mean_brightness = arr.mean()
    
    # Skip if still loading (too bright = not rendered yet)
    if mean_brightness > 200:
        print(f"  Skipping - page still loading (brightness: {mean_brightness:.0f})")
        os.remove(temp_path)
        time.sleep(hold)
        return False
    
    final_path = os.path.join(FRAMES_DIR, f"frame_{frame_count:04d}.png")
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        background.save(final_path, 'PNG')
    else:
        os.rename(temp_path, final_path)
    
    frame_count += 1
    print(f"  Frame {frame_count} (brightness: {mean_brightness:.0f})")
    time.sleep(hold)
    return True

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})

    print("Loading page...")
    page.goto("http://localhost:7861", wait_until="networkidle")
    
    # Wait for dark theme to render (background should be dark)
    print("Waiting for dark theme to load...")
    for i in range(10):
        page.screenshot(path="/tmp/check_load.png")
        img = Image.open("/tmp/check_load.png")
        brightness = np.array(img).mean()
        print(f"  Attempt {i+1}: brightness {brightness:.0f}")
        
        if brightness < 50:  # Dark theme loaded
            print("  ✓ Dark theme loaded")
            break
        time.sleep(2)
    else:
        print("  ⚠ Timeout waiting for dark theme")
    
    # Wait for UI elements
    page.wait_for_selector("textarea", timeout=10000)
    page.wait_for_selector("#btn-forge", timeout=10000)
    time.sleep(2)  # Extra buffer for all CSS to apply
    
    print("\nRecording UI demo...")
    
    # Capture empty UI (3 frames)
    print("\n1. Empty UI")
    for _ in range(3):
        capture(page, hold=0.8)

    # Type prompt
    print("\n2. Typing prompt")
    textarea = page.locator("textarea").first
    textarea.click()
    time.sleep(0.5)
    capture(page, hold=0.3)

    prompt = "a beautiful mountain landscape at sunset"
    for i in range(0, len(prompt), 4):
        chunk = prompt[i:i+4]
        textarea.type(chunk, delay=80)
        time.sleep(0.1)
        capture(page, hold=0.15)

    for _ in range(2):
        capture(page, hold=0.5)

    # Click FORGE
    print("\n3. Clicking FORGE")
    forge_btn = page.locator("#btn-forge").first
    forge_btn.click()
    capture(page, hold=1.0)

    # Wait for generation
    print("\n4. Waiting for generation")
    prev_body = ""
    for i in range(36):
        time.sleep(5)
        body = page.inner_text("body")

        if body != prev_body:
            capture(page, hold=0.5)
            prev_body = body

        if "converged" in body.lower():
            print("  ✓ Converged!")
            for _ in range(5):
                capture(page, hold=0.8)
            break

        if "Error:" in body and "NSFW" not in body:
            print("  ✗ Error")
            capture(page, hold=0.5)
            for _ in range(3):
                capture(page, hold=0.5)
            break

    browser.close()

print(f"\nCaptured {frame_count} frames")

# Build GIF
import subprocess
cmd = [
    "ffmpeg", "-y",
    "-framerate", "6",
    "-i", os.path.join(FRAMES_DIR, "frame_%04d.png"),
    "-vf", "fps=6,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=floyd_steinberg",
    "-loop", "0",
    OUTPUT_GIF
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    size = os.path.getsize(OUTPUT_GIF)
    print(f"✓ GIF: {OUTPUT_GIF} ({size/1024:.0f}KB)")
else:
    print(f"✗ Error: {result.stderr[-300:]}")

shutil.rmtree(FRAMES_DIR)
