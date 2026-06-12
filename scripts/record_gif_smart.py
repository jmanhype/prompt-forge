"""Record UI demo - wait for actual content before capturing."""
import time
import os
import shutil
from playwright.sync_api import sync_playwright
from PIL import Image
import numpy as np

FRAMES_DIR = "/Users/speed/prompt-forge/content/frames"
OUTPUT_GIF = "/Users/speed/prompt-forge/content/demo.gif"

# Clean up old frames
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)
os.makedirs(FRAMES_DIR)

frame_count = 0

def capture(page, hold=0.5):
    """Capture only if frame has actual content (not blank white)."""
    global frame_count
    
    # Take screenshot to temp file
    temp_path = os.path.join(FRAMES_DIR, f"temp.png")
    page.screenshot(path=temp_path)
    
    # Check if it's mostly white (blank)
    img = Image.open(temp_path)
    arr = np.array(img)
    mean_brightness = arr.mean()
    
    # If mean > 240, it's mostly white - skip it
    if mean_brightness > 240:
        print(f"  Skipping blank frame (brightness: {mean_brightness:.0f})")
        os.remove(temp_path)
        time.sleep(hold)
        return False
    
    # Otherwise save it
    final_path = os.path.join(FRAMES_DIR, f"frame_{frame_count:04d}.png")
    os.rename(temp_path, final_path)
    frame_count += 1
    print(f"  Captured frame {frame_count} (brightness: {mean_brightness:.0f})")
    time.sleep(hold)
    return True

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})

    # 1. Load page and WAIT for it to actually render
    print("Loading page...")
    page.goto("http://localhost:7861", wait_until="networkidle")
    time.sleep(3)  # Extra time for CSS/JS to render
    
    # Wait for textarea to be visible
    page.wait_for_selector("textarea", timeout=5000)
    time.sleep(1)

    # Hold on empty UI (3 frames)
    print("\nCapturing empty UI...")
    for _ in range(3):
        capture(page, hold=0.8)

    # 2. Click textarea and type SAFE prompt slowly
    print("\nTyping prompt...")
    textarea = page.locator("textarea").first
    textarea.click()
    time.sleep(0.5)
    capture(page, hold=0.3)

    prompt = "a beautiful mountain landscape at sunset"
    
    # Type in chunks for visual effect
    for i in range(0, len(prompt), 4):
        chunk = prompt[i:i+4]
        textarea.type(chunk, delay=80)
        time.sleep(0.1)
        capture(page, hold=0.15)

    # Hold on filled prompt (2 frames)
    for _ in range(2):
        capture(page, hold=0.5)

    # 3. Click FORGE
    print("\nClicking FORGE...")
    forge_btn = page.locator("#btn-forge").first
    forge_btn.click()
    capture(page, hold=1.0)

    # 4. Poll for updates - capture each state change
    print("\nWaiting for generation...")
    prev_body = ""
    for i in range(36):
        time.sleep(5)
        body = page.inner_text("body")

        # Only capture when something changed
        if body != prev_body:
            capture(page, hold=0.5)
            prev_body = body

        # Check if converged
        if "converged" in body.lower():
            print("  Converged!")
            for _ in range(5):
                capture(page, hold=0.8)
            break

        # Check for error (skip safety filter frames)
        if "Error:" in body and "NSFW" not in body and "safety" not in body.lower():
            print("  Error detected")
            capture(page, hold=0.5)
            for _ in range(3):
                capture(page, hold=0.5)
            break

    browser.close()

print(f"\nTotal captured: {frame_count} frames")

# Convert to GIF
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
    print(f"\n✓ GIF saved: {OUTPUT_GIF}")
    print(f"  Size: {size / 1024:.0f}KB")
else:
    print(f"\n✗ ffmpeg error: {result.stderr[-500:]}")

# Cleanup
shutil.rmtree(FRAMES_DIR)
print("  Frames cleaned up")
