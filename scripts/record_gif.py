"""Record UI demo as screenshot sequence → GIF."""
import time
import os
import shutil
from playwright.sync_api import sync_playwright

FRAMES_DIR = "/Users/speed/prompt-forge/content/frames"
OUTPUT_GIF = "/Users/speed/prompt-forge/content/demo.gif"

# Clean up old frames
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)
os.makedirs(FRAMES_DIR)

frame_count = 0

def capture(page, hold=0.5):
    global frame_count
    path = os.path.join(FRAMES_DIR, f"frame_{frame_count:04d}.png")
    page.screenshot(path=path)
    frame_count += 1
    time.sleep(hold)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})

    # 1. Load page
    page.goto("http://localhost:7861", wait_until="networkidle")
    time.sleep(2)

    # Hold on empty UI (3 frames)
    for _ in range(3):
        capture(page, hold=0.8)

    # 2. Click textarea and type prompt slowly
    textarea = page.locator("textarea").first
    textarea.click()
    time.sleep(0.5)
    capture(page, hold=0.3)

    prompt = "ektachrome vintage photograph of a cat on a bookshelf"
    # Type in chunks for visual effect
    for i in range(0, len(prompt), 4):
        chunk = prompt[i:i+4]
        textarea.type(chunk, delay=80)
        capture(page, hold=0.15)

    # Hold on filled prompt (2 frames)
    for _ in range(2):
        capture(page, hold=0.5)

    # 3. Click FORGE
    forge_btn = page.locator("#btn-forge").first
    forge_btn.click()
    capture(page, hold=1.0)

    # 4. Poll for updates — capture each state change
    prev_body = ""
    for i in range(36):  # 3 minutes max
        time.sleep(5)
        body = page.inner_text("body")

        # Only capture when something changed
        if body != prev_body:
            capture(page, hold=0.5)
            prev_body = body

        # Check if converged
        if "converged" in body.lower() or "Converged" in body:
            # Hold on final result (5 frames)
            for _ in range(5):
                capture(page, hold=0.8)
            break

        # Check for error
        if "Error:" in body:
            capture(page, hold=0.5)
            for _ in range(3):
                capture(page, hold=0.5)
            break

    browser.close()

print(f"Captured {frame_count} frames")

# Convert to GIF using ffmpeg
# 6 fps for snappy feel, optimized palette
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
    print(f"\nGIF saved: {OUTPUT_GIF}")
    print(f"Size: {size / 1024:.0f}KB ({size / (1024*1024):.1f}MB)")
else:
    print(f"ffmpeg error: {result.stderr[-500:]}")

# Cleanup frames
shutil.rmtree(FRAMES_DIR)
print("Frames cleaned up")
