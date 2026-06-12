"""Record UI demo video using Playwright."""
import time
from playwright.sync_api import sync_playwright


def record_demo():
    with sync_playwright() as p:
        # Launch browser with video recording
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1400, "height": 900},
            record_video_dir="/Users/speed/prompt-forge/content/",
            record_video_size={"width": 1400, "height": 900}
        )
        page = context.new_page()
        
        print("Recording UI demo...")
        print("1. Loading page...")
        page.goto("http://localhost:7861", wait_until="networkidle")
        time.sleep(2)
        
        # Show the empty UI
        print("2. Showing empty UI...")
        time.sleep(3)
        
        # Type a prompt
        print("3. Entering prompt...")
        textarea = page.locator("textarea").first
        textarea.click()
        time.sleep(0.5)
        
        # Type character by character for visual effect
        prompt = "ektachrome vintage photograph of a cat sitting on a bookshelf"
        for char in prompt:
            textarea.type(char, delay=50)
            time.sleep(0.02)
        time.sleep(2)
        
        # Click FORGE
        print("4. Clicking FORGE button...")
        forge_btn = page.locator("#btn-forge").first
        forge_btn.click()
        time.sleep(2)
        
        # Wait for generation and scoring
        print("5. Waiting for generation (this takes ~60s per iteration)...")
        
        # Poll for updates
        max_wait = 180  # 3 minutes max
        start = time.time()
        last_iteration = 0
        
        while time.time() - start < max_wait:
            time.sleep(5)
            
            # Check for iteration updates
            try:
                iteration_text = page.locator(".iteration-counter").first.inner_text()
                iteration = int(iteration_text) if iteration_text.isdigit() else 0
                
                if iteration > last_iteration:
                    print(f"   Iteration {iteration} started...")
                    last_iteration = iteration
                
                # Check for scores
                overall = page.locator("#score-overall").first.inner_text()
                if overall and overall != "0":
                    print(f"   Score: {overall}%")
                    
            except Exception:
                pass
            
            # Check if converged
            body_text = page.locator("body").inner_text()
            if "CONVERGED" in body_text.upper():
                print("6. Converged!")
                time.sleep(3)
                break
        
        # Show final state
        print("7. Showing final result...")
        time.sleep(5)
        
        # Scroll to show different parts
        page.evaluate("window.scrollTo(0, 300)")
        time.sleep(2)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(2)
        
        # Close and save video
        print("8. Saving video...")
        video_path = page.video.path()
        context.close()
        browser.close()
        
        print(f"\n✓ Video saved to: {video_path}")
        return video_path


if __name__ == "__main__":
    path = record_demo()
    print(f"\nDemo video: {path}")
    print("\nTo convert to GIF (if needed):")
    print(f"  ffmpeg -i {path} -vf 'fps=10,scale=800:-1:flags=lanczos' -loop 0 content/demo.gif")
