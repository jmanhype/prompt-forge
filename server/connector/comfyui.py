"""ComfyUI REST client with polling fallback. Robust, no WebSocket dependency."""
from __future__ import annotations

import asyncio
import json
import uuid
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import aiohttp


@dataclass
class GenerationResult:
    prompt_id: str = ""
    images: list[str] = field(default_factory=list)  # local file paths
    duration_ms: int = 0
    error: Optional[str] = None


@dataclass
class ProgressUpdate:
    type: str = ""  # "progress" | "executing" | "executed" | "error"
    value: float = 0.0  # 0-1
    node: str = ""
    message: str = ""


class ComfyUIConnector:
    """Communicates with ComfyUI via REST API with polling fallback."""
    
    def __init__(self, url: str = "http://127.0.0.1:8188"):
        self.url = url.rstrip("/")
        self._client_id = str(uuid.uuid4())
    
    async def is_reachable(self) -> bool:
        """Check if ComfyUI is running and reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.url}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow and return the prompt_id."""
        payload = {
            "prompt": workflow,
            "client_id": self._client_id,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.url}/prompt", json=payload,
                                   timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"ComfyUI queue failed ({resp.status}): {text}")
                data = await resp.json()
                return data["prompt_id"]
    
    async def _check_history(self, prompt_id: str) -> Optional[dict]:
        """Check /history endpoint for completed prompt. Returns outputs dict or None."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.url}/history/{prompt_id}",
                                      timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return None
                    history = await resp.json()
                    if prompt_id in history:
                        return history[prompt_id]
        except Exception:
            pass
        return None
    
    async def _check_queue(self, prompt_id: str) -> Optional[str]:
        """Check /queue endpoint. Returns 'running', 'pending', or None (done/gone)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.url}/queue",
                                      timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return None
                    queue = await resp.json()
                    
                    # Check running queue
                    for item in queue.get("queue_running", []):
                        if len(item) > 1 and item[1] == prompt_id:
                            return "running"
                    
                    # Check pending queue
                    for item in queue.get("queue_pending", []):
                        if len(item) > 1 and item[1] == prompt_id:
                            return "pending"
        except Exception:
            pass
        return None
    
    async def wait_for_completion(
        self, prompt_id: str, timeout: int = 300
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """Wait for prompt completion using REST polling (no WebSocket dependency).
        
        Polls /queue and /history endpoints every 3 seconds.
        More reliable than WebSocket which can silently disconnect.
        """
        start = time.time()
        last_queue_status = "pending"
        
        while time.time() - start < timeout:
            # Check if it's in history (completed)
            history = await self._check_history(prompt_id)
            if history is not None:
                # Check for errors in status
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    error_msg = msgs[-1][1] if msgs else "Unknown ComfyUI error"
                    yield ProgressUpdate(type="error", message=str(error_msg))
                    return
                
                yield ProgressUpdate(type="executed", value=1.0)
                return
            
            # Check queue status
            queue_status = await self._check_queue(prompt_id)
            if queue_status == "running" and last_queue_status != "running":
                yield ProgressUpdate(type="executing", value=0.5, message="Generating...")
                last_queue_status = "running"
            elif queue_status == "pending":
                yield ProgressUpdate(type="progress", value=0.1, message="Queued...")
            elif queue_status is None and last_queue_status == "running":
                # Left queue but not in history yet — probably just finished
                # Give it a moment then check history again
                await asyncio.sleep(2)
                history = await self._check_history(prompt_id)
                if history is not None:
                    yield ProgressUpdate(type="executed", value=1.0)
                    return
            
            await asyncio.sleep(3)
        
        yield ProgressUpdate(type="error", message=f"Timeout after {timeout}s")
    
    async def get_outputs(self, prompt_id: str, output_dir: Path) -> list[str]:
        """Retrieve generated images for a completed prompt."""
        history = await self._check_history(prompt_id)
        if history is None:
            return []
        
        outputs = history.get("outputs", {})
        image_paths = []
        
        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                
                # Download the image
                params = {"filename": filename, "subfolder": subfolder, "type": "output"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.url}/view", params=params,
                                          timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            local_path = output_dir / f"{prompt_id}_{filename}"
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            local_path.write_bytes(content)
                            image_paths.append(str(local_path))
        
        return image_paths
    
    async def generate(
        self, workflow: dict, output_dir: Path, timeout: int = 300
    ) -> AsyncGenerator[ProgressUpdate | GenerationResult, None]:
        """Full generation pipeline: queue → poll → retrieve."""
        start = time.time()
        
        # Queue
        try:
            prompt_id = await self.queue_prompt(workflow)
        except Exception as e:
            yield GenerationResult(error=f"Queue failed: {e}")
            return
        
        yield ProgressUpdate(type="queued", message=f"Prompt {prompt_id[:8]}")
        
        # Wait via polling
        async for update in self.wait_for_completion(prompt_id, timeout):
            if update.type == "error":
                yield GenerationResult(prompt_id=prompt_id, error=update.message)
                return
            yield update
            if update.type == "executed":
                break
        
        # Retrieve
        images = await self.get_outputs(prompt_id, output_dir)
        duration = int((time.time() - start) * 1000)
        
        yield GenerationResult(
            prompt_id=prompt_id,
            images=images,
            duration_ms=duration,
        )
