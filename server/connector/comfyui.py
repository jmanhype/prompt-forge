"""ComfyUI REST + WebSocket client. Connector pattern — never vendor."""
from __future__ import annotations

import asyncio
import json
import uuid
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
    type: str = ""  # "progress" | "executing" | "executed"
    value: float = 0.0  # 0-1
    node: str = ""
    message: str = ""


class ComfyUIConnector:
    """Communicates with a local ComfyUI instance via REST + WebSocket."""
    
    def __init__(self, url: str = "http://127.0.0.1:8188"):
        self.url = url.rstrip("/")
        self.ws_url = self.url.replace("http", "ws") + "/ws"
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
            async with session.post(f"{self.url}/prompt", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"ComfyUI queue failed ({resp.status}): {text}")
                data = await resp.json()
                return data["prompt_id"]
    
    async def wait_for_completion(
        self, prompt_id: str, timeout: int = 300
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """Wait for a prompt to complete, yielding progress updates."""
        import websockets
        
        uri = f"{self.ws_url}?clientId={self._client_id}"
        try:
            async with websockets.connect(uri) as ws:
                while True:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                    except asyncio.TimeoutError:
                        yield ProgressUpdate(type="error", message=f"Timeout after {timeout}s")
                        return
                    
                    msg_type = msg.get("type", "")
                    data = msg.get("data", {})
                    
                    if msg_type == "progress":
                        value = data.get("value", 0) / max(data.get("max", 1), 1)
                        yield ProgressUpdate(type="progress", value=value)
                    
                    elif msg_type == "executing":
                        node = data.get("node", "")
                        if node is None:
                            # Execution complete
                            yield ProgressUpdate(type="executed", value=1.0)
                            return
                        yield ProgressUpdate(type="executing", node=node)
                    
                    elif msg_type == "execution_error":
                        if data.get("prompt_id") == prompt_id:
                            yield ProgressUpdate(
                                type="error",
                                message=data.get("exception_message", "Unknown error"),
                            )
                            return
        
        except Exception as e:
            yield ProgressUpdate(type="error", message=str(e))
    
    async def get_outputs(self, prompt_id: str, output_dir: Path) -> list[str]:
        """Retrieve generated images for a completed prompt."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/history/{prompt_id}") as resp:
                if resp.status != 200:
                    return []
                history = await resp.json()
        
        if prompt_id not in history:
            return []
        
        outputs = history[prompt_id].get("outputs", {})
        image_paths = []
        
        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                
                # Download the image
                params = {"filename": filename, "subfolder": subfolder, "type": "output"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.url}/view", params=params) as resp:
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
        """Full generation pipeline: queue → wait → retrieve."""
        import time
        start = time.time()
        
        # Queue
        prompt_id = await self.queue_prompt(workflow)
        yield ProgressUpdate(type="queued", message=f"Prompt {prompt_id[:8]}")
        
        # Wait
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
