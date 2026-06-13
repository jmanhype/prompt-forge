"""ComfyUI capability detection via /object_info."""
from __future__ import annotations

import aiohttp
from dataclasses import dataclass, field

from ..patterns.registry import NodeRegistry


@dataclass
class ComfyUICapabilities:
    """Detected capabilities of the connected ComfyUI instance."""
    connected: bool = False
    raw_nodes: dict = field(default_factory=dict)
    
    # Regional conditioning strategies
    has_gligen: bool = False
    has_attention_couple: bool = False
    has_ipadapter_region: bool = False
    has_flux_guidance: bool = False
    has_regional_condition: bool = False
    
    # Model architecture support
    has_flux: bool = False
    has_ideogram4: bool = False
    flux_checkpoints: list[str] = field(default_factory=list)
    ideogram4_checkpoints: list[str] = field(default_factory=list)
    
    # Model availability
    available_checkpoints: list[str] = field(default_factory=list)
    available_loras: list[str] = field(default_factory=list)
    available_vaes: list[str] = field(default_factory=list)
    
    # Text encoder info
    ideogram4_clip: str = ""
    ideogram4_vae: str = ""
    
    @property
    def best_strategy(self) -> str:
        # Ideogram 4 preferred when LoRAs trained on it are present
        if self.has_ideogram4:
            return "ideogram4"
        if self.has_flux:
            return "flux"
        if self.has_gligen:
            return "gligen"
        return "mega_prompt"
    
    @property
    def strategy_description(self) -> str:
        descs = {
            "ideogram4": "Ideogram 4.0 + LoRA",
            "flux": "Flux Dev + mega-prompt + LoRA",
            "gligen": "GLIGEN native bbox conditioning (SD1.5)",
            "mega_prompt": "Structured mega-prompt (no regional nodes)",
        }
        return descs.get(self.best_strategy, "Unknown")


async def probe_capabilities(comfyui_url: str) -> ComfyUICapabilities:
    """Query ComfyUI /object_info and detect available capabilities."""
    caps = ComfyUICapabilities()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{comfyui_url}/object_info", timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return caps
                caps.raw_nodes = await resp.json()
                caps.connected = True
    except Exception:
        return caps
    
    # Check node signatures using NodeRegistry
    node_names = set(caps.raw_nodes.keys())
    node_registry = NodeRegistry()
    for cap_name in ["has_gligen", "has_attention_couple", "has_ipadapter_region", 
                     "has_flux_guidance", "has_regional_condition", "has_flux", "has_ideogram4"]:
        if node_registry.check_capability(cap_name, node_names):
            setattr(caps, cap_name, True)
    
    # Extract available checkpoints from CheckpointLoaderSimple
    if "CheckpointLoaderSimple" in caps.raw_nodes:
        ckpt_info = caps.raw_nodes["CheckpointLoaderSimple"]
        ckpts = ckpt_info.get("input", {}).get("required", {}).get("ckpt_name", [[]])
        caps.available_checkpoints = ckpts[0] if ckpts else []
    
    # Detect Ideogram 4 models from UNETLoader
    if "UNETLoader" in caps.raw_nodes:
        unet_info = caps.raw_nodes["UNETLoader"]
        unets = unet_info.get("input", {}).get("required", {}).get("unet_name", [[]])
        unet_names = unets[0] if unets else []
        caps.ideogram4_checkpoints = [u for u in unet_names if "ideogram" in u.lower()]
        caps.flux_checkpoints = [u for u in unet_names if "flux" in u.lower()]
    
    # Detect Ideogram 4 CLIP (CLIPLoader with type=ideogram4)
    if "CLIPLoader" in caps.raw_nodes:
        clip_info = caps.raw_nodes["CLIPLoader"]
        clip_names = clip_info.get("input", {}).get("required", {}).get("clip_name", [[]])[0]
        clip_types = clip_info.get("input", {}).get("required", {}).get("type", [[]])[0]
        
        if "ideogram4" in clip_types:
            caps.has_ideogram4 = True
            # Find the Qwen3VL clip for Ideogram 4
            for c in clip_names:
                if "qwen3vl" in c.lower() or "qwen_3" in c.lower():
                    caps.ideogram4_clip = c
                    break
        
        if "flux" in clip_types:
            caps.has_flux = True
    
    # Detect Ideogram 4 VAE (flux2-vae based on the symlink we found)
    if "VAELoader" in caps.raw_nodes:
        vae_info = caps.raw_nodes["VAELoader"]
        vaes = vae_info.get("input", {}).get("required", {}).get("vae_name", [[]])
        caps.available_vaes = vaes[0] if vaes else []
        # Ideogram 4 uses flux2-vae
        for v in caps.available_vaes:
            if "flux2" in v.lower():
                caps.ideogram4_vae = v
                break
    
    # Extract available LoRAs
    if "LoraLoader" in caps.raw_nodes:
        lora_info = caps.raw_nodes["LoraLoader"]
        loras = lora_info.get("input", {}).get("required", {}).get("lora_name", [[]])
        caps.available_loras = loras[0] if loras else []
    
    return caps
