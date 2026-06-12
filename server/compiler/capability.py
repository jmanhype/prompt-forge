"""ComfyUI capability detection via /object_info."""
from __future__ import annotations

import aiohttp
from dataclasses import dataclass, field


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
    
    # Model availability
    available_checkpoints: list[str] = field(default_factory=list)
    available_loras: list[str] = field(default_factory=list)
    
    # Best strategy for this install
    @property
    def best_strategy(self) -> str:
        if self.has_gligen:
            return "gligen"
        if self.has_attention_couple:
            return "attention_couple"
        if self.has_ipadapter_region:
            return "ipadapter_region"
        return "mega_prompt"
    
    @property
    def strategy_description(self) -> str:
        descs = {
            "gligen": "GLIGEN native bbox conditioning (SD1.5)",
            "attention_couple": "Attention Couple regional (SDXL)",
            "ipadapter_region": "IPAdapter Regional (SDXL/Flux)",
            "mega_prompt": "Structured mega-prompt (no regional nodes)",
        }
        return descs.get(self.best_strategy, "Unknown")


# Node class names that indicate capability
NODE_SIGNATURES = {
    "has_gligen": ["GLIGENLoader", "GLIGEN"],
    "has_attention_couple": ["AttentionCouple", "AttentionCoupleBase"],
    "has_ipadapter_region": ["IPAdapterRegional", "IPAdapterAdvanced"],
    "has_flux_guidance": ["FluxGuidance", "FluxGuidanceApply"],
    "has_regional_condition": ["RegionalConditioning", "RegionalPrompter"],
}


async def probe_capabilities(comfyui_url: str) -> ComfyUICapabilities:
    """Query ComfyUI /object_info and detect available capabilities."""
    caps = ComfyUICapabilities()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{comfyui_url}/object_info") as resp:
                if resp.status != 200:
                    return caps
                caps.raw_nodes = await resp.json()
                caps.connected = True
    except Exception:
        return caps
    
    # Check for node signatures
    node_names = set(caps.raw_nodes.keys())
    for cap_name, signatures in NODE_SIGNATURES.items():
        if any(sig in node_names for sig in signatures):
            setattr(caps, cap_name, True)
    
    # Extract available models
    if "CheckpointLoaderSimple" in caps.raw_nodes:
        ckpt_info = caps.raw_nodes["CheckpointLoaderSimple"]
        caps.available_checkpoints = ckpt_info.get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
    
    if "LoraLoader" in caps.raw_nodes:
        lora_info = caps.raw_nodes["LoraLoader"]
        caps.available_loras = lora_info.get("input", {}).get("required", {}).get("lora_name", [[]])[0]
    
    return caps
