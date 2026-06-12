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
    
    # Flux support
    has_flux: bool = False
    flux_checkpoints: list[str] = field(default_factory=list)
    
    # Model availability
    available_checkpoints: list[str] = field(default_factory=list)
    available_loras: list[str] = field(default_factory=list)
    available_vaes: list[str] = field(default_factory=list)
    
    @property
    def best_strategy(self) -> str:
        if self.has_flux:
            return "flux"
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
            "flux": "Flux Dev + mega-prompt + LoRA (best quality)",
            "gligen": "GLIGEN native bbox conditioning (SD1.5)",
            "attention_couple": "Attention Couple regional (SDXL)",
            "ipadapter_region": "IPAdapter Regional (SDXL/Flux)",
            "mega_prompt": "Structured mega-prompt (no regional nodes)",
        }
        return descs.get(self.best_strategy, "Unknown")


# Node class names that indicate capability
NODE_SIGNATURES = {
    "has_gligen": ["GLIGENLoader", "GLIGENTextBoxApply"],
    "has_attention_couple": ["AttentionCouple", "AttentionCoupleBase"],
    "has_ipadapter_region": ["IPAdapterRegionalConditioning", "IPAdapterRegional"],
    "has_flux_guidance": ["FluxGuidance", "FluxDisableGuidance"],
    "has_regional_condition": ["RegionalConditioning", "RegionalPrompter", "RegionalSampler"],
    "has_flux": ["CLIPTextEncodeFlux", "ModelSamplingFlux", "DualCLIPLoader"],
}


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
    
    # Check for node signatures
    node_names = set(caps.raw_nodes.keys())
    for cap_name, signatures in NODE_SIGNATURES.items():
        if any(sig in node_names for sig in signatures):
            setattr(caps, cap_name, True)
    
    # Extract available checkpoints
    if "CheckpointLoaderSimple" in caps.raw_nodes:
        ckpt_info = caps.raw_nodes["CheckpointLoaderSimple"]
        ckpts = ckpt_info.get("input", {}).get("required", {}).get("ckpt_name", [[]])
        caps.available_checkpoints = ckpts[0] if ckpts else []
        
        # Detect Flux checkpoints
        caps.flux_checkpoints = [c for c in caps.available_checkpoints if "flux" in c.lower()]
    
    # Also check UNETLoader for Flux models
    if "UNETLoader" in caps.raw_nodes:
        unet_info = caps.raw_nodes["UNETLoader"]
        unets = unet_info.get("input", {}).get("required", {}).get("unet_name", [[]])
        unet_names = unets[0] if unets else []
        flux_unets = [u for u in unet_names if "flux" in u.lower()]
        if flux_unets:
            caps.has_flux = True
            caps.flux_checkpoints.extend(flux_unets)
    
    # Extract available LoRAs
    if "LoraLoader" in caps.raw_nodes:
        lora_info = caps.raw_nodes["LoraLoader"]
        loras = lora_info.get("input", {}).get("required", {}).get("lora_name", [[]])
        caps.available_loras = loras[0] if loras else []
    
    # Extract available VAEs
    if "VAELoader" in caps.raw_nodes:
        vae_info = caps.raw_nodes["VAELoader"]
        vaes = vae_info.get("input", {}).get("required", {}).get("vae_name", [[]])
        caps.available_vaes = vaes[0] if vaes else []
    
    return caps
