"""Progress forwarding — bridge ComfyUI WS to frontend WS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ForgeProgress:
    """Unified progress format for the frontend."""
    session_id: str
    phase: str  # "analyzing" | "compiling" | "generating" | "scoring" | "mutating"
    iteration: int = 0
    max_iterations: int = 0
    progress: float = 0.0  # 0-1 within current phase
    message: str = ""
    
    def to_ws_message(self) -> dict:
        return {
            "type": self.phase,
            "data": {
                "session_id": self.session_id,
                "iteration": self.iteration,
                "max_iterations": self.max_iterations,
                "progress": self.progress,
                "message": self.message,
            }
        }
