import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ComfyUI
    COMFYUI_URL: str = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
    COMFYUI_ROOT: str = os.getenv("COMFYUI_ROOT", "")
    
    # Florence-2
    FLORENCE_MODEL: str = os.getenv("FLORENCE_MODEL", "microsoft/Florence-2-base-ft")
    
    # Optional Qwen-VL
    QWEN_VL_ENABLED: bool = os.getenv("QWEN_VL_ENABLED", "false").lower() == "true"
    QWEN_VL_URL: str = os.getenv("QWEN_VL_URL", "")
    QWEN_VL_MODEL: str = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")
    
    # Forge
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    CONVERGENCE_THRESHOLD: float = float(os.getenv("CONVERGENCE_THRESHOLD", "0.85"))
    FORGE_PORT: int = int(os.getenv("FORGE_PORT", "7861"))
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "./data")
    OUTPUTS_DIR: Path = DATA_DIR / "outputs"
    CACHE_DIR: Path = DATA_DIR / "cache"
    DB_PATH: Path = DATA_DIR / "forge.db"
    TEMPLATES_DIR: Path = PROJECT_ROOT / "workflows" / "templates"
    FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"
    
    @classmethod
    def ensure_dirs(cls):
        for d in [cls.DATA_DIR, cls.OUTPUTS_DIR, cls.CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    @property
    def loras_dir(self) -> Path:
        if self.COMFYUI_ROOT:
            return Path(self.COMFYUI_ROOT) / "models" / "loras"
        return Path("/dev/null")  # won't find anything

config = Config()
