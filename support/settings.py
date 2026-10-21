
from dataclasses import dataclass
from pathlib import Path
import json

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass(frozen=True)
class ModelConfig:

    model_name: str
    batch_size: int
    device: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Model name must be a non-empty string")
        if (isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) 
            or self.batch_size < 1):
            raise ValueError("Batch size must be a positive integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Device must be a non-empty string")


def load_settings(path: Path = CONFIG_PATH) -> ModelConfig:
    
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)              
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object, got {type(raw).__name__}")
    try:
        return ModelConfig(model_name=raw["model_name"], batch_size=raw["batch_size"], device=raw["device"])
    except KeyError as err:
        raise ValueError(f"Missing key in config file: {err}") from err

