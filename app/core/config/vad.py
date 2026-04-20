from app.utils.config_loader import get_toml_config
import os

# Load interaction settings from TOML
toml_config = get_toml_config()

vad_config = toml_config.get_section("vad")

# VAD config
VAD_MODEL_NAME = vad_config.get("VAD_MODEL_NAME", "pyannote/segmentation-3.0")
VAD_DEVICE = vad_config.get("VAD_DEVICE", "auto")

# Token
HF_TOKEN = os.getenv("HF_TOKEN","")
