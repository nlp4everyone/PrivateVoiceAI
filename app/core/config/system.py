from app.utils.config_loader import get_toml_config

# Load interaction settings from TOML
toml_config = get_toml_config()

system_config = toml_config.get_section("system")

# Model config
AUDIO_TEMP_DIR = system_config.get("AUDIO_TEMP_DIR", "/dev/shm")
VAD_DEPLOYMENT_NAME = system_config.get("VAD_DEPLOYMENT_NAME", "VADService")
RAY_HOST = system_config.get("RAY_HOST", "0.0.0.0")
RAY_PORT = system_config.get("RAY_PORT", 8000)