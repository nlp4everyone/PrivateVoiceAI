from app.utils.config_loader import get_toml_config

# Load interaction settings from TOML
toml_config = get_toml_config()

asr_config = toml_config.get_section("asr")

# Model config
ASR_MODEL_NAME = asr_config.get("ASR_MODEL_NAME", "nvidia/parakeet-ctc-0.6b-vi")
ASR_DEVICE = asr_config.get("ASR_DEVICE", "auto")
SPLIT_MIXED_BATCH = asr_config.get("SPLIT_MIXED_BATCH", True)