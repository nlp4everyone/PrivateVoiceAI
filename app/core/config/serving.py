from app.utils.config_loader import get_toml_config

# Load interaction settings from TOML
toml_config = get_toml_config()

serving_config = toml_config.get_section("serving")

# Model config
NUM_GPUS = serving_config.get("NUM_GPUS", 1)
NUM_REPLICAS = serving_config.get("NUM_REPLICAS", 1)
MAX_ONGOING_REQUESTS = serving_config.get("MAX_ONGOING_REQUESTS", 16)
MAX_BATCH_SIZE = serving_config.get("MAX_BATCH_SIZE", 8)
BATCH_WAIT_TIMEOUT_S = serving_config.get("BATCH_WAIT_TIMEOUT_S", 0.1)