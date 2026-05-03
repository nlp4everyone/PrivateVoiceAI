from app.utils.config_loader import get_toml_config

# Load interaction settings from TOML
toml_config = get_toml_config()

asr_config = toml_config.get_section("asr")
vad_config = toml_config.get_section("vad")
ingress_config = toml_config.get_section("ingress")

# ASR deployment config
ASR_NUM_GPUS = asr_config.get("ASR_NUM_GPUS", 1)
ASR_NUM_REPLICAS = asr_config.get("ASR_NUM_REPLICAS", 1)
ASR_MAX_ONGOING_REQUESTS = asr_config.get("ASR_MAX_ONGOING_REQUESTS", 4)
ASR_MAX_BATCH_SIZE = asr_config.get("ASR_MAX_BATCH_SIZE", 4)
ASR_BATCH_WAIT_TIMEOUT_S = asr_config.get("ASR_BATCH_WAIT_TIMEOUT_S", 0.1)

# VAD deployment config
VAD_NUM_GPUS = vad_config.get("VAD_NUM_GPUS", 1)
VAD_NUM_REPLICAS = vad_config.get("VAD_NUM_REPLICAS", 1)
VAD_MAX_ONGOING_REQUESTS = vad_config.get("VAD_MAX_ONGOING_REQUESTS", 4)

# Ingress deployment config
INGRESS_NUM_GPUS = ingress_config.get("INGRESS_NUM_GPUS", 0)
INGRESS_NUM_REPLICAS = ingress_config.get("INGRESS_NUM_REPLICAS", 1)
INGRESS_MAX_ONGOING_REQUESTS = ingress_config.get("INGRESS_MAX_ONGOING_REQUESTS", 16)