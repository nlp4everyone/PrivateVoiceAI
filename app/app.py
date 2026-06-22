# Ray Component
import ray
from ray import serve
# Dependencies
import logging

# Suppress NeMo and its dependencies before any NeMo import occurs.
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
for _logger in ("nemo", "nemo_logger", "lightning", "pytorch_lightning",
                "filelock", "datasets", "huggingface_hub"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

# Config
from .core.config.serving import *
from .core.config.system import *
from .services.deployments.asr_deployment import ASRService

logging.getLogger("ray.serve").setLevel(logging.INFO)

ray.init(logging_level=logging.WARNING)

logger = logging.getLogger("ray.serve")
# Bind the ASR service deployment
deployment = ASRService.bind()

logger.info(f"{DEPLOYMENT_NAME} deployment bound with {NUM_REPLICAS} replica(s), {NUM_GPUS} GPU(s)")
# Start Ray Serve with HTTP server configuration
# The ingress app routes are automatically exposed at /v1/audio/transcriptions
serve.start(
    detached=False,
    http_options={
        "host": RAY_HOST,
        "port": RAY_PORT
    }
)

