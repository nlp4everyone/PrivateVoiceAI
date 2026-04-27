# Ray Component
from ray import serve
# Dependencies
import logging
# Config
from .core.config.serving import *
from .core.config.system import *
from .services.deployments.vad_deployment import VADService

logger = logging.getLogger("ray.serve")
# Bind the VAD service deployment
deployment = VADService.bind()

# Configure logger for Ray Serve
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger.info(f"{VAD_DEPLOYMENT_NAME} deployment bound with {NUM_REPLICAS} replica(s), {NUM_GPUS} GPU(s)")
# Start Ray Serve with HTTP server configuration
# The ingress app routes are automatically exposed at /v1/audio/vad
serve.start(
    detached=False,
    http_options={
        "host": RAY_HOST,
        "port": RAY_PORT
    }
)

