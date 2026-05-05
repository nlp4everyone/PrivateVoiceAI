# Ray Component
from ray import serve
# Dependencies
import logging
# Config
from .core.config.serving import *
from .core.config.system import *
from .services.deployments.ingress_deployment import bind_app

logger = logging.getLogger("ray.serve")

# Bind the deployment graph (VAD -> ASR -> Ingress)
deployment = bind_app()

# Configure logger for Ray Serve
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger.info(f"Ingress deployment bound with {INGRESS_NUM_GPUS} replica(s), {INGRESS_NUM_GPUS} GPU(s)")

# Start Ray Serve with HTTP server configuration
# The ingress app routes are automatically exposed at /v1/audio/transcriptions
serve.start(
    detached=False,
    http_options={
        "host": RAY_HOST,
        "port": RAY_PORT
    }
)

