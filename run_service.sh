#!/bin/bash
# =============================================================================
# VAD Service Startup Script
# =============================================================================
# This script builds and starts the Voice Activity Detection (VAD) service
# using Docker Compose with the following configuration:
#   - Environment variables loaded from .env file
#   - Docker Compose configuration from docker/docker-compose.yml
#   - Builds the Docker image from scratch (--build)
#   - Removes orphaned containers from previous runs (--remove-orphans)
#   - Requires sudo privileges for Docker operations
# =============================================================================

sudo docker compose --env-file .env -f docker/docker-compose.yml up --build --remove-orphans
