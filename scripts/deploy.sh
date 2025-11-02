#!/bin/bash
# Production Deployment Script
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-"production"}
COMPOSE_FILES="-f docker-compose.yml"

if [ "$ENVIRONMENT" = "production" ]; then
    COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.prod.yml"
    ENV_FILE=".env.production"
else
    ENV_FILE=".env"
fi

echo "Deploying Arthur Image Recognition to $ENVIRONMENT..."

# Pre-deployment checks
echo "Running pre-deployment checks..."

# Check if required files exist
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found!"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running!"
    exit 1
fi

# Load environment variables
set -a
source "$ENV_FILE"
set +a

# Backup current data (production only)
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Creating backup before deployment..."
    ./scripts/backup-qdrant.sh "pre-deploy-$(date +%Y%m%d-%H%M%S)" || echo "Backup failed, continuing..."
fi

# Pull latest images
echo "Pulling latest images..."
docker-compose $COMPOSE_FILES pull

# Build application image
echo "Building application..."
docker-compose $COMPOSE_FILES build --no-cache arthur-imgreco

# Deploy with zero-downtime strategy
echo "Deploying services..."

# Start infrastructure services first
docker-compose $COMPOSE_FILES up -d postgres redis qdrant

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Health check for Qdrant
until curl -f http://localhost:${QDRANT_PORT:-6333}/health > /dev/null 2>&1; do
    echo "Waiting for Qdrant to be ready..."
    sleep 5
done

# Health check for Redis
until docker-compose $COMPOSE_FILES exec -T redis redis-cli ping > /dev/null 2>&1; do
    echo "Waiting for Redis to be ready..."
    sleep 5
done

# Deploy application
docker-compose $COMPOSE_FILES up -d arthur-imgreco celery-worker

# Wait for application to be ready
echo "Waiting for application to be ready..."
sleep 15

# Health check for application
until curl -f http://localhost:${APP_PORT:-9000}/health > /dev/null 2>&1; do
    echo "Waiting for application to be ready..."
    sleep 5
done

# Start remaining services
docker-compose $COMPOSE_FILES up -d

# Verify deployment
echo "Verifying deployment..."
docker-compose $COMPOSE_FILES ps

# Run health check
echo "Running comprehensive health check..."
curl -s http://localhost:${APP_PORT:-9000}/api/health | jq '.' || echo "Health check failed"

echo "Deployment completed successfully!"
echo "Services are available at:"
echo "  Application: http://localhost:${APP_PORT:-9000}"
echo "  Qdrant: http://localhost:${QDRANT_PORT:-6333}"
echo "  Grafana: http://localhost:${GRAFANA_PORT:-3000}"
echo "  Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}"