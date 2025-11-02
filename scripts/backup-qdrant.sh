#!/bin/bash
# Qdrant Backup Script
# Usage: ./backup-qdrant.sh [backup-name]

set -e

BACKUP_NAME=${1:-"qdrant-backup-$(date +%Y%m%d-%H%M%S)"}
BACKUP_DIR="./backups/qdrant"
CONTAINER_NAME="arthur-imgreco_qdrant_1"

echo "Starting Qdrant backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Method 1: Docker volume backup (recommended)
echo "Creating docker volume backup..."
docker run --rm -v arthur-imgreco_qdrant_data:/source -v "$(pwd)/$BACKUP_DIR":/backup alpine \
  tar czf "/backup/$BACKUP_NAME.tar.gz" -C /source .

# Method 2: API-based backup (if needed)
echo "Creating API snapshot..."
curl -X POST "http://localhost:6333/collections/arthur_images/snapshots" \
  -H "Content-Type: application/json" \
  -d '{"name": "'$BACKUP_NAME'"}' || echo "API snapshot failed (container might be down)"

echo "Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
echo "Backup size: $(du -h "$BACKUP_DIR/$BACKUP_NAME.tar.gz" | cut -f1)"

# Clean up old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete || true

echo "Backup script completed successfully!"