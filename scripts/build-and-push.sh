#!/bin/bash
set -e

IMAGE="chronollm/sn38-validator"
VERSION="${1:-latest}"

echo "Building ${IMAGE}:${VERSION}"
cd ..
docker build --platform linux/amd64 -f Dockerfile.validator -t ${IMAGE}:${VERSION} .

echo "Pushing to DockerHub..."
docker push ${IMAGE}:${VERSION}

DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' ${IMAGE}:${VERSION} | cut -d@ -f2)
echo ""
echo "Image pushed: ${IMAGE}:${VERSION}"
echo "Digest: ${DIGEST}"
echo ""
echo "Add this digest to your backend ALLOWED_COMPOSE_HASHES:"
echo "  ${DIGEST}"
