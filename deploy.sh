docker buildx build --platform linux/arm64 --load -t verekia/v1v2-engine .
docker save verekia/v1v2-engine | gzip > /tmp/v1v2-engine.tar.gz
scp /tmp/v1v2-engine.tar.gz midgar:/tmp/
ssh midgar docker load --input /tmp/v1v2-engine.tar.gz
ssh midgar docker compose up -d v1v2-engine