#!/usr/bin/env bash
# 把知选部署到一台装好 Docker 的服务器：同步运行文件 → 传 .env（600）→ 远端一键启动。
# 用法：./scripts/deploy_server.sh user@host [远端目录，默认 /opt/zhixuan]
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${1:?用法：$0 user@host [远端目录]}
DIR=${2:-/opt/zhixuan}
[ -f .env ] || { echo "[ERROR] 本地没有 .env，先 cp .env.example .env 并填写"; exit 1; }
echo "[INFO] 同步到 $TARGET:$DIR"
ssh "$TARGET" "mkdir -p $DIR"
rsync -az --delete --exclude '.git/' --exclude '.env*' --exclude 'var/' --exclude 'node_modules/' --exclude 'review/' --exclude 'screenshots/' --exclude 'data/raw/' \
  server.mjs package.json api lib public data docker docker-compose.yml scripts .env.example "$TARGET:$DIR/"
rsync -az --chmod=F600 .env "$TARGET:$DIR/.env"
ssh "$TARGET" "cd $DIR && ./scripts/start_all.sh --all"
echo "[SUCCESS] 完成：http://${TARGET#*@}"
