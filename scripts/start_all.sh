#!/usr/bin/env bash
# 知选一键启动脚本：检查环境 → 准备 .env → 构建并启动容器 → 等待健康检查。
set -euo pipefail

VERSION="1.0.0"
cd "$(dirname "$0")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { printf "${BLUE}[INFO]${NC} %s\n" "$*"; }
warn()    { printf "${YELLOW}[WARNING]${NC} %s\n" "$*"; }
error()   { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
success() { printf "${GREEN}[SUCCESS]${NC} %s\n" "$*"; }

usage() {
  cat <<USAGE
知选启动脚本 v${VERSION}
用法: $0 [选项]
  -h, --help       显示帮助信息
  -a, --all        检查环境并启动所有服务（默认）
  -s, --stop       停止所有服务（保留数据卷）
  -c, --check      只检查环境，不启动
  -r, --restart    重新构建并重启（可指定服务：app 或 frontend）
  -l, --list       列出正在运行的容器
  -p, --pull       拉取最新的基础镜像
      --logs       跟随查看日志（Ctrl+C 只退出日志）
      --lite       不用 Docker，直接用 Node 单进程启动（需要 Node 22+）
  -v, --version    显示版本信息
USAGE
}

COMPOSE=""
check_docker() {
  if ! command -v docker >/dev/null 2>&1; then error "没有找到 docker 命令，请先安装 Docker（或改用 --lite）"; return 1; fi
  if docker compose version >/dev/null 2>&1; then COMPOSE="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then COMPOSE="docker-compose"
  else error "没有找到 docker compose 插件"; return 1; fi
  if ! docker info >/dev/null 2>&1; then error "Docker 守护进程没有运行，请先启动 Docker"; return 1; fi
  success "Docker 可用（$COMPOSE）"
}

env_get() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- || true; }

check_env_file() {
  if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    warn "没有 .env，已从 .env.example 复制一份；如需 Agent 讲解或 Supabase，请编辑 .env"
  fi
  local perm; perm=$(stat -f '%Lp' .env 2>/dev/null || stat -c '%a' .env)
  [ "$perm" = "600" ] || warn ".env 权限是 $perm，建议 chmod 600 .env（文件里有密钥）"
  [ -n "$(env_get AGENT_API_KEY)" ] && info "Agent 讲解：已启用" || info "Agent 讲解：未配置，使用内置文案（选品和预算判定不受影响）"
  if [ -n "$(env_get SUPABASE_URL)" ] && [ -n "$(env_get SUPABASE_SERVICE_ROLE_KEY)" ]; then info "数据存储：Supabase"; else info "数据存储：SQLite（app-data 卷）"; fi
}

check_port() {
  local port=$1 name=$2
  if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    if ! $COMPOSE ps --status running 2>/dev/null | grep -q zhixuan; then warn "端口 $port（$name）已被占用，可在 .env 里修改"; fi
  fi
}

check_data() {
  for f in data/catalog.json data/posts.json data/embeddings.json docker/Dockerfile.app docker/nginx/default.conf.template; do
    [ -f "$f" ] || { error "缺少 $f"; return 1; }
  done
  success "商品池与部署文件齐全"
}

wait_healthy() {
  info "等待应用通过健康检查…"
  for _ in $(seq 1 30); do
    if [ "$(docker inspect -f '{{.State.Health.Status}}' zhixuan-app 2>/dev/null)" = "healthy" ]; then success "应用已就绪"; return 0; fi
    sleep 2
  done
  error "应用 60 秒内没有就绪，查看日志：$COMPOSE logs app"; return 1
}

start_all() {
  check_docker; check_env_file; check_data
  check_port "${FRONTEND_PORT:-$(env_get FRONTEND_PORT)}" frontend
  info "构建并启动容器…"
  $COMPOSE up --build -d
  wait_healthy
  local port; port=$(env_get FRONTEND_PORT); port=${port:-80}
  success "知选已启动：http://localhost$([ "$port" = 80 ] || echo ":$port")"
  info "查看日志：$0 --logs    停止：$0 --stop"
}

start_lite() {
  command -v node >/dev/null 2>&1 || { error "没有找到 node，请安装 Node 22 或更高版本"; exit 1; }
  local major; major=$(node -p 'process.versions.node.split(".")[0]')
  [ "$major" -ge 22 ] || { error "需要 Node 22+，当前是 $(node -v)"; exit 1; }
  check_env_file; check_data
  info "以单进程方式启动（SQLite 文件：${DB_FILE:-var/app.db}）"
  exec node --env-file-if-exists=.env server.mjs
}

case "${1:---all}" in
  -h|--help) usage ;;
  -a|--all) start_all ;;
  -s|--stop) check_docker; $COMPOSE down --remove-orphans; success "已停止，数据卷 app-data 保留" ;;
  -c|--check) ok=0; check_docker || ok=1; check_env_file; check_data || ok=1; [ $ok = 0 ] && success "环境检查通过" || { error "环境检查未通过"; exit 1; } ;;
  -r|--restart) check_docker; $COMPOSE up --build -d --force-recreate ${2:-}; wait_healthy ;;
  -l|--list) check_docker; $COMPOSE ps ;;
  -p|--pull) check_docker; docker pull node:24-slim; docker pull nginx:1.27-alpine; success "基础镜像已更新，执行 $0 --restart 生效" ;;
  --logs) check_docker; $COMPOSE logs -f --tail=100 app frontend ;;
  --lite) start_lite ;;
  -v|--version) echo "知选启动脚本 v${VERSION}" ;;
  *) error "未知选项：$1"; usage; exit 1 ;;
esac
