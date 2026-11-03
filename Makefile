.PHONY: start-all stop restart check list logs lite test bench

start-all: ## 检查环境并用 Docker 启动
	./scripts/start_all.sh --all
stop:
	./scripts/start_all.sh --stop
restart:
	./scripts/start_all.sh --restart
check:
	./scripts/start_all.sh --check
list:
	./scripts/start_all.sh --list
logs:
	./scripts/start_all.sh --logs
lite: ## 不用 Docker，Node 单进程启动
	./scripts/start_all.sh --lite
test:
	npm test && npm run bench && npm run scan:copy
