# 知选应用镜像：Node 单进程，无第三方依赖。数据库文件放在 /data（挂载卷）。
FROM node:24-slim
WORKDIR /app
COPY package.json server.mjs ./
COPY api ./api
COPY lib ./lib
COPY public ./public
COPY data/catalog.json data/posts.json data/embeddings.json ./data/
RUN mkdir -p /data && chown node:node /data
ENV NODE_ENV=production HOST=0.0.0.0 PORT=8080 DB_FILE=/data/app.db TRUST_PROXY=1
VOLUME ["/data"]
EXPOSE 8080
USER node
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD node -e "fetch('http://127.0.0.1:8080/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "server.mjs"]
