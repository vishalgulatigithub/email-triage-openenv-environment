FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN printf '%s\n' \
'events {}' \
'http {' \
'  include /etc/nginx/mime.types;' \
'  server {' \
'    listen 7860;' \
'    client_max_body_size 100M;' \
'    location /api/ {' \
'      proxy_pass http://127.0.0.1:8000/;' \
'      proxy_http_version 1.1;' \
'      proxy_set_header Host $host;' \
'      proxy_set_header X-Forwarded-Prefix /api;' \
'    }' \
'    location / {' \
'      proxy_pass http://127.0.0.1:8501/;' \
'      proxy_http_version 1.1;' \
'      proxy_set_header Host $host;' \
'      proxy_set_header Upgrade $http_upgrade;' \
'      proxy_set_header Connection "upgrade";' \
'      proxy_read_timeout 86400;' \
'    }' \
'  }' \
'}' > /etc/nginx/nginx.conf

EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --root-path /api & streamlit run visualization/dashboard.py --server.port=8501 --server.address=127.0.0.1 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false & exec nginx -g 'daemon off;'"]