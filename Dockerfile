FROM python:3.10

WORKDIR /app

COPY . .

# Prevent buffering issues
ENV PYTHONUNBUFFERED=1

# Install deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn pydantic requests python-dotenv openai

# HF requires this
ENV PORT=7860

EXPOSE 7860

# 🔥 CRITICAL: use shell form for env variable
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT