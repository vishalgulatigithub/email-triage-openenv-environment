FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install fastapi uvicorn pydantic requests python-dotenv openai

ENV PORT=7860

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT