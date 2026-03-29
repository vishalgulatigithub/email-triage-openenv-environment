FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

# FORCE INSTALL (bypass requirements.txt issue)
RUN pip install fastapi uvicorn pydantic requests python-dotenv openai

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]