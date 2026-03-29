FROM python:3.10

WORKDIR /app

COPY . .

# 🔥 DEBUG: check file exists
RUN ls -l

# 🔥 DEBUG: print contents
RUN cat requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]