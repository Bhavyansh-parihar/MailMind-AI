FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Metadata for OpenEnv
LABEL org.openenv.version="1.2.0"
LABEL org.openenv.name="MailMind AI"
LABEL org.openenv.type="real-world-simulation"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
