FROM python:3.11-slim

# Unbuffered Output für sauberes Live-Logging im Docker-Protokoll
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Git installieren für Repositories
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# PIP Pakete installieren
RUN pip install --no-cache-dir gradio ollama qdrant-client pypdf

# Quellcode und den neuen Plugin-Ordner kopieren
COPY processors/ /app/processors/
COPY rag_ingest_app.py /app/rag_ingest_app.py

EXPOSE 7861

CMD ["python", "rag_ingest_app.py"]
