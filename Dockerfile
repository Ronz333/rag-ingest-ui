FROM python:3.11-slim

# Systemabhängigkeiten für Git, C-Builds und Dateiverarbeitung
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-Abhängigkeiten
RUN pip install --no-cache-dir \
    gradio>=4.0.0 \
    ollama \
    qdrant-client \
    pypdf \
    skidl \
    sexpdata

# Anwendungs-Code kopieren
COPY . /app

EXPOSE 7861

CMD ["python", "rag_ingest_app.py"]
