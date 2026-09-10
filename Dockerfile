FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir gradio ollama qdrant-client pypdf
COPY rag_ingest_app.py /app/rag_ingest_app.py
EXPOSE 7861
CMD ["python", "rag_ingest_app.py"]
