import os
import json
from typing import List, Dict, Any
import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from processors.processor_registry import ProcessorRegistry

FILTER_FILE_PATH = "repo_filters.json"


# ==========================================
# FILTER & PIPELINE LOGIK
# ==========================================

def load_repo_filters(filters_path: str = FILTER_FILE_PATH) -> List[str]:
    """Lädt die Blacklist-Muster aus der JSON-Konfigurationsdatei."""
    if not os.path.exists(filters_path):
        return []
    try:
        with open(filters_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("ignore_patterns", [])
    except Exception:
        return []


def save_repo_filters(filters_text: str, filters_path: str = FILTER_FILE_PATH) -> str:
    """Speichert die Blacklist-Muster aus dem Gradio-Textfeld."""
    try:
        patterns = [line.strip() for line in filters_text.split("\n") if line.strip()]
        with open(filters_path, "w", encoding="utf-8") as f:
            json.dump({"ignore_patterns": patterns}, f, indent=2, ensure_ascii=False)
        return "✅ Filterliste erfolgreich in `repo_filters.json` gespeichert!"
    except Exception as e:
        return f"❌ Fehler beim Speichern: {e}"


def is_blacklisted(file_path: str, ignore_patterns: List[str]) -> bool:
    """Prüft, ob ein Dateipfad ein Blacklist-Muster enthält."""
    normalized_path = file_path.replace("\\", "/")
    for pattern in ignore_patterns:
        if pattern in normalized_path:
            return True
    return False


def run_ingestion_pipeline(
    target_dir: str,
    qdrant_url: str,
    collection_name: str,
    vector_size: float,
    chunk_size: float,
    chunk_overlap: float,
    min_chunk_len: float,
    dry_run: bool
) -> str:
    """Führt den Ingestion-Run aus und gibt Statusmeldungen für die UI zurück."""
    if not os.path.exists(target_dir):
        return f"❌ Das Verzeichnis `{target_dir}` wurde nicht gefunden."

    ignore_patterns = load_repo_filters()
    registry = ProcessorRegistry()

    # Prozessoren dynamisch mit den UI-Parametern konfigurieren
    registry.default_processor.min_chunk_len = int(min_chunk_len)
    registry.default_processor.chunk_size = int(chunk_size)
    registry.default_processor.chunk_overlap = int(chunk_overlap)

    all_payloads = []
    file_stats = {"total": 0, "processed": 0, "skipped": 0}

    file_paths = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            file_paths.append(os.path.join(root, f))

    file_stats["total"] = len(file_paths)

    for full_path in file_paths:
        if is_blacklisted(full_path, ignore_patterns):
            file_stats["skipped"] += 1
            continue

        processor = registry.get_processor(full_path)

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Verarbeitet die Datei und baut das schlanke Payload
            payloads = processor.process(content, full_path)
            if payloads:
                all_payloads.extend(payloads)
                file_stats["processed"] += 1
            else:
                file_stats["skipped"] += 1

        except Exception:
            file_stats["skipped"] += 1

    summary = (
        f"=== INGESTION ZUSAMMENFASSUNG ===\n"
        f"Gefundene Dateien: {file_stats['total']}\n"
        f"Verarbeitete Dateien: {file_stats['processed']}\n"
        f"Übersprungene Dateien: {file_stats['skipped']}\n"
        f"Generierte valide Chunks: {len(all_payloads)}\n\n"
    )

    if dry_run or not all_payloads:
        summary += "ℹ️ Dry Run beendet (Keine Vektoren an Qdrant gesendet)."
        return summary

    # Qdrant Upload Phase
    try:
        client = QdrantClient(url=qdrant_url)
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=int(vector_size), distance=Distance.COSINE)
            )

        batch_size = 50
        points = []
        for idx, payload in enumerate(all_payloads):
            # Platzhalter-Vektor (oder durch deine Embedding-Funktion ersetzen)
            dummy_vector = [0.0] * int(vector_size)

            points.append(
                PointStruct(
                    id=idx,
                    vector=dummy_vector,
                    payload=payload  # Enthält nur 'content' & Metadaten, keine Redundanzen
                )
            )

            if len(points) >= batch_size or idx == len(all_payloads) - 1:
                client.upsert(collection_name=collection_name, points=points)
                points = []

        summary += f"🚀 ERFOLG: {len(all_payloads)} Chunks wurden erfolgreich nach Qdrant hochgeladen!"
    except Exception as e:
        summary += f"❌ Fehler beim Upload nach Qdrant: {e}"

    return summary


# ==========================================
# GRADIO UI OBERFLÄCHE
# ==========================================

def build_ui():
    with gr.Blocks(title="RAG Ingest Pipeline - PCB Generator") as app:
        gr.Markdown("# ⚡ RAG Ingest UI - Autonomous PCB Generator Pipeline")
        gr.Markdown(
            "Verarbeitet Hardware-Repositories, filtert physikalischen Layout-Müll "
            "und erstellt schlanke Payloads für Qdrant."
        )

        with gr.Tab("🚀 Ingestion Run"):
            with gr.Row():
                with gr.Column():
                    target_dir = gr.Textbox(label="Ziel-Verzeichnis zum Scannen", value="./src")
                    qdrant_url = gr.Textbox(label="Qdrant URL", value="http://localhost:6333")
                    collection_name = gr.Textbox(label="Collection Name", value="hardware_docs")
                    vector_size = gr.Number(label="Vektor-Dimension", value=384, precision=0)
                with gr.Column():
                    chunk_size = gr.Number(label="Chunk Size (Chars)", value=1000, precision=0)
                    chunk_overlap = gr.Number(label="Chunk Overlap (Chars)", value=100, precision=0)
                    min_chunk_len = gr.Number(label="Min Chunk Length (Chars)", value=60, precision=0)
                    dry_run = gr.Checkbox(label="Dry Run (Nur Analyse)", value=False)

            run_btn = gr.Button("🚀 Pipeline Starten", variant="primary")
            output_text = gr.Textbox(label="Ergebnis & Logs", lines=12)

            run_btn.click(
                fn=run_ingestion_pipeline,
                inputs=[
                    target_dir,
                    qdrant_url,
                    collection_name,
                    vector_size,
                    chunk_size,
                    chunk_overlap,
                    min_chunk_len,
                    dry_run
                ],
                outputs=output_text
            )

        with gr.Tab("🛡️ Filterliste (repo_filters.json)"):
            current_filters = "\n".join(load_repo_filters())
            filters_input = gr.Textbox(label="Musterliste (Ein Eintrag pro Zeile)", value=current_filters, lines=15)
            save_btn = gr.Button("💾 Filterliste Speichern")
            filter_status = gr.Markdown()

            save_btn.click(
                fn=save_repo_filters,
                inputs=filters_input,
                outputs=filter_status
            )

    return app


if __name__ == "__main__":
    app = build_ui()
    # Startet auf Port 7860 (Standard für Gradio & Docker)
    app.launch(server_name="0.0.0.0", server_port=7860)
