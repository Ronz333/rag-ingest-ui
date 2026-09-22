import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import gradio as gr
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Pfad für lokale Module auflösen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from processors.processor_registry import ProcessorRegistry
except ImportError:
    # Fallback falls processors im übergeordneten Ordner liegt
    from processors.processor_registry import ProcessorRegistry

FILTER_FILE_PATH = "repo_filters.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==========================================
# FILTER & CONFIG MANAGEMENT
# ==========================================

def load_repo_filters(filters_path: str = FILTER_FILE_PATH) -> List[str]:
    """Lädt die Blacklist-Muster aus der JSON-Konfigurationsdatei."""
    if not os.path.exists(filters_path):
        return []
    try:
        with open(filters_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("ignore_patterns", [])
    except Exception as e:
        logging.error(f"Fehler beim Laden von {filters_path}: {e}")
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
    """Prüft, ob ein Dateipfad ein Blacklist-Muster aus repo_filters.json enthält."""
    normalized_path = file_path.replace("\\", "/")
    for pattern in ignore_patterns:
        if pattern in normalized_path:
            return True
    return False


# ==========================================
# EMBEDDING GENERATOR
# ==========================================

def get_embedding(
    text: str,
    provider: str,
    model_name: str,
    ollama_url: str,
    openai_api_key: str,
    vector_size: int
) -> List[float]:
    """Generiert Embeddings über den gewählten Provider oder liefert einen Dummy-Vektor."""
    if provider == "Ollama":
        try:
            url = f"{ollama_url.rstrip('/')}/api/embeddings"
            response = requests.post(url, json={"model": model_name, "prompt": text}, timeout=30)
            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                if len(embedding) == vector_size:
                    return embedding
        except Exception as e:
            logging.warning(f"Ollama Embedding Fehler: {e}")

    elif provider == "OpenAI" and openai_api_key:
        try:
            headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
            payload = {"input": text, "model": model_name}
            res = requests.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                return res.json()["data"][0]["embedding"]
        except Exception as e:
            logging.warning(f"OpenAI Embedding Fehler: {e}")

    # Fallback: Zero-Vector / Dummy Vector
    return [0.0] * vector_size


# ==========================================
# INGESTION PIPELINE CORE
# ==========================================

def run_ingestion_pipeline(
    target_dir: str,
    qdrant_url: str,
    collection_name: str,
    vector_size: float,
    distance_metric: str,
    embedding_provider: str,
    embedding_model: str,
    ollama_url: str,
    openai_api_key: str,
    chunk_size: float,
    chunk_overlap: float,
    min_chunk_len: float,
    dry_run: bool
) -> str:
    """
    Vollständige Ingestion-Pipeline mit optimiertem Payload (ohne text/document Duplikate),
    Qualitätsfilterung und automatischer Zuordnung über ProcessorRegistry.
    """
    if not os.path.exists(target_dir):
        return f"❌ Das Verzeichnis `{target_dir}` wurde nicht gefunden."

    v_size = int(vector_size)
    c_size = int(chunk_size)
    c_overlap = int(chunk_overlap)
    m_len = int(min_chunk_len)

    ignore_patterns = load_repo_filters()
    registry = ProcessorRegistry()

    # Parameter dynamisch an Prozessoren übergeben
    registry.default_processor.min_chunk_len = m_len
    registry.default_processor.chunk_size = c_size
    registry.default_processor.chunk_overlap = c_overlap

    all_payloads = []
    file_stats = {"total": 0, "processed": 0, "skipped": 0}

    # Repository scannen
    file_paths = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            file_paths.append(os.path.join(root, f))

    file_stats["total"] = len(file_paths)
    logs = [f"🔍 Verzeichnis-Scan gestartet in: {target_dir}", f"📁 Gesamte Dateien gefunden: {len(file_paths)}"]

    for full_path in file_paths:
        # 1. Pfad-Blacklist Filter
        if is_blacklisted(full_path, ignore_patterns):
            file_stats["skipped"] += 1
            continue

        processor = registry.get_processor(full_path)

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 2. Datei verarbeiten & Chunks generieren
            payloads = processor.process(content, full_path)
            
            # 3. Qualitätssicherung: Chunks filtern (Länge & Struktur)
            valid_payloads = []
            for p in payloads:
                chunk_text = p.get("content", "").strip()
                
                # Minimum Chunk-Länge prüfen
                if len(chunk_text) < m_len:
                    continue
                    
                # LISP/Syntaktischen Müll filtern (z.B. reine Klammerzeilen)
                if chunk_text.startswith("(") and chunk_text.endswith(")") and len(chunk_text.split()) <= 2:
                    continue

                valid_payloads.append(p)

            if valid_payloads:
                all_payloads.extend(valid_payloads)
                file_stats["processed"] += 1
            else:
                file_stats["skipped"] += 1

        except Exception as e:
            logging.error(f"Fehler bei {full_path}: {e}")
            file_stats["skipped"] += 1

    logs.append("\n=== INGESTION ZUSAMMENFASSUNG ===")
    logs.append(f"Verarbeitete Dateien: {file_stats['processed']}")
    logs.append(f"Übersprungene Dateien: {file_stats['skipped']}")
    logs.append(f"Generierte valide Chunks: {len(all_payloads)}")

    if dry_run:
        logs.append("\nℹ️ DRY RUN BEENDET: Es wurden keine Daten nach Qdrant hochgeladen.")
        return "\n".join(logs)

    if not all_payloads:
        logs.append("\n⚠️ Keine gültigen Chunks erzeugt. Upload wird abgebrochen.")
        return "\n".join(logs)

    # 4. Qdrant Upload Phase
    logs.append(f"\n📤 Verbinde mit Qdrant unter {qdrant_url}...")
    try:
        client = QdrantClient(url=qdrant_url)
        
        # Distanzmetrik zuweisen
        dist = Distance.COSINE
        if distance_metric == "Euclidean":
            dist = Distance.EUCLID
        elif distance_metric == "Dot":
            dist = Distance.DOT

        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            logs.append(f"⚙️ Erstelle neue Collection `{collection_name}` (Dimension: {v_size})...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=v_size, distance=dist)
            )

        logs.append(f"⏳ Erzeuge Embeddings via `{embedding_provider}` und lade Punkte hoch...")
        
        batch_size = 50
        points = []
        
        for idx, payload in enumerate(all_payloads):
            chunk_content = payload.get("content", "")
            
            # Vector erzeugen
            vector = get_embedding(
                text=chunk_content,
                provider=embedding_provider,
                model_name=embedding_model,
                ollama_url=ollama_url,
                openai_api_key=openai_api_key,
                vector_size=v_size
            )

            # Schlankes Payload-Objekt an Qdrant senden (ohne 'text' / 'document' Duplikate)
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload
                )
            )

            if len(points) >= batch_size or idx == len(all_payloads) - 1:
                client.upsert(collection_name=collection_name, points=points)
                points = []

        logs.append(f"\n🎉 ERFOLG! {len(all_payloads)} Chunks wurden erfolgreich in Qdrant gespeichert.")

    except Exception as e:
        logs.append(f"\n❌ QDRANT ERROR: {e}")

    return "\n".join(logs)


def search_qdrant_inspection(
    qdrant_url: str,
    collection_name: str,
    query_text: str,
    limit: float,
    embedding_provider: str,
    embedding_model: str,
    ollama_url: str,
    openai_api_key: str,
    vector_size: float
) -> str:
    """Sucht in Qdrant und zeigt die abgerufenen Payloads zur Qualitätskontrolle an."""
    try:
        client = QdrantClient(url=qdrant_url)
        v_size = int(vector_size)
        
        query_vector = get_embedding(
            text=query_text,
            provider=embedding_provider,
            model_name=embedding_model,
            ollama_url=ollama_url,
            openai_api_key=openai_api_key,
            vector_size=v_size
        )

        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=int(limit)
        )

        output = []
        for i, res in enumerate(results):
            output.append(f"--- ERGEBNIS {i+1} (Score: {res.score:.4f}) ---")
            output.append(json.dumps(res.payload, indent=2, ensure_ascii=False))
            output.append("\n")

        return "\n".join(output) if output else "Keine Treffer gefunden."
    except Exception as e:
        return f"❌ Fehler bei Suche: {e}"


# ==========================================
# GRADIO UI GUI BUILDER
# ==========================================

def build_app():
    with gr.Blocks(title="RAG Ingest Pipeline - Autonomous PCB Generator") as app:
        gr.Markdown("# ⚡ RAG Ingest UI - Autonomous PCB Generator Pipeline")
        gr.Markdown(
            "Verarbeitet Hardware-Repositories, filtert physikalischen Layout-Müll "
            "und erstellt schlanke, hochqualitative Payloads für Qdrant."
        )

        with gr.Tab("🚀 Ingestion Run"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Pfad & Repository")
                    target_dir = gr.Textbox(label="Ziel-Verzeichnis zum Scannen", value="./src")
                    dry_run = gr.Checkbox(label="Dry Run (Analyse ohne Qdrant Upload)", value=False)

                    gr.Markdown("### ⚙️ Qdrant Konfiguration")
                    qdrant_url = gr.Textbox(label="Qdrant URL", value="http://localhost:6333")
                    collection_name = gr.Textbox(label="Collection Name", value="hardware_docs")
                    vector_size = gr.Number(label="Vektor-Dimension", value=384, precision=0)
                    distance_metric = gr.Dropdown(label="Distanzmetrik", choices=["Cosine", "Euclidean", "Dot"], value="Cosine")

                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Embedding Model")
                    embedding_provider = gr.Dropdown(label="Provider", choices=["Ollama", "OpenAI", "None (Dummy)"], value="Ollama")
                    embedding_model = gr.Textbox(label="Model Name", value="nomic-embed-text")
                    ollama_url = gr.Textbox(label="Ollama Base URL", value="http://localhost:11434")
                    openai_api_key = gr.Textbox(label="OpenAI API Key (falls genutzt)", type="password")

                    gr.Markdown("### ✂️ Chunking & Qualitäts-Filter")
                    chunk_size = gr.Number(label="Chunk Size (Chars)", value=1000, precision=0)
                    chunk_overlap = gr.Number(label="Chunk Overlap (Chars)", value=100, precision=0)
                    min_chunk_len = gr.Number(label="Min Chunk Length (Chars)", value=60, precision=0)

            run_btn = gr.Button("🚀 Pipeline Ausführen", variant="primary")
            output_logs = gr.Textbox(label="Ausgabe & Ingestion Logs", lines=15)

            run_btn.click(
                fn=run_ingestion_pipeline,
                inputs=[
                    target_dir,
                    qdrant_url,
                    collection_name,
                    vector_size,
                    distance_metric,
                    embedding_provider,
                    embedding_model,
                    ollama_url,
                    openai_api_key,
                    chunk_size,
                    chunk_overlap,
                    min_chunk_len,
                    dry_run
                ],
                outputs=output_logs
            )

        with gr.Tab("🛡️ Filterliste (repo_filters.json)"):
            gr.Markdown("### Müll- und Build-Filter verwalten")
            current_filters = "\n".join(load_repo_filters())
            filters_input = gr.Textbox(label="Musterliste (Ein Eintrag pro Zeile)", value=current_filters, lines=18)
            save_btn = gr.Button("💾 Filterliste Speichern", variant="primary")
            filter_status = gr.Markdown()

            save_btn.click(
                fn=save_repo_filters,
                inputs=filters_input,
                outputs=filter_status
            )

        with gr.Tab("🔍 Qdrant Inspector & Suche"):
            gr.Markdown("### Suche & Payload-Struktur überprüfen")
            with gr.Row():
                search_query = gr.Textbox(label="Suchanfrage / Prompt", value="ESP32 Power Supply Circuit")
                search_limit = gr.Slider(label="Anzahl Ergebnisse", minimum=1, maximum=20, value=3, step=1)
            
            search_btn = gr.Button("🔎 Qdrant Durchsuchen")
            search_results = gr.Textbox(label="Gefundene Payloads (JSON)", lines=18)

            search_btn.click(
                fn=search_qdrant_inspection,
                inputs=[
                    qdrant_url,
                    collection_name,
                    search_query,
                    search_limit,
                    embedding_provider,
                    embedding_model,
                    ollama_url,
                    openai_api_key,
                    vector_size
                ],
                outputs=search_results
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
