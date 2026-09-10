import os
import re
import uuid
import gradio as gr
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader

# Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_1")
EMBED_MODEL = "bge-m3"
COLLECTION_NAME = "pcb_knowledge_base"

# Clients initialisieren
ollama_client = ollama.Client(host=OLLAMA_HOST)
qdrant_client = QdrantClient(url=QDRANT_HOST)

def ensure_qdrant_collection(vector_size: int):
    """Erstellt die Qdrant-Collection bei Bedarf mit korrekter Vektordimension."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def extract_text_from_file(file_path: str) -> str:
    """Liets Text aus PDF-, Python-, Markdown- und Textdateien aus."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def process_and_ingest(files):
    if not files:
        yield "❌ Bitte mindestens eine Datei hochladen."
        return

    status_log = f"🚀 Starte Ingestion-Pipeline ({len(files)} Datei(en))\n"
    status_log += f"Embedding-Modell: {EMBED_MODEL} | Qdrant: {QDRANT_HOST}\n\n"
    yield status_log

    for idx, file_obj in enumerate(files, 1):
        filename = os.path.basename(file_obj.name)
        status_log += f"[{idx}/{len(files)}] Verarbeite: {filename}\n"
        yield status_log

        # 1. Text extrahieren
        raw_text = extract_text_from_file(file_obj.name)
        if not raw_text.strip():
            status_log += f"   ⚠️ Datei leer oder ungültig. Übersprungen.\n"
            yield status_log
            continue

        # 2. LLM-Analyse & Strukturierung
        status_log += f"   ↳ Strukturierung via LLM ({LLM_MODEL})...\n"
        yield status_log

        prompt = f"""
        Du bist ein Ingestion-Agent für ein EDA/PCB-RAG-System.
        Analysiere den folgenden Inhalt und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

        INHALT:
        {raw_text[:6000]}

        REGELN:
        1. Generiere in der ERSTEN Zeile ein exaktes Kategorie-Schlagwort in eckigen Klammern, z.B. [TAG: SKIDL_API], [TAG: KICAD_PCBNEW], [TAG: SPICE_SIM].
        2. Fasse den Zweck kurz zusammen.
        3. Extrahiere oder erstelle ein lauffähiges Codebeispiel oder eine präzise Konfigurationsanleitung.
        """

        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        processed_md = response['message']['content']

        # Tag aus der ersten Zeile extrahieren
        tag_match = re.search(r'\[TAG:\s*([A-Z0-9_]+)\]', processed_md, re.IGNORECASE)
        category_tag = tag_match.group(1).upper() if tag_match else "GENERAL_EDA"

        # 3. Multilinguales Embedding mit bge-m3 erzeugen
        status_log += f"   ↳ Generiere Vektor-Embedding ({EMBED_MODEL})...\n"
        yield status_log

        embed_res = ollama_client.embeddings(model=EMBED_MODEL, prompt=processed_md)
        vector = embed_res['embedding']

        # 4. Speichern in Qdrant
        ensure_qdrant_collection(len(vector))

        point_id = str(uuid.uuid4())
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "filename": filename,
                        "category_tag": category_tag,
                        "content": processed_md
                    }
                )
            ]
        )

        status_log += f"   ✅ Erfasst in '{COLLECTION_NAME}' | **Schlagwort: #{category_tag}**\n\n"
        yield status_log

    status_log += "🎉 Ingestion abgeschlossen! Dokumente sind für den PCB-Agenten verfügbar."
    yield status_log

# UI Definition
with gr.Blocks(title="RAG Knowledge Ingest") as demo:
    gr.Markdown("# 📥 RAG Knowledge Ingestion Pipeline")
    gr.Markdown("Automatisierte Aufbereitung, Multilinguale Vektorisierung (`bge-m3`) und Ablage in Qdrant.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Quelldaten hochladen",
                file_count="multiple",
                file_types=[".pdf", ".py", ".md", ".txt", ".json"]
            )
            start_btn = gr.Button("🚀 Ingestion Starte", variant="primary")

        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Prozess-Protokoll & Tags", interactive=False, lines=18)

    start_btn.click(
        fn=process_and_ingest,
        inputs=[file_input],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
