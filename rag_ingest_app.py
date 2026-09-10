import os
import re
import uuid
import shutil
import tempfile
import zipfile
import subprocess
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

# Untergestützte Dateiendungen (Inklusive KiCad 8 S-Expressions, SPICE, C/C++)
ALLOWED_EXTENSIONS = {
    ".pdf", ".py", ".md", ".txt", ".json",
    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch",
    ".cir", ".lib", ".c", ".cpp", ".h", ".hpp", ".dxf"
}

# Clients initialisieren
ollama_client = ollama.Client(host=OLLAMA_HOST)
qdrant_client = QdrantClient(url=QDRANT_HOST)

def ensure_qdrant_collection(vector_size: int):
    """Erstellt die Qdrant-Collection bei Bedarf."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def extract_text_from_file(file_path: str) -> str:
    """Liest Text aus PDFs sowie allen textbasierten EDA-Dateien (KiCad, SPICE, Code)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def collect_valid_files_from_dir(target_dir: str):
    """Durchsucht einen Ordner rekursiv nach allen unterstützen Dateiformaten."""
    valid_files = []
    for root, dirs, files in os.walk(target_dir):
        # .git Ordner ignorieren
        if ".git" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                valid_files.append(os.path.join(root, file))
    return valid_files

def process_and_ingest(files, github_url):
    all_files_to_process = []
    temp_dirs_to_clean = []

    status_log = f"🚀 Starte Ingestion-Pipeline\n"
    status_log += f"Embedding-Modell: {EMBED_MODEL} | Qdrant: {QDRANT_HOST}\n\n"
    yield status_log

    # 1. GitHub Repository verarbeiten (falls angegeben)
    if github_url and github_url.strip():
        url = github_url.strip()
        status_log += f"📦 Clone GitHub Repository: {url}...\n"
        yield status_log

        repo_dir = tempfile.mkdtemp(prefix="git_repo_")
        temp_dirs_to_clean.append(repo_dir)

        try:
            subprocess.run(["git", "clone", "--depth", "1", url, repo_dir], check=True, capture_output=True, text=True)
            repo_files = collect_valid_files_from_dir(repo_dir)
            all_files_to_process.extend(repo_files)
            status_log += f"   ↳ {len(repo_files)} relevante Dateien im Repository gefunden.\n\n"
            yield status_log
        except Exception as e:
            status_log += f"   ❌ Fehler beim Clonen von Git: {e}\n\n"
            yield status_log

    # 2. Hochgeladene Dateien & ZIPs verarbeiten
    if files:
        for file_obj in files:
            filepath = file_obj.name
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".zip":
                status_log += f"📂 Entpacke ZIP-Archiv: {filename}...\n"
                yield status_log
                zip_dir = tempfile.mkdtemp(prefix="zip_extract_")
                temp_dirs_to_clean.append(zip_dir)

                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(zip_dir)

                extracted_files = collect_valid_files_from_dir(zip_dir)
                all_files_to_process.extend(extracted_files)
                status_log += f"   ↳ {len(extracted_files)} relevante Dateien im ZIP gefunden.\n\n"
                yield status_log
            elif ext in ALLOWED_EXTENSIONS:
                all_files_to_process.append(filepath)

    if not all_files_to_process:
        status_log += "❌ Keine verarbeitbaren Dateien gefunden."
        yield status_log
        return

    status_log += f"📊 Gesamtzahl zu verarbeitender Dateien: {len(all_files_to_process)}\n\n"
    yield status_log

    # 3. Schleife über alle gesammelten Dateien (LLM-Aufbereitung + Embedding)
    try:
        for idx, filepath in enumerate(all_files_to_process, 1):
            filename = os.path.basename(filepath)
            status_log += f"[{idx}/{len(all_files_to_process)}] Verarbeite: {filename}\n"
            yield status_log

            # Text extrahieren
            raw_text = extract_text_from_file(filepath)
            if not raw_text.strip():
                status_log += f"   ⚠️ Datei leer oder nicht lesbar. Übersprungen.\n"
                yield status_log
                continue

            # LLM-Analyse
            status_log += f"   ↳ Strukturierung via LLM ({LLM_MODEL})...\n"
            yield status_log

            prompt = f"""
            Du bist ein Ingestion-Agent für ein EDA/PCB-RAG-System.
            Analysiere den folgenden Inhalt (Datei: {filename}) und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

            INHALT:
            {raw_text[:6000]}

            REGELN:
            1. Generiere in der ERSTEN Zeile ein exaktes Kategorie-Schlagwort in eckigen Klammern, z.B. [TAG: SKIDL_API], [TAG: KICAD_SYMBOL], [TAG: KICAD_FOOTPRINT], [TAG: KICAD_PCBNEW], [TAG: SPICE_SIM].
            2. Fasse den Zweck/Inhalt kurz zusammen.
            3. Extrahiere oder erstelle ein lauffähiges Codebeispiel, eine Symbol-Spezifikation oder eine Konfigurationsanleitung.
            """

            response = ollama_client.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
            processed_md = response['message']['content']

            # Tag extrahieren
            tag_match = re.search(r'\[TAG:\s*([A-Z0-9_]+)\]', processed_md, re.IGNORECASE)
            category_tag = tag_match.group(1).upper() if tag_match else "GENERAL_EDA"

            # Vektorisierung mit bge-m3
            embed_res = ollama_client.embeddings(model=EMBED_MODEL, prompt=processed_md)
            vector = embed_res['embedding']

            # Qdrant Ingest
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

        status_log += "🎉 Ingestion vollständig abgeschlossen! Die Daten stehen im Wissensspeicher bereit."
        yield status_log

    finally:
        # Aufräumen temporärer Ordner (Git Clones / Unzipped Folder)
        for t_dir in temp_dirs_to_clean:
            shutil.rmtree(t_dir, ignore_errors=True)

# UI Definition
with gr.Blocks(title="Universal RAG Knowledge Ingest") as demo:
    gr.Markdown("# 📥 Universal PCB/EDA Knowledge Ingest")
    gr.Markdown("Unterstützt Einzeldateien, `.zip`-Archive, KiCad 8 S-Expressions (`.kicad_sym`, `.kicad_mod`, `.kicad_sch`, `.kicad_pcb`), SPICE-Dateien sowie direkte **GitHub Repository URLs**.")

    with gr.Row():
        with gr.Column(scale=1):
            github_input = gr.Textbox(
                label="GitHub Repository URL (optional)",
                placeholder="https://github.com/xesscorp/skidl",
                lines=1
            )
            file_input = gr.File(
                label="Dateien & ZIP-Archive hochladen",
                file_count="multiple",
                file_types=[
                    ".pdf", ".py", ".md", ".txt", ".json", ".zip",
                    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch",
                    ".cir", ".lib", ".c", ".cpp", ".h"
                ]
            )
            start_btn = gr.Button("🚀 Ingestion Starte", variant="primary")

        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Prozess-Protokoll & Tags", interactive=False, lines=20)

    start_btn.click(
        fn=process_and_ingest,
        inputs=[file_input, github_input],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
