import os
import re
import uuid
import shutil
import zipfile
import hashlib
import subprocess
import gradio as gr
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from pypdf import PdfReader

# Konfiguration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_1")
EMBED_MODEL = "bge-m3"
COLLECTION_NAME = "pcb_knowledge_base"

# Relevante Dateiendungen (KiCad + Code + Doku)
TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json",
    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch",
    ".sch", ".net", ".cir", ".c", ".h", ".cpp"
}

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

def calculate_sha256(text: str) -> str:
    """Erzeugt einen eindeutigen SHA256-Hash des Dateiinhalts."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def is_file_indexed(rel_path: str, content_hash: str) -> bool:
    """Prüft im Qdrant-Payload, ob exakt dieser Pfad mit genau diesem Inhalt existiert."""
    try:
        results, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=rel_path)),
                    FieldCondition(key="content_hash", match=MatchValue(value=content_hash))
                ]
            ),
            limit=1
        )
        return len(results) > 0
    except Exception:
        return False

def extract_text_from_file(file_path: str) -> str:
    """Extrahiert Text aus PDFs und allen Klartext-/KiCad-Formaten."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            return ""
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

def collect_files_from_dir(directory: str):
    """Durchsucht ein Verzeichnis rekursiv nach unterstützten Dateitypen."""
    collected = []
    for root, _, files in os.walk(directory):
        if ".git" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in TEXT_EXTENSIONS or ext == ".pdf":
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, directory)
                collected.append((rel_path, full_path))
    return collected

def process_and_ingest(files, github_url):
    if not files and not (github_url and github_url.strip()):
        yield "❌ Bitte Dateien/ZIPs hochladen oder eine GitHub-URL angeben."
        return

    session_id = str(uuid.uuid4())[:8]
    temp_work_dir = os.path.join("/tmp", f"rag_ingest_{session_id}")
    os.makedirs(temp_work_dir, exist_ok=True)

    status_log = f"🚀 Starte Ingestion-Prozess (Session: {session_id})\n"
    status_log += f"Embedding-Modell: {EMBED_MODEL} | Qdrant: {QDRANT_HOST}\n\n"
    yield status_log

    files_to_process = []  # Liste aus (relativer Pfad, absoluter Dateipfad)

    try:
        # 1. Dateiuploads & ZIPs verarbeiten
        if files:
            for file_obj in files:
                fname = os.path.basename(file_obj.name)
                ext = os.path.splitext(fname)[1].lower()

                if ext == ".zip":
                    status_log += f"📦 Entpacke ZIP-Archiv: {fname}...\n"
                    yield status_log
                    zip_extract_dir = os.path.join(temp_work_dir, f"zip_{uuid.uuid4()[:4]}")
                    with zipfile.ZipFile(file_obj.name, 'r') as zip_ref:
                        zip_ref.extractall(zip_extract_dir)
                    extracted = collect_files_from_dir(zip_extract_dir)
                    files_to_process.extend(extracted)
                    status_log += f"   ↳ {len(extracted)} relevante Datei(en) im ZIP gefunden.\n"
                    yield status_log
                elif ext in TEXT_EXTENSIONS or ext == ".pdf":
                    files_to_process.append((fname, file_obj.name))

        # 2. GitHub Repository crawlen
        if github_url and github_url.strip():
            url = github_url.strip()
            status_log += f"🌐 Crawle GitHub Repository: {url}...\n"
            yield status_log
            repo_dir = os.path.join(temp_work_dir, "repo")
            res = subprocess.run(
                ["git", "clone", "--depth", "1", url, repo_dir],
                capture_output=True, text=True
            )
            if res.returncode == 0:
                repo_files = collect_files_from_dir(repo_dir)
                files_to_process.extend(repo_files)
                status_log += f"   ↳ {len(repo_files)} relevante Datei(en) im Git-Repo gefunden.\n"
            else:
                status_log += f"   ❌ Git-Clone fehlgeschlagen: {res.stderr[:200]}\n"
            yield status_log

        if not files_to_process:
            status_log += "\n❌ Keine unterstützten Dateien (.py, .kicad_*, .pdf, .md etc.) gefunden."
            yield status_log
            return

        status_log += f"\n📊 Gesamt: {len(files_to_process)} Datei(en) in der Warteschlange.\n\n"
        yield status_log

        # 3. Indizierung der Dateien mit Hash-Deduplizierung
        for idx, (rel_path, file_path) in enumerate(files_to_process, 1):
            raw_text = extract_text_from_file(file_path)
            if not raw_text.strip():
                status_log += f"[{idx}/{len(files_to_process)}] ⚠️ Datei leer/ungültig: {rel_path}\n"
                yield status_log
                continue

            content_hash = calculate_sha256(raw_text)

            # Prüfe, ob exakt dieser relativer Pfad mit demselben Inhalt bereits in Qdrant liegt
            if is_file_indexed(rel_path, content_hash):
                status_log += f"[{idx}/{len(files_to_process)}] ⏭️ Unverändert übersprungen: {rel_path}\n"
                yield status_log
                continue

            status_log += f"[{idx}/{len(files_to_process)}] Verarbeite: {rel_path}\n"
            yield status_log

            # LLM-Analyse & Aufbereitung
            prompt = f"""
            Du bist ein Ingestion-Agent für ein EDA/PCB-RAG-System.
            Analysiere den folgenden Inhalt (z.B. Python-Code, KiCad-Symbol/Footprint, Doku) und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

            DATEIPFAD: {rel_path}
            INHALT:
            {raw_text[:6000]}

            REGELN:
            1. Generiere in der ERSTEN Zeile ein exaktes Kategorie-Schlagwort in eckigen Klammern, z.B. [TAG: SKIDL_API], [TAG: KICAD_SYM], [TAG: KICAD_FOOTPRINT], [TAG: KICAD_PCBNEW], [TAG: SPICE_SIM].
            2. Fasse den Zweck kurz zusammen.
            3. Extrahiere oder erstelle ein lauffähiges Codebeispiel, eine Symbol-/Footprint-Beschreibung oder Konfiguration.
            """

            response = ollama_client.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
            processed_md = response['message']['content']

            tag_match = re.search(r'\[TAG:\s*([A-Z0-9_]+)\]', processed_md, re.IGNORECASE)
            category_tag = tag_match.group(1).upper() if tag_match else "GENERAL_EDA"

            # Vektorisierung via bge-m3
            embed_res = ollama_client.embeddings(model=EMBED_MODEL, prompt=processed_md)
            vector = embed_res['embedding']

            ensure_qdrant_collection(len(vector))

            # Eindeutige ID basierend auf dem relativen Pfad
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rel_path))

            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "filename": os.path.basename(rel_path),
                            "file_path": rel_path,
                            "content_hash": content_hash,
                            "category_tag": category_tag,
                            "content": processed_md
                        }
                    )
                ]
            )

            status_log += f"   ✅ In Qdrant indiziert | **Tag: #{category_tag}**\n\n"
            yield status_log

        status_log += "🎉 Ingestion & Crawling vollständig abgeschlossen!"
        yield status_log

    finally:
        if os.path.exists(temp_work_dir):
            shutil.rmtree(temp_work_dir, ignore_errors=True)

# UI Definition
with gr.Blocks(title="Universal RAG Ingest") as demo:
    gr.Markdown("# 📥 Universal RAG Knowledge Ingestion")
    gr.Markdown("Lade Quelldateien, **ZIP-Archive** oder ein **GitHub-Repository** hoch. Mit automatischer Hash-Deduplizierung & KiCad-Unterstützung.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Dateien / ZIP-Archive hochladen",
                file_count="multiple",
                file_types=[
                    ".zip", ".pdf", ".py", ".md", ".txt", ".json",
                    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch",
                    ".sch", ".net", ".cir", ".c", ".h"
                ]
            )
            github_input = gr.Textbox(
                label="Oder GitHub Repository URL crawlen",
                placeholder="https://github.com/xesscorp/skidl"
            )
            start_btn = gr.Button("🚀 Ingestion & Crawling Starten", variant="primary")

        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Ingestion-Protokoll", interactive=False, lines=20)

    start_btn.click(
        fn=process_and_ingest,
        inputs=[file_input, github_input],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
