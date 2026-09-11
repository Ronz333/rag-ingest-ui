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
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
EMBED_MODEL = "bge-m3"

# Whitelist aller unterstützten Formate
TEXT_EXTENSIONS = {
    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch", ".kicad_prj", ".kicad_dru",
    ".sch", ".net", ".cir", ".lib", ".mod", ".sym", ".spice", ".sub", ".mcb", ".dxf",
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".c", ".h", ".cpp", ".hpp",
    ".js", ".ts", ".html", ".css", ".rst", ".csv", ".ini", ".conf", ".sh",
    ".pdf"
}

# Wissens-Kategorien & verschärfte System-Prompts gegen Halluzinationen
CATEGORIES = {
    "⚡ PCB & Hardware Design": {
        "collection": "pcb_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für ein EDA/PCB-RAG-System.
Analysiere den Inhalt (z.B. Python-Code, KiCad-Symbol/Footprint, SPICE, Doku) und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

STRIKTE REGELN:
1. ERSTE ZEILE: Zwingend ein exaktes Kategorie-Schlagwort in eckigen Klammern, z.B.:
   [TAG: KICAD_FOOTPRINT], [TAG: KICAD_SYM], [TAG: KICAD_PCBNEW], [TAG: SKIDL_API] oder [TAG: SPICE_SIM].
2. ABSOLUTES HALLUZINATIONSVERBOT: Verarbeite den DATEINAMEN und den INHALT strikt faktengetreu. Erfinde NIEMALS abweichende Bauteilabmessungen (z.B. kein 0603 ausgeben, wenn die Datei einen 10x12.5mm Kondensator beschreibt) oder falsche Pin-Belegungen!
3. CODE-INTEGRITÄT: Bette bei KiCad-Dateien (.kicad_mod, .kicad_sym) und SPICE-Netzlisten (.cir) den bereitgestellten ORIGINAL-CODE 1:1 unverändert im Markdown-Codeblock ein.
4. Zusammenfassung: Fasse Zweck, Parameter und Pinbelegung sachlich zusammen."""
    },
    "💻 Programmiersprachen & Software": {
        "collection": "programming_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für Software-Dokumentation und Source Code.
Analysiere den Quellcode oder die API-Dokumentation und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

REGELN:
1. ERSTE ZEILE: Exaktes Kategorie-Schlagwort in eckigen Klammern, z.B. [TAG: PYTHON_API], [TAG: ALGORITHM], [TAG: DOCKER], [TAG: REST_API].
2. Fasse Architektur und Verwendungszweck prägnant zusammen.
3. Bette den originalen, kommentierten Code sauber ein."""
    },
    "🔬 Wissenschaft & Forschung": {
        "collection": "science_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für wissenschaftliche Arbeiten und Forschungsdokumente.
Analysiere das Dokument und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

REGELN:
1. ERSTE ZEILE: Exaktes Kategorie-Schlagwort in eckigen Klammern, z.B. [TAG: METHODOLOGY], [TAG: FORMULA], [TAG: EXPERIMENT].
2. Fasse Abstract, Methodik und Kernergebnisse zusammen."""
    },
    "🩺 Gesundheit & Medizin": {
        "collection": "health_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für medizinische und gesundheitswissenschaftliche Dokumente.
Analysiere den Text und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

REGELN:
1. ERSTE ZEILE: Exaktes Kategorie-Schlagwort in eckigen Klammern, z.B. [TAG: ANATOMY], [TAG: PHARMA], [TAG: CLINICAL_STUDY].
2. Fasse Fachbegriffe und Wirkungsweisen sachlich zusammen."""
    },
    "📚 Allgemeines Wissen & Dokumente": {
        "collection": "general_knowledge_base",
        "system_prompt": """Du bist ein allgemeiner Dokumenten-Ingestion-Agent.
Analysiere den Text und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

REGELN:
1. ERSTE ZEILE: Exaktes Kategorie-Schlagwort in eckigen Klammern, z.B. [TAG: SUMMARY], [TAG: GUIDE], [TAG: NOTES].
2. Erstelle eine verständliche Gliederung mit Kernaussagen."""
    }
}

ollama_client = ollama.Client(host=OLLAMA_HOST)
qdrant_client = QdrantClient(url=QDRANT_HOST)

def get_ollama_models():
    """Lädt Ollama-Modelle und unterscheidet zwischen Lokal und Cloud."""
    try:
        res = ollama_client.list()
        models_data = res.get('models', []) if isinstance(res, dict) else getattr(res, 'models', [])
        choices = []
        for m in models_data:
            m_name = (
                (m.get('model') or m.get('name'))
                if isinstance(m, dict)
                else (getattr(m, 'model', None) or getattr(m, 'name', None))
            )
            if m_name and m_name != EMBED_MODEL:
                is_cloud = any(term in m_name.lower() for term in ["cloud", "gpt", "claude", "gemini", "remote", "openai", "deepseek-v3"])
                label = f"{m_name} ({'☁️ Cloud' if is_cloud else '💻 Lokal'})"
                choices.append((label, m_name))
        if choices:
            return choices
    except Exception as e:
        print(f"Fehler beim Abrufen der Ollama-Modelle: {e}")
    return [(f"{DEFAULT_MODEL} (💻 Lokal)", DEFAULT_MODEL)]

def ensure_qdrant_collection(collection_name: str, vector_size: int):
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in collections:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def calculate_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def is_file_indexed(collection_name: str, rel_path: str, content_hash: str) -> bool:
    try:
        results, _ = qdrant_client.scroll(
            collection_name=collection_name,
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

def collect_files_from_dir(directory: str, target_subfolder: str = ""):
    collected = []
    base_search_path = os.path.join(directory, target_subfolder) if target_subfolder else directory

    if not os.path.exists(base_search_path):
        return collected

    for root, _, files in os.walk(base_search_path):
        if ".git" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in TEXT_EXTENSIONS:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, directory)
                collected.append((rel_path, full_path))
    return collected

def handle_folder_selection(selected):
    """Entfernt automatisch die Gesamtauswahl, sobald ein Unterordner angeklickt wird."""
    if not selected:
        return []
    if "ALL_REPO" in selected and len(selected) > 1:
        return [item for item in selected if item != "ALL_REPO"]
    return selected

def scan_github_repository(github_url):
    """Klont das Repository vorab, baut eine Ordner-Baumstruktur und erkennt Dateiformate."""
    if not github_url or not github_url.strip():
        return (
            gr.update(choices=[], value=[], visible=False),
            gr.update(choices=[], value=[], visible=False),
            "",
            "❌ Bitte valide Repository URL angeben."
        )

    session_id = str(uuid.uuid4())[:8]
    repo_dir = os.path.join("/tmp", f"scan_repo_{session_id}")

    res = subprocess.run(
        ["git", "clone", "--depth", "1", github_url.strip(), repo_dir],
        capture_output=True, text=True
    )

    if res.returncode != 0:
        return (
            gr.update(choices=[], value=[], visible=False),
            gr.update(choices=[], value=[], visible=False),
            "",
            f"❌ Git-Clone fehlgeschlagen:\n{res.stderr[:300]}"
        )

    valid_dirs = set()
    ext_counts = {}

    for root, _, files in os.walk(repo_dir):
        if ".git" in root:
            continue
        has_valid = False
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in TEXT_EXTENSIONS:
                has_valid = True
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

        if has_valid:
            rel = os.path.relpath(root, repo_dir)
            if rel == ".":
                valid_dirs.add("")
            else:
                parts = rel.split(os.sep)
                for i in range(1, len(parts) + 1):
                    valid_dirs.add(os.path.sep.join(parts[:i]))

    sorted_dirs = sorted(list(valid_dirs))
    folder_choices = [("📁 / (Gesamtes Repository)", "ALL_REPO")]

    for d in sorted_dirs:
        if d == "":
            continue
        parts = d.split(os.sep)
        depth = len(parts) - 1
        indent = "│   " * depth + "├── "
        label = f"{indent}📁 {parts[-1]}"
        folder_choices.append((label, d))

    ext_choices = [
        (f"{ext} ({count} Dateie{'n' if count > 1 else ''})", ext)
        for ext, count in sorted(ext_counts.items(), key=lambda x: x[0])
    ]

    log_msg = f"✅ Repository gescannt! {len(folder_choices)-1} Ordner und {len(ext_choices)} Dateiformate erkannt."

    return (
        gr.update(choices=folder_choices, value=["ALL_REPO"], visible=True),
        gr.update(choices=ext_choices, value=[e[1] for e in ext_choices], visible=True),
        repo_dir,
        log_msg
    )

def process_and_ingest(files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model):
    if not files and not (scanned_repo_path and selected_folders):
        yield "❌ Bitte entweder Dateien hochladen oder ein Repository scannen und auswählen."
        return

    category_info = CATEGORIES.get(category_key, CATEGORIES["📚 Allgemeines Wissen & Dokumente"])
    target_collection = category_info["collection"]
    system_prompt = category_info["system_prompt"]
    active_model = selected_model if selected_model else DEFAULT_MODEL

    session_id = str(uuid.uuid4())[:8]
    temp_work_dir = os.path.join("/tmp", f"rag_ingest_{session_id}")
    os.makedirs(temp_work_dir, exist_ok=True)

    status_log = f"🚀 Starte Ingestion-Prozess (Session: {session_id})\n"
    status_log += f"Modell: {active_model} | Collection: '{target_collection}' | Embedding: {EMBED_MODEL}\n\n"
    yield status_log

    files_to_process = []

    try:
        # 1. Uploads & ZIPs
        if files:
            for file_obj in files:
                fname = os.path.basename(file_obj.name)
                ext = os.path.splitext(fname)[1].lower()

                if ext == ".zip":
                    status_log += f"📦 Entpacke ZIP: {fname}...\n"
                    yield status_log
                    zip_extract_dir = os.path.join(temp_work_dir, f"zip_{uuid.uuid4()[:4]}")
                    with zipfile.ZipFile(file_obj.name, 'r') as zip_ref:
                        zip_ref.extractall(zip_extract_dir)
                    extracted = collect_files_from_dir(zip_extract_dir)
                    files_to_process.extend(extracted)
                    status_log += f"   ↳ {len(extracted)} Datei(en) im ZIP entpackt.\n"
                    yield status_log
                elif ext in TEXT_EXTENSIONS:
                    files_to_process.append((fname, file_obj.name))

        # 2. Ausgewählte Git-Ordner
        if scanned_repo_path and os.path.exists(scanned_repo_path) and selected_folders:
            status_log += f"🌐 Erfassung ausgewählter Git-Ordner...\n"
            yield status_log

            if "ALL_REPO" in selected_folders:
                repo_files = collect_files_from_dir(scanned_repo_path)
            else:
                repo_files = []
                for subfolder in selected_folders:
                    repo_files.extend(collect_files_from_dir(scanned_repo_path, target_subfolder=subfolder))

            files_to_process.extend(repo_files)

        if not files_to_process:
            status_log += "\n❌ Keine Dateien erfasst."
            yield status_log
            return

        # 3. Filterung nach ausgewählten Dateiendungen
        if selected_exts:
            files_to_process = [
                (rel_p, full_p) for rel_p, full_p in files_to_process
                if os.path.splitext(rel_p)[1].lower() in selected_exts
            ]

        # Duplikate entfernen
        files_to_process = list({rel_p: full_p for rel_p, full_p in files_to_process}.items())

        if not files_to_process:
            status_log += "\n❌ Keine Dateien entsprechen den ausgewählten Dateiformat-Filtern."
            yield status_log
            return

        status_log += f"📊 Gesamt: {len(files_to_process)} eindeutige Datei(en) zur Indizierung bereit.\n\n"
        yield status_log

        # 4. Indizierung & Deduplizierung via Hash
        for idx, (rel_path, file_path) in enumerate(files_to_process, 1):
            raw_text = extract_text_from_file(file_path)
            if not raw_text.strip():
                status_log += f"[{idx}/{len(files_to_process)}] ⚠️ Datei leer oder ungültig: {rel_path}\n"
                yield status_log
                continue

            content_hash = calculate_sha256(raw_text)

            if is_file_indexed(target_collection, rel_path, content_hash):
                status_log += f"[{idx}/{len(files_to_process)}] ⏭️ Unverändert übersprungen: {rel_path}\n"
                yield status_log
                continue

            status_log += f"[{idx}/{len(files_to_process)}] Verarbeite: {rel_path}\n"
            yield status_log

            # Strikter Prompt-Zusatz für strukturierte EDA-Dateien zur Vermeidung von Halluzinationen
            ext = os.path.splitext(rel_path)[1].lower()
            eda_prompt_guard = ""
            if ext in {".kicad_mod", ".kicad_sym", ".kicad_pcb", ".kicad_sch", ".cir", ".net"}:
                eda_prompt_guard = (
                    "\n\nSTRIKTE ANWEISUNG FÜR NATIVE EDA-DATEIEN:\n"
                    "- Bette den bereitgestellten ORIGINALTEXT zwingend 1:1 im Markdown-Codeblock ein.\n"
                    "- Erfinde keine Geometrien, Pin-Anzahlen oder Maße, die nicht im Originaltext stehen!\n"
                )

            full_user_prompt = f"DATEIPFAD: {rel_path}\nINHALT:\n{raw_text[:6000]}{eda_prompt_guard}"

            response = ollama_client.chat(
                model=active_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_user_prompt}
                ]
            )
            processed_md = response['message']['content']

            # Erweiterte Tag-Extraktion (fängt [TAG: ...], [PAGE: ...] und [CATEGORY: ...] ab)
            tag_match = re.search(r'\[(?:TAG|PAGE|CATEGORY):\s*([A-Z0-9_]+)\]', processed_md, re.IGNORECASE)
            category_tag = tag_match.group(1).upper() if tag_match else "GENERAL"

            embed_res = ollama_client.embeddings(model=EMBED_MODEL, prompt=processed_md)
            vector = embed_res['embedding']

            ensure_qdrant_collection(target_collection, len(vector))

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{target_collection}_{rel_path}"))

            qdrant_client.upsert(
                collection_name=target_collection,
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

            status_log += f"   ✅ Indiziert in '{target_collection}' | **Tag: #{category_tag}**\n\n"
            yield status_log

        status_log += f"🎉 Ingestion abgeschlossen! Dokumente sind in Collection '{target_collection}' verfügbar."
        yield status_log

    finally:
        if os.path.exists(temp_work_dir):
            shutil.rmtree(temp_work_dir, ignore_errors=True)
        if scanned_repo_path and os.path.exists(scanned_repo_path):
            shutil.rmtree(scanned_repo_path, ignore_errors=True)

# Gradio Interface
initial_models = get_ollama_models()
default_model_value = initial_models[0][1] if initial_models else DEFAULT_MODEL

with gr.Blocks(title="Universal RAG Knowledge Ingest") as demo:
    repo_state = gr.State("")

    gr.Markdown("# 📥 Universal RAG Knowledge Ingestion Pipeline")
    gr.Markdown("Multilinguale Vektorisierung (`bge-m3`), Hash-Deduplizierung und erweiterte Repository-Filterung.")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=initial_models,
                    value=default_model_value,
                    label="LLM Modell auswählen",
                    interactive=True,
                    scale=4
                )
                refresh_models_btn = gr.Button("🔄", variant="secondary", scale=1)

            category_dropdown = gr.Dropdown(
                choices=list(CATEGORIES.keys()),
                value=list(CATEGORIES.keys())[0],
                label="Ziel-Kategorie / Knowledge Collection",
                interactive=True
            )

            file_input = gr.File(
                label="Dateien / ZIP-Archive hochladen",
                file_count="multiple"
            )

            with gr.Group():
                github_input = gr.Textbox(
                    label="GitHub / GitLab Repository URL",
                    placeholder="https://gitlab.com/kicad/libraries/kicad-symbols.git"
                )
                scan_repo_btn = gr.Button("🔍 Repository scannen", variant="secondary")

                folder_checkboxes = gr.CheckboxGroup(
                    label="Ordnerstruktur im Repository",
                    choices=[],
                    visible=False,
                    interactive=True
                )

                ext_checkboxes = gr.CheckboxGroup(
                    label="Erkannte Dateiformate filtern",
                    choices=[],
                    visible=False,
                    interactive=True
                )

            with gr.Row():
                start_btn = gr.Button("🚀 Ingestion Starten", variant="primary", scale=3)
                stop_btn = gr.Button("🛑 Ingestion Abbrechen", variant="stop", scale=2)

        with gr.Column(scale=1):
            status_output = gr.Textbox(
                label="Ingestion-Protokoll",
                interactive=False,
                lines=25,
                autoscroll=True
            )

    refresh_models_btn.click(
        fn=lambda: gr.Dropdown(choices=get_ollama_models()),
        outputs=[model_dropdown]
    )

    scan_repo_btn.click(
        fn=scan_github_repository,
        inputs=[github_input],
        outputs=[folder_checkboxes, ext_checkboxes, repo_state, status_output]
    )

    folder_checkboxes.change(
        fn=handle_folder_selection,
        inputs=[folder_checkboxes],
        outputs=[folder_checkboxes]
    )

    start_event = start_btn.click(
        fn=process_and_ingest,
        inputs=[file_input, repo_state, folder_checkboxes, ext_checkboxes, category_dropdown, model_dropdown],
        outputs=[status_output]
    )

    stop_btn.click(
        fn=lambda log: log + "\n\n🛑 Ingestion durch Benutzer abgebrochen.",
        inputs=[status_output],
        outputs=[status_output],
        cancels=[start_event]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
