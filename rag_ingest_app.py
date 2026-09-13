import os
import re
import uuid
import shutil
import zipfile
import hashlib
import json
import threading
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
CONFIG_FILE = "/tmp/rag_ingest_config.json"

# Whitelist aller unterstützten Formate
TEXT_EXTENSIONS = {
    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch", ".kicad_prj", ".kicad_dru",
    ".sch", ".net", ".cir", ".lib", ".mod", ".sym", ".spice", ".sub", ".mcb", ".dxf",
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".c", ".h", ".cpp", ".hpp",
    ".js", ".ts", ".html", ".css", ".rst", ".csv", ".ini", ".conf", ".sh",
    ".pdf"
}

# Wissens-Kategorien & Prompts
CATEGORIES = {
    "⚡ PCB & Hardware Design": {
        "collection": "pcb_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für ein EDA/PCB-RAG-System.
Analysiere den Inhalt (z.B. Python-Code, KiCad-Symbol/Footprint, SPICE, Doku) und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

STRIKTE REGELN:
1. ERSTE ZEILE: Zwingend ein exaktes Kategorie-Schlagwort in eckigen Klammern, z.B.:
   [TAG: KICAD_FOOTPRINT], [TAG: KICAD_SYM], [TAG: KICAD_PCBNEW], [TAG: SKIDL_API] oder [TAG: SPICE_SIM].
2. ABSOLUTES HALLUZINATIONSVERBOT: Verarbeite den DATEINAMEN und den INHALT strikt faktengetreu. Erfinde NIEMALS abweichende Bauteilabmessungen oder falsche Pin-Belegungen!
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

# --- CONFIG PERSISTENCE HELPERS ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(data: dict):
    try:
        current = load_config()
        current.update(data)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Fehler beim Speichern der Konfiguration: {e}")

# --- THREAD-SICHERER BACKGROUND TASK MANAGER ---
class IngestTaskManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.cancel_event = threading.Event()
        self.log_messages = ["Inaktiv. Bereit für neuen Ingestion-Job."]
        self.status_header = "⚪ Status: Inaktiv"
        self.total_files = 0
        self.processed_files = 0
        self.current_thread = None

    def append_log(self, text: str):
        with self.lock:
            self.log_messages.append(text)

    def set_status(self, header: str):
        with self.lock:
            self.status_header = header

    def get_ui_snapshot(self):
        with self.lock:
            full_log = "\n".join(self.log_messages)
            return full_log, f"### {self.status_header}"

    def request_cancel(self):
        with self.lock:
            if self.is_running:
                self.cancel_event.set()
                self.append_log("\n🛑 Abbruch-Signal empfangen. Aktuelle Datei wird noch zu Ende indiziert...")
                self.status_header = "🟡 Status: WIRD ABGEBROCHEN..."
                return f"### {self.status_header}"
            return f"### {self.status_header}"

    def start_background_job(self, files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model):
        with self.lock:
            if self.is_running:
                return f"### ⚠️ Status: JOB LÄUFT BEREITS"

            # Speichere Modell- und Kategorie-Präferenz
            save_config({"last_model": selected_model, "last_category": category_key})

            self.is_running = True
            self.cancel_event.clear()
            self.log_messages = []
            self.processed_files = 0
            self.total_files = 0
            self.status_header = "🟢 Status: WIRD GESTARTET..."

            self.current_thread = threading.Thread(
                target=self._run_job,
                args=(files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model),
                daemon=True
            )
            self.current_thread.start()
            return f"### {self.status_header}"

    def _run_job(self, files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model):
        category_info = CATEGORIES.get(category_key, CATEGORIES["📚 Allgemeines Wissen & Dokumente"])
        target_collection = category_info["collection"]
        system_prompt = category_info["system_prompt"]
        active_model = selected_model if selected_model else DEFAULT_MODEL

        session_id = str(uuid.uuid4())[:8]
        temp_work_dir = os.path.join("/tmp", f"rag_ingest_{session_id}")
        os.makedirs(temp_work_dir, exist_ok=True)

        self.append_log(f"🚀 Starte entkoppelten Hintergrund-Ingest (Session: {session_id})")
        self.append_log(f"Modell: {active_model} | Collection: '{target_collection}' | Embedding: {EMBED_MODEL}\n")

        files_to_process = []

        try:
            # 1. Uploads & ZIPs
            if files:
                for file_obj in files:
                    fname = os.path.basename(file_obj.name)
                    ext = os.path.splitext(fname)[1].lower()

                    if ext == ".zip":
                        self.append_log(f"📦 Entpacke ZIP: {fname}...")
                        zip_extract_dir = os.path.join(temp_work_dir, f"zip_{uuid.uuid4()[:4]}")
                        with zipfile.ZipFile(file_obj.name, 'r') as zip_ref:
                            zip_ref.extractall(zip_extract_dir)
                        extracted = collect_files_from_dir(zip_extract_dir)
                        files_to_process.extend(extracted)
                        self.append_log(f"   ↳ {len(extracted)} Datei(en) im ZIP entpackt.")
                    elif ext in TEXT_EXTENSIONS:
                        files_to_process.append((fname, file_obj.name))

            # 2. Ausgewählte Git-Ordner
            if scanned_repo_path and os.path.exists(scanned_repo_path) and selected_folders:
                self.append_log("🌐 Erfassung ausgewählter Git-Ordner...")
                if "ALL_REPO" in selected_folders:
                    repo_files = collect_files_from_dir(scanned_repo_path)
                else:
                    repo_files = []
                    for subfolder in selected_folders:
                        repo_files.extend(collect_files_from_dir(scanned_repo_path, target_subfolder=subfolder))
                files_to_process.extend(repo_files)

            if not files_to_process:
                self.append_log("\n❌ Keine Dateien zur Verarbeitung gefunden.")
                self.set_status("🔴 Status: BEENDET (Keine Dateien)")
                return

            # 3. Filterung nach Dateiendungen
            if selected_exts:
                files_to_process = [
                    (rel_p, full_p) for rel_p, full_p in files_to_process
                    if os.path.splitext(rel_p)[1].lower() in selected_exts
                ]

            # Duplikate filtern
            files_to_process = list({rel_p: full_p for rel_p, full_p in files_to_process}.items())

            if not files_to_process:
                self.append_log("\n❌ Keine Dateien entsprechen den Dateiformat-Filtern.")
                self.set_status("🔴 Status: BEENDET (Keine Übereinstimmung)")
                return

            self.total_files = len(files_to_process)
            self.append_log(f"\n📊 Gesamt: {self.total_files} eindeutige Datei(en) bereit zur Indizierung.\n")

            # 4. Haupt-Schleife mit transparenter Abbruch-Quittierung
            for idx, (rel_path, file_path) in enumerate(files_to_process, 1):
                if self.cancel_event.is_set():
                    self.append_log("🛑 Ingestion vorzeitig abgebrochen.")
                    self.set_status(f"🔴 Status: ABGEBROCHEN ({self.processed_files}/{self.total_files})")
                    return

                self.processed_files = idx
                self.set_status(f"🟢 Status: LÄUFT ({idx}/{self.total_files} - {os.path.basename(rel_path)})")

                raw_text = extract_text_from_file(file_path)
                if not raw_text.strip():
                    self.append_log(f"[{idx}/{self.total_files}] ⚠️ Datei leer oder ungültig: {rel_path}")
                    continue

                content_hash = calculate_sha256(raw_text)

                if is_file_indexed(target_collection, rel_path, content_hash):
                    self.append_log(f"[{idx}/{self.total_files}] ⏭️ Unverändert übersprungen: {rel_path}")
                    continue

                self.append_log(f"[{idx}/{self.total_files}] Verarbeite via LLM: {rel_path}")

                ext = os.path.splitext(rel_path)[1].lower()
                eda_guard = ""
                if ext in {".kicad_mod", ".kicad_sym", ".kicad_pcb", ".kicad_sch", ".cir", ".net"}:
                    eda_guard = (
                        "\n\nSTRIKTE ANWEISUNG FÜR NATIVE EDA-DATEIEN:\n"
                        "- Bette den bereitgestellten ORIGINALTEXT zwingend 1:1 im Markdown-Codeblock ein.\n"
                        "- Erfinde keine Geometrien, Pin-Anzahlen oder Maße, die nicht im Originaltext stehen!\n"
                    )

                full_user_prompt = f"DATEIPFAD: {rel_path}\nINHALT:\n{raw_text[:6000]}{eda_guard}"

                try:
                    response = ollama_client.chat(
                        model=active_model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': full_user_prompt}
                        ]
                    )
                    processed_md = response['message']['content']

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

                    # Quittierung erfolgt IMMER nach erfolgreichem Einpflegen in Qdrant
                    self.append_log(f"   ✅ Indiziert in '{target_collection}' | **Tag: #{category_tag}**\n")

                except Exception as e:
                    self.append_log(f"   ❌ Fehler bei Verarbeitung: {str(e)}\n")

                # Prüfe Abbruch direkt nach Quittierung des aktuellen Dokuments
                if self.cancel_event.is_set():
                    self.append_log("🛑 Letzte Datei erfolgreich quittiert. Ingestion jetzt beendet.")
                    self.set_status(f"🔴 Status: ABGEBROCHEN ({idx}/{self.total_files})")
                    return

            self.append_log(f"\n🎉 Ingestion vollständig abgeschlossen! Alle Daten sind in Collection '{target_collection}' verfügbar.")
            self.set_status(f"✅ Status: ABGESCHLOSSEN ({self.total_files}/{self.total_files})")

        finally:
            with self.lock:
                self.is_running = False
            if os.path.exists(temp_work_dir):
                shutil.rmtree(temp_work_dir, ignore_errors=True)
            if scanned_repo_path and os.path.exists(scanned_repo_path):
                shutil.rmtree(scanned_repo_path, ignore_errors=True)

# Instanziierung
task_manager = IngestTaskManager()

# --- HILFSFUNKTIONEN ---
def get_ollama_models():
    config = load_config()
    saved_model = config.get("last_model", DEFAULT_MODEL)

    try:
        res = ollama_client.list()
        models_data = res.get('models', []) if isinstance(res, dict) else getattr(res, 'models', [])
        choices = []
        found_saved = False

        for m in models_data:
            m_name = (
                (m.get('model') or m.get('name'))
                if isinstance(m, dict)
                else (getattr(m, 'model', None) or getattr(m, 'name', None))
            )
            if m_name and m_name != EMBED_MODEL:
                if m_name == saved_model:
                    found_saved = True
                is_cloud = any(term in m_name.lower() for term in ["cloud", "gpt", "claude", "gemini", "remote", "openai", "deepseek-v3"])
                label = f"{m_name} ({'☁️ Cloud' if is_cloud else '💻 Lokal'})"
                choices.append((label, m_name))

        if choices:
            selected_default = saved_model if found_saved else choices[0][1]
            return choices, selected_default
    except Exception as e:
        print(f"Fehler beim Abrufen der Ollama-Modelle: {e}")

    return [(f"{DEFAULT_MODEL} (💻 Lokal)", DEFAULT_MODEL)], DEFAULT_MODEL

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
    if not selected:
        return []
    if "ALL_REPO" in selected and len(selected) > 1:
        return [item for item in selected if item != "ALL_REPO"]
    return selected

def update_model_preference(model_name):
    save_config({"last_model": model_name})

def update_category_preference(cat_name):
    save_config({"last_category": cat_name})

def scan_github_repository(github_url):
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

# --- GRADIO CONTROL CENTER (OPTIMIERTES DASHBOARD-LAYOUT) ---
initial_model_choices, initial_default_model = get_ollama_models()
saved_cfg = load_config()
initial_default_category = saved_cfg.get("last_category", list(CATEGORIES.keys())[0])

with gr.Blocks(title="Universal RAG Control Center", css="footer {visibility: hidden}") as demo:
    repo_state = gr.State("")

    # Auto-Polling Timer
    status_timer = gr.Timer(value=2.0)

    gr.Markdown("# 🏢 Universal RAG Ingestion Control Center")

    with gr.Row():
        # LINKS: Steuerung & Quellen (50% Breite)
        with gr.Column(scale=1):
            status_banner = gr.Markdown("### ⚪ Status: Inaktiv")

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=initial_model_choices,
                    value=initial_default_model,
                    label="LLM Modell (Gespeichert)",
                    interactive=True,
                    scale=3
                )
                refresh_models_btn = gr.Button("🔄", variant="secondary", scale=1)
                category_dropdown = gr.Dropdown(
                    choices=list(CATEGORIES.keys()),
                    value=initial_default_category,
                    label="Knowledge Collection",
                    interactive=True,
                    scale=3
                )

            # Quellen in Tabs unterteilt -> Extrem platzsparend!
            with gr.Tabs():
                with gr.Tab("📁 Dateiupload / ZIP"):
                    file_input = gr.File(
                        label="Dateien oder ZIP-Archiv hochladen",
                        file_count="multiple"
                    )

                with gr.Tab("🌐 Git Repository Crawler"):
                    with gr.Row():
                        github_input = gr.Textbox(
                            label="Repository URL",
                            placeholder="https://gitlab.com/kicad/libraries/kicad-symbols.git",
                            scale=4
                        )
                        scan_repo_btn = gr.Button("🔍 Scannen", variant="secondary", scale=1)

                    with gr.Row():
                        folder_checkboxes = gr.CheckboxGroup(
                            label="Ordnerstruktur",
                            choices=[],
                            visible=False,
                            interactive=True,
                            scale=1
                        )
                        ext_checkboxes = gr.CheckboxGroup(
                            label="Dateiformate Filter",
                            choices=[],
                            visible=False,
                            interactive=True,
                            scale=1
                        )

            with gr.Row():
                start_btn = gr.Button("🚀 Ingest Starten", variant="primary", scale=3)
                stop_btn = gr.Button("🛑 Abbrechen", variant="stop", scale=2)

        # RECHTS: Live Protokoll (50% Breite)
        with gr.Column(scale=1):
            status_output = gr.Textbox(
                label="Server Live-Protokoll",
                interactive=False,
                lines=24,
                autoscroll=True
            )

    # Event Bindings
    status_timer.tick(
        fn=task_manager.get_ui_snapshot,
        outputs=[status_output, status_banner]
    )

    model_dropdown.change(
        fn=update_model_preference,
        inputs=[model_dropdown]
    )

    category_dropdown.change(
        fn=update_category_preference,
        inputs=[category_dropdown]
    )

    refresh_models_btn.click(
        fn=lambda: gr.Dropdown(choices=get_ollama_models()[0]),
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

    start_btn.click(
        fn=task_manager.start_background_job,
        inputs=[file_input, repo_state, folder_checkboxes, ext_checkboxes, category_dropdown, model_dropdown],
        outputs=[status_banner]
    )

    stop_btn.click(
        fn=task_manager.request_cancel,
        outputs=[status_banner]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
