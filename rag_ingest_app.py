import os
import re
import uuid
import shutil
import zipfile
import hashlib
import json
import ast
import threading
import subprocess
import gradio as gr
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader

# --- KONFIGURATION & KONSTANTEN ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
EMBED_MODEL = "bge-m3"
CONFIG_FILE = "/tmp/rag_ingest_config.json"
DEFAULT_NUM_CTX = 8192
TB = "```"

IGNORED_FILENAMES = {"setup.py", "conftest.py", "__init__.py"}
IGNORED_PATH_PARTS = ["/docs/", "/tests/", "/build/", "/dist/"]

TEXT_EXTENSIONS = {
    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch", ".kicad_prj", ".kicad_dru",
    ".sch", ".net", ".cir", ".lib", ".mod", ".sym", ".spice", ".sub", ".mcb", ".dxf",
    ".rules", ".dsn", ".ses",
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".c", ".h", ".cpp", ".hpp",
    ".js", ".ts", ".html", ".css", ".rst", ".csv", ".ini", ".conf", ".sh",
    ".pdf"
}

CATEGORIES = {
    "⚡ PCB & Hardware Design": {
        "collection": "pcb_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für ein EDA/PCB-RAG-System.
Analysiere den Inhalt (z.B. Python/SKiDL-Code, KiCad-Dateien, DRC/ERC Rules) und erstelle ein strukturiertes RAG-Dokument im Markdown-Format.

STRIKTE REGELN:
1. ERSTE ZEILE: Zwingend ein exaktes Kategorie-Schlagwort in eckigen Klammern, z.B.:
   [TAG: SKIDL_API], [TAG: SKIDL_GOLDEN_EXAMPLE], [TAG: KICAD_PCBNEW], [TAG: KICAD_DRC_RULES], [TAG: FREEROUTING_RULES] oder [TAG: SPICE_SIM].
2. ABSOLUTES HALLUZINATIONSVERBOT: Verarbeite den DATEINAMEN und den INHALT strikt faktengetreu.
3. CODE-INTEGRITÄT: Bette bereitgestellten Quellcode/Regeln 1:1 im Markdown-Codeblock ein.
4. Zusammenfassung: Fasse Zweck, Parameter, Regeln und Schnittstellen sachlich zusammen."""
    },
    "💻 Programmiersprachen & Software": {
        "collection": "programming_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für Software-Dokumentation und Source Code."""
    },
    "🔬 Wissenschaft & Forschung": {
        "collection": "science_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für wissenschaftliche Arbeiten."""
    },
    "🩺 Gesundheit & Medizin": {
        "collection": "health_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für medizinische Dokumente."""
    },
    "📚 Allgemeines Wissen & Dokumente": {
        "collection": "general_knowledge_base",
        "system_prompt": """Du bist ein allgemeiner Dokumenten-Ingestion-Agent."""
    }
}

ollama_client = ollama.Client(host=OLLAMA_HOST)
qdrant_client = QdrantClient(url=QDRANT_HOST)

# --- SANITIZATION & CONFIG HELPERS ---
def sanitize_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    md_match = re.search(r'\((https?://[^\)]+)\)', raw_url)
    if md_match:
        return md_match.group(1).strip()
    url_match = re.search(r'https?://[^\s>\]\)]+', raw_url)
    if url_match:
        return url_match.group(0).strip()
    return raw_url.strip("[]()'\" ")

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

def get_indexed_hashes_set(collection_name: str) -> set:
    indexed_set = set()
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if collection_name not in collections:
            return indexed_set

        offset = None
        while True:
            records, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=["file_path", "content_hash"],
                with_vectors=False
            )
            for r in records:
                if r.payload:
                    fp = r.payload.get("file_path")
                    ch = r.payload.get("content_hash")
                    if fp and ch:
                        indexed_set.add((fp, ch))
            if next_offset is None or len(records) == 0:
                break
            offset = next_offset
    except Exception as e:
        print(f"Fehler beim Batch-Laden der Qdrant-Hashes: {e}")
    return indexed_set

# --- SPEZIALPARSER FÜR KICAD, DRC & RULES ---
def fast_parse_kicad(rel_path: str, raw_text: str) -> tuple[str, str]:
    ext = os.path.splitext(rel_path)[1].lower()
    filename = os.path.basename(rel_path)

    if ext == ".kicad_dru":
        category_tag = "KICAD_DRC_RULES"
        rules_found = re.findall(r'\(rule\s+"([^"]+)"', raw_text)
        rules_str = ", ".join(rules_found) if rules_found else "Benutzerdefinierte DRC-Regeln"
        markdown_content = f"""[TAG: KICAD_DRC_RULES]

# KiCad Custom DRC Rules: {filename}

- **Dateipfad:** `{rel_path}`
- **Erkannte Regeln:** `{rules_str}`

## Regel-Definition (Custom Design Rules)
{TB}lisp
{raw_text[:8000]}
{TB}
"""
        return category_tag, markdown_content

    elif ext == ".rules":
        category_tag = "FREEROUTING_RULES"
        markdown_content = f"""[TAG: FREEROUTING_RULES]

# FreeRouting Rules & Net Classes: {filename}

- **Dateipfad:** `{rel_path}`

## Routing-Konfiguration (Specctra Rules)
{TB}lisp
{raw_text[:8000]}
{TB}
"""
        return category_tag, markdown_content

    elif ext == ".kicad_mod":
        category_tag = "KICAD_FOOTPRINT"
        fp_match = re.search(r'\(footprint\s+"?([^"\s)]+)"?', raw_text)
        fp_name = fp_match.group(1) if fp_match else filename
        descr_match = re.search(r'\(descr\s+"([^"]+)"\)', raw_text)
        descr = descr_match.group(1) if descr_match else "Keine Beschreibung"
        tags_match = re.search(r'\(tags\s+"([^"]+)"\)', raw_text)
        tags = tags_match.group(1) if tags_match else "-"
        pads = re.findall(r'\(pad\s+"([^"]+)"\s+([^\s]+)\s+([^\s]+)', raw_text)
        pad_info = f"Gesamt: {len(pads)} Pads (" + ", ".join([f"Pad {p[0]} [{p[1]}]" for p in pads[:6]]) + ")" if pads else "Keine Pads"

        markdown_content = f"""[TAG: KICAD_FOOTPRINT]

# KiCad Footprint: {fp_name}

- **Dateipfad:** `{rel_path}`
- **Bauteil-Name:** `{fp_name}`
- **Beschreibung:** {descr}
- **Schlagwörter:** `{tags}`
- **Anschlüsse / Pads:** {pad_info}

## Verwendung für PCB-Layout (`pcbnew`)
Dieser Footprint wird in `pcbnew` über den Footprint-Bezeichner `{fp_name}` adressiert.
"""
        return category_tag, markdown_content

    elif ext == ".kicad_sym":
        category_tag = "KICAD_SYM"
        sym_matches = re.findall(r'\(symbol\s+"([^"]+)"', raw_text)
        main_sym = sym_matches[0] if sym_matches else filename
        descr_match = re.search(r'\(property\s+"Description"\s+"([^"]+)"', raw_text)
        descr = descr_match.group(1) if descr_match else "Keine Beschreibung"
        fp_ref_match = re.search(r'\(property\s+"Footprint"\s+"([^"]+)"', raw_text)
        fp_ref = fp_ref_match.group(1) if fp_ref_match else "-"
        pins = re.findall(r'\(pin\s+([^\s]+)\s+([^\s]+)', raw_text)
        pin_summary = f"{len(pins)} Pins vorhanden" if pins else "Keine expliziten Pins"

        markdown_content = f"""[TAG: KICAD_SYM]

# KiCad Schaltplan-Symbol: {main_sym}

- **Dateipfad:** `{rel_path}`
- **Symbol-Name:** `{main_sym}`
- **Beschreibung:** {descr}
- **Zugeordneter Footprint:** `{fp_ref}`
- **Pin-Konfiguration:** {pin_summary}

## Verwendung für SKiDL / Schaltungssynthese
Dieses Symbol steht für die Netzlistenerzeugung via SKiDL unter dem Bauteilnamen `{main_sym}` bereit.
"""
        return category_tag, markdown_content

    else:
        category_tag = "KICAD_EDA"
        markdown_content = f"[TAG: KICAD_EDA]\n\n# KiCad Datei: {filename}\n- **Dateipfad:** `{rel_path}`"
        return category_tag, markdown_content

# --- FEINGLIEDRIGER AST & DOMAIN PARSER ---
def hybrid_ast_llm_parse(rel_path: str, raw_code: str, active_model: str) -> tuple[str, str] | tuple[None, None]:
    filename = os.path.basename(rel_path)
    rel_lower = rel_path.lower()

    # 1. Dateinamen & Pfad-Filter (Build-/Doku-Skripte verwerfen)
    if filename in IGNORED_FILENAMES or any(p in rel_lower for p in IGNORED_PATH_PARTS):
        return None, None

    # 2. Differenzierung: Ist es SKiDL API Internals oder ein echtes Schaltungsbeispiel?
    is_skidl_api_internal = any(p in rel_lower for p in ["/src/skidl/", "skidl/skidl/", "/skidl/src/"])
    is_pcbnew_plugin = "pcbnew" in rel_lower or "action_plugin" in rel_lower

    # 3. Dynamisches Größen-Limit:
    # API- & Plugin-Dateien dürfen bis zu 150 KB groß sein (z.B. circuit.py mit 54 KB),
    # reine Schaltungsbeispiele bleiben für hohe Relevanz auf 50 KB gedeckelt.
    max_allowed_size = 150000 if (is_skidl_api_internal or is_pcbnew_plugin) else 50000
    if len(raw_code) > max_allowed_size:
        return None, None

    # 4. Syntax-Prüfung via AST
    try:
        tree = ast.parse(raw_code)
    except SyntaxError:
        return None, None

    has_skidl_instantiation = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            
            if func_name in ["Part", "Net", "Bus", "generate_netlist", "generate_pcb", "subcircuit", "Circuit"]:
                has_skidl_instantiation = True
                break

    # Falls weder API-Code noch ein Plugin noch ein SKiDL-Schaltplan vorliegt -> Überspringen
    if not (is_skidl_api_internal or is_pcbnew_plugin or has_skidl_instantiation):
        return None, None

    docstring = ast.get_docstring(tree) or "Kein Modul-Docstring vorhanden"
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # Bestimme exakte Kategorie-Tags
    if is_skidl_api_internal:
        category_tag = "SKIDL_API"
        tag_title = f"SKiDL API Modul: {filename}"
    elif is_pcbnew_plugin:
        category_tag = "KICAD_PCBNEW"
        tag_title = f"KiCad PCBNew Plugin: {filename}"
    else:
        category_tag = "SKIDL_GOLDEN_EXAMPLE"
        tag_title = f"Golden Example: {filename}"

    enrichment_prompt = f"""Analysiere diesen Python-Code für ein EDA/PCB-System:

{raw_code[:5000]}

ERSTELLE FOLGENDE DREI ABSCHNITTE AUF DEUTSCH:
1. Zweck: (Zusammenfassung der Funktion, API-Klassen oder Schaltung)
2. Hauptkomponenten / Schnittstellen: (Verwendete Klassen, Bauteile, Methoden oder Signale)
3. 3 Anwendungsfragen: (Drei typische Fragen eines Hardware-Entwicklers)

Verändere den Code NICHT."""

    try:
        response = ollama_client.chat(
            model=active_model,
            messages=[{'role': 'user', 'content': enrichment_prompt}],
            options={"num_ctx": DEFAULT_NUM_CTX}
        )
        enrichment_text = response['message']['content']
    except Exception:
        enrichment_text = f"**Zweck:** Python Modul ({rel_path})\n**Docstring:** {docstring}"

    code_snippet = raw_code[:10000] + ("\n# ... [Code gekürzt wegen Dateigröße]" if len(raw_code) > 10000 else "")

    markdown_content = f"""[TAG: {category_tag}]

# {tag_title}

- **Quelle/Dateipfad:** `{rel_path}`
- **Erkannte Importe:** `{', '.join(set(imports))}`

## Code-Analyse & Dokumentation
{enrichment_text}

## Validierter Quellcode (Python)
{TB}python
{code_snippet}
{TB}
"""
    return category_tag, markdown_content

# --- TASK MANAGER ---
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

    def start_background_job(self, files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model, mode="standard", mining_repo_url=""):
        with self.lock:
            if self.is_running:
                return "### ⚠️ Status: JOB LÄUFT BEREITS"
            
            self.is_running = True
            self.cancel_event.clear()
            self.log_messages = []
            self.processed_files = 0
            self.total_files = 0
            self.status_header = "🟢 Status: WIRD GESTARTET..."
            
            save_config({"last_model": selected_model, "last_category": category_key})

            self.current_thread = threading.Thread(
                target=self._run_job,
                args=(files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model, mode, mining_repo_url),
                daemon=True
            )
            self.current_thread.start()
            return f"### {self.status_header}"

    def _run_job(self, files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model, mode, mining_repo_url):
        temp_work_dir = None
        try:
            category_info = CATEGORIES.get(category_key, CATEGORIES["⚡ PCB & Hardware Design"])
            target_collection = category_info["collection"]
            system_prompt = category_info["system_prompt"]
            active_model = selected_model if selected_model else DEFAULT_MODEL

            session_id = str(uuid.uuid4())[:8]
            temp_work_dir = os.path.join("/tmp", f"rag_ingest_{session_id}")
            os.makedirs(temp_work_dir, exist_ok=True)

            files_to_process = []

            # REPOSITORY MINING MODUS VIA GIT CLONE
            if mode == "repo_mining":
                clean_repo_url = sanitize_url(mining_repo_url)
                if not clean_repo_url:
                    self.append_log("❌ Keine valide Repository-URL für das Mining angegeben.")
                    self.set_status("🔴 Status: BEENDET (Ungültige URL)")
                    return

                self.append_log(f"⛏️ Starte Git-Mining via `git clone`: {clean_repo_url}")
                mined_repo_dir = os.path.join(temp_work_dir, "mined_repo")
                
                res = subprocess.run(
                    ["git", "clone", "--depth", "1", clean_repo_url, mined_repo_dir],
                    capture_output=True, text=True
                )

                if res.returncode != 0:
                    self.append_log(f"❌ Git-Clone fehlgeschlagen: {res.stderr[:300]}")
                    self.set_status("🔴 Status: GIT CLONE FEHLER")
                    return

                self.append_log("   ↳ Repository erfolgreich geklont. Scanne Dateien...")
                for root, _, filenames in os.walk(mined_repo_dir):
                    if ".git" in root:
                        continue
                    for f in filenames:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in TEXT_EXTENSIONS:
                            full_p = os.path.join(root, f)
                            rel_p = os.path.relpath(full_p, mined_repo_dir)
                            repo_base_name = os.path.basename(clean_repo_url.rstrip("/"))
                            files_to_process.append((f"{repo_base_name}/{rel_p}", full_p))

            # STANDARD DATEI & GIT CRAWLER MODUS
            else:
                if files:
                    for file_item in files:
                        fpath = file_item.name if hasattr(file_item, 'name') else (file_item.get('name') if isinstance(file_item, dict) else str(file_item))
                        fname = os.path.basename(fpath)
                        ext = os.path.splitext(fname)[1].lower()

                        if ext == ".zip":
                            self.append_log(f"📦 Entpacke ZIP: {fname}...")
                            zip_extract_dir = os.path.join(temp_work_dir, f"zip_{uuid.uuid4()[:4]}")
                            with zipfile.ZipFile(fpath, 'r') as zip_ref:
                                zip_ref.extractall(zip_extract_dir)
                            extracted = collect_files_from_dir(zip_extract_dir)
                            files_to_process.extend(extracted)
                        elif ext in TEXT_EXTENSIONS:
                            files_to_process.append((fname, fpath))

                if scanned_repo_path and os.path.exists(scanned_repo_path) and selected_folders:
                    if "ALL_REPO" in selected_folders:
                        repo_files = collect_files_from_dir(scanned_repo_path)
                    else:
                        repo_files = []
                        for subfolder in selected_folders:
                            repo_files.extend(collect_files_from_dir(scanned_repo_path, target_subfolder=subfolder))
                    files_to_process.extend(repo_files)

                if selected_exts:
                    files_to_process = [
                        (rel_p, full_p) for rel_p, full_p in files_to_process
                        if os.path.splitext(rel_p)[1].lower() in selected_exts
                    ]

            files_to_process = list({rel_p: full_p for rel_p, full_p in files_to_process}.items())

            if not files_to_process:
                self.append_log("❌ Keine verwertbaren Dateien gefunden.")
                self.set_status("🔴 Status: BEENDET (Keine Dateien)")
                return

            self.total_files = len(files_to_process)
            self.append_log(f"📊 Gesamt: {self.total_files} Datei(en) bereit zur Indizierung.")
            existing_hashes = get_indexed_hashes_set(target_collection)
            self.append_log(f"   ↳ {len(existing_hashes)} bereits indizierte Datei(en) in '{target_collection}' übersprungen.\n")

            # MAIN INGESTION LOOP
            for idx, (rel_path, file_path) in enumerate(files_to_process, 1):
                if self.cancel_event.is_set():
                    self.append_log("🛑 Ingestion vorzeitig abgebrochen.")
                    self.set_status(f"🔴 Status: ABGEBROCHEN ({self.processed_files}/{self.total_files})")
                    return

                self.processed_files = idx
                self.set_status(f"🟢 Status: LÄUFT ({idx}/{self.total_files} - {os.path.basename(rel_path)})")

                raw_text = extract_text_from_file(file_path)
                if not raw_text.strip():
                    continue

                content_hash = calculate_sha256(raw_text)
                if (rel_path, content_hash) in existing_hashes:
                    self.append_log(f"[{idx}/{self.total_files}] ⏭️ Unverändert übersprungen: {rel_path}")
                    continue

                ext = os.path.splitext(rel_path)[1].lower()

                try:
                    if mode == "repo_mining" or ext == ".py":
                        self.append_log(f"[{idx}/{self.total_files}] Hybrid AST-LLM Parse: {rel_path}")
                        category_tag, processed_md = hybrid_ast_llm_parse(rel_path, raw_text, active_model)
                        if not processed_md:
                            self.append_log(f"   ⚠️ AST/Filter übersprungen (Kein SKiDL/API Code oder Build/Doku): {rel_path}")
                            continue

                    elif ext in {".kicad_mod", ".kicad_sym", ".kicad_pcb", ".kicad_sch", ".kicad_dru", ".rules"}:
                        self.append_log(f"[{idx}/{self.total_files}] Fast-Pass Parsing (ohne LLM): {rel_path}")
                        category_tag, processed_md = fast_parse_kicad(rel_path, raw_text)

                    else:
                        self.append_log(f"[{idx}/{self.total_files}] Verarbeite via LLM ({active_model}): {rel_path}")
                        full_user_prompt = f"DATEIPFAD: {rel_path}\nINHALT:\n{raw_text[:6000]}"
                        response = ollama_client.chat(
                            model=active_model,
                            messages=[
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': full_user_prompt}
                            ],
                            options={"num_ctx": DEFAULT_NUM_CTX}
                        )
                        processed_md = response['message']['content']
                        tag_match = re.search(r'\[(?:TAG|PAGE|CATEGORY):\s*([A-Z0-9_]+)\]', processed_md, re.IGNORECASE)
                        category_tag = tag_match.group(1).upper() if tag_match else "GENERAL"

                    if self.cancel_event.is_set():
                        return

                    # Abgesicherter Embedding-Aufruf mit Fallback bei Überlänge
                    embed_prompt = processed_md[:7500]
                    try:
                        embed_res = ollama_client.embeddings(
                            model=EMBED_MODEL, 
                            prompt=embed_prompt,
                            options={"num_ctx": DEFAULT_NUM_CTX}
                        )
                    except Exception as embed_err:
                        self.append_log(f"   ⚠️ Token-Limit erreicht ({embed_err}). Kürze Embedding-Input für {rel_path}...")
                        embed_res = ollama_client.embeddings(
                            model=EMBED_MODEL, 
                            prompt=processed_md[:3000],
                            options={"num_ctx": DEFAULT_NUM_CTX}
                        )

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
                    
                    self.append_log(f"   ✅ Indiziert in '{target_collection}' | **Tag: #{category_tag}**\n")

                except Exception as e:
                    self.append_log(f"   ❌ Fehler bei Verarbeitung: {str(e)}\n")

            self.append_log(f"\n🎉 Ingestion abgeschlossen! Dokumente sind in Collection '{target_collection}' verfügbar.")
            self.set_status(f"✅ Status: ABGESCHLOSSEN ({self.total_files}/{self.total_files})")

        except Exception as top_e:
            self.append_log(f"\n❌ Unerwarteter Systemfehler: {str(top_e)}")
            self.set_status("🔴 Status: FEHLER")
        finally:
            with self.lock:
                self.is_running = False
                self.cancel_event.clear()
            if temp_work_dir and os.path.exists(temp_work_dir):
                shutil.rmtree(temp_work_dir, ignore_errors=True)

task_manager = IngestTaskManager()

# --- HILFSFUNKTIONEN UI & GIT ---
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

def scan_github_repository(github_url, old_scanned_repo):
    if old_scanned_repo and os.path.exists(old_scanned_repo):
        shutil.rmtree(old_scanned_repo, ignore_errors=True)

    clean_url = sanitize_url(github_url)
    if not clean_url:
        yield (
            gr.update(choices=[], value=[], visible=False),
            gr.update(choices=[], value=[], visible=False),
            "",
            "❌ Bitte valide Repository URL angeben."
        )
        return

    task_manager.set_status("🟡 Status: SCANNE REPOSITORY...")
    task_manager.append_log(f"🌐 Starte Scan für Repository: {clean_url} ... Bitte warten.")

    yield (
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        f"🌐 Klone Repository {clean_url} im Hintergrund..."
    )

    session_id = str(uuid.uuid4())[:8]
    repo_dir = os.path.join("/tmp", f"scan_repo_{session_id}")

    res = subprocess.run(
        ["git", "clone", "--depth", "1", clean_url, repo_dir],
        capture_output=True, text=True
    )

    if res.returncode != 0:
        task_manager.set_status("🔴 Status: SCAN FEHLGESCHLAGEN")
        task_manager.append_log(f"❌ Git-Clone fehlgeschlagen: {res.stderr[:200]}")
        yield (
            gr.update(choices=[], value=[], visible=False),
            gr.update(choices=[], value=[], visible=False),
            "",
            f"❌ Git-Clone fehlgeschlagen:\n{res.stderr[:300]}"
        )
        return

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
    task_manager.append_log(log_msg)
    task_manager.set_status("⚪ Status: Inaktiv (Scan bereit)")

    yield (
        gr.update(choices=folder_choices, value=["ALL_REPO"], visible=True),
        gr.update(choices=ext_choices, value=[e[1] for e in ext_choices], visible=True),
        repo_dir,
        log_msg
    )

# --- STYLES & JAVASCRIPT ---
custom_css = """
footer { visibility: hidden; }
.row-stretch {
    display: flex !important;
    align-items: stretch !important;
}
.full-height-btn {
    height: 100% !important;
    min-height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-end !important;
}
.full-height-btn button {
    height: 100% !important;
    min-height: 100% !important;
    margin: 0 !important;
}
#log-textbox textarea {
    font-family: monospace;
    font-size: 0.85rem;
    scroll-behavior: smooth;
}
"""

autoscroll_js = """
function() {
    window.ragUserScrolledUp = false;
    
    const checkAndScroll = () => {
        const textarea = document.querySelector('#log-textbox textarea');
        if (!textarea) return;

        if (!textarea.dataset.scrollBound) {
            textarea.dataset.scrollBound = "true";
            textarea.addEventListener('scroll', () => {
                const distance = textarea.scrollHeight - textarea.scrollTop - textarea.clientHeight;
                window.ragUserScrolledUp = (distance > 50);
            });
        }

        if (!window.ragUserScrolledUp) {
            textarea.scrollTop = textarea.scrollHeight;
        }
    };

    setInterval(checkAndScroll, 300);
}
"""

# --- GRADIO GUI BUILDER ---
initial_model_choices, initial_default_model = get_ollama_models()
saved_cfg = load_config()
initial_default_category = saved_cfg.get("last_category", list(CATEGORIES.keys())[0])

with gr.Blocks(title="Universal RAG Control Center") as demo:
    repo_state = gr.State("")
    status_timer = gr.Timer(value=2.0)

    gr.Markdown("# 🏢 Universal RAG Ingestion Control Center")

    with gr.Row():
        with gr.Column(scale=1):
            status_banner = gr.Markdown("### ⚪ Status: Inaktiv")

            with gr.Row(elem_classes=["row-stretch"]):
                model_dropdown = gr.Dropdown(
                    choices=initial_model_choices,
                    value=initial_default_model,
                    label="LLM Modell (Gespeichert)",
                    interactive=True,
                    scale=4
                )
                refresh_models_btn = gr.Button("🔄", variant="secondary", scale=1, elem_classes=["full-height-btn"])

            category_dropdown = gr.Dropdown(
                choices=list(CATEGORIES.keys()),
                value=initial_default_category,
                label="Knowledge Collection",
                interactive=True
            )

            with gr.Tabs():
                with gr.Tab("📁 Dateiupload / ZIP"):
                    file_input = gr.File(
                        label="Dateien oder ZIP-Archiv hochladen", 
                        file_count="multiple"
                    )

                with gr.Tab("🌐 Git Repository Crawler"):
                    with gr.Row(elem_classes=["row-stretch"]):
                        github_input = gr.Textbox(
                            label="Repository URL",
                            placeholder="[https://gitlab.com/kicad/libraries/kicad-symbols.git](https://gitlab.com/kicad/libraries/kicad-symbols.git)",
                            scale=4
                        )
                        scan_repo_btn = gr.Button("🔍 Scannen", variant="secondary", scale=1, elem_classes=["full-height-btn"])

                    with gr.Row():
                        folder_checkboxes = gr.CheckboxGroup(label="Ordnerstruktur", choices=[], visible=False, interactive=True, scale=1)
                        ext_checkboxes = gr.CheckboxGroup(label="Dateiformate Filter", choices=[], visible=False, interactive=True, scale=1)

                with gr.Tab("⭐ EDA & Rule Mining"):
                    mining_repo_input = gr.Dropdown(
                        choices=[
                            ("SKiDL Haupt-Repository (Offiziell)", "[https://github.com/xesscorp/skidl](https://github.com/xesscorp/skidl)"),
                            ("KiCad Python Action Plugins", "[https://github.com/KiCad/kicad-python](https://github.com/KiCad/kicad-python)"),
                            ("Freerouting Java / Config Core", "[https://github.com/freerouting/freerouting](https://github.com/freerouting/freerouting)")
                        ],
                        value="[https://github.com/xesscorp/skidl](https://github.com/xesscorp/skidl)",
                        label="Ziel-Repository für EDA Mining",
                        allow_custom_value=True,
                        interactive=True
                    )
                    start_mining_btn = gr.Button("⛏️ Mining & Hybrid Ingestion Starten", variant="primary")

        with gr.Column(scale=1):
            status_output = gr.Textbox(
                label="Server Live-Protokoll", 
                interactive=False, 
                lines=24, 
                autoscroll=False,
                elem_id="log-textbox"
            )

            with gr.Row():
                start_btn = gr.Button("🚀 Ingest Starten", variant="primary", scale=3)
                stop_btn = gr.Button("🛑 Abbrechen", variant="stop", scale=2)

    status_timer.tick(
        fn=task_manager.get_ui_snapshot,
        outputs=[status_output, status_banner],
        show_progress="hidden"
    )

    model_dropdown.change(fn=update_model_preference, inputs=[model_dropdown])
    category_dropdown.change(fn=update_category_preference, inputs=[category_dropdown])
    refresh_models_btn.click(fn=lambda: gr.Dropdown(choices=get_ollama_models()[0]), outputs=[model_dropdown])

    scan_repo_btn.click(
        fn=scan_github_repository,
        inputs=[github_input, repo_state],
        outputs=[folder_checkboxes, ext_checkboxes, repo_state, status_output]
    )

    folder_checkboxes.change(fn=handle_folder_selection, inputs=[folder_checkboxes], outputs=[folder_checkboxes])

    start_btn.click(
        fn=lambda files, repo, f_cb, e_cb, cat, mod: task_manager.start_background_job(
            files, repo, f_cb, e_cb, cat, mod, mode="standard"
        ),
        inputs=[file_input, repo_state, folder_checkboxes, ext_checkboxes, category_dropdown, model_dropdown],
        outputs=[status_banner]
    )

    start_mining_btn.click(
        fn=lambda cat, mod, url: task_manager.start_background_job(
            None, None, None, None, cat, mod, mode="repo_mining", mining_repo_url=url
        ),
        inputs=[category_dropdown, model_dropdown, mining_repo_input],
        outputs=[status_banner]
    )

    stop_btn.click(fn=task_manager.request_cancel, outputs=[status_banner])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7861,
        css=custom_css,
        js=autoscroll_js
    )