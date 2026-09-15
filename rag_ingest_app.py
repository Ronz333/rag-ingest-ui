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

# Dateien & Pfade, die für Schaltungssynthese/Rules reines Rauschen sind
IGNORED_FILENAMES = {"setup.py", "conftest.py", "__init__.py", "pyproject.toml", "pom.xml"}
IGNORED_PATH_PARTS = ["/docs/", "/tests/", "/build/", "/dist/", "/.github/", "/site-packages/"]

TEXT_EXTENSIONS = {
    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch", ".kicad_prj", ".kicad_dru", ".kicad_pro",
    ".sch", ".net", ".cir", ".lib", ".mod", ".sym", ".spice", ".sub", ".mcb", ".dxf", ".dsn", ".ses", ".rules",
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".c", ".h", ".cpp", ".hpp", ".java",
    ".js", ".ts", ".html", ".css", ".rst", ".csv", ".ini", ".conf", ".sh",
    ".pdf"
}

CATEGORIES = {
    "⚡ PCB & Hardware Design": {
        "collection": "pcb_knowledge_base",
        "system_prompt": """Du bist ein Spezial-Ingestion-Agent für ein EDA/PCB RAG-System.
Deine Aufgabe ist es, Quellcode, KiCad-Dateien, DRC/ERC-Regeln und SKiDL-Skripte präzise zu analysieren und für KI-gestützte Hardware-Entwicklung aufzubereiten.

STRIKTE REGELN:
1. ERSTE ZEILE: Verwende ZWINGEND eines der folgenden Kategorie-Tags in eckigen Klammern:
   - [TAG: SKIDL_GOLDEN_EXAMPLE] (Lauffähiger SKiDL-Schaltplan/Subcircuit)
   - [TAG: SKIDL_API] (SKIDL Bibliotheks-Internals & SDK API)
   - [TAG: KICAD_PCBNEW_API] (KiCad Python pcbnew Layout-Steuerung)
   - [TAG: KICAD_DRC_RULES] (KiCad DRC Custom Rules & ERC Prüfvorschriften)
   - [TAG: FREEROUTING_RULES] (FreeRouting DSN / Autorouter Konfiguration)
   - [TAG: KICAD_FOOTPRINT] / [TAG: KICAD_SYM] (Symbol & Footprint Bibs)
2. FAKTEN-TREUE: Ergänze nur strukturelle Erklärung und Anwendungsfragen. Verändere den Quellcode/Regel-Inhalt NICHT!
3. CODE-INTEGRITÄT: Bette Quellcode/Regeln exakt in Markdown-Codeblöcken ein."""
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

# --- DOMÄNENSPEZIFISCHE PARSER ---

def fast_parse_kicad(rel_path: str, raw_text: str) -> tuple[str, str]:
    """Schnell-Parsing ohne LLM-Overhead für hochstrukturierte KiCad S-Expression Dateiformate."""
    ext = os.path.splitext(rel_path)[1].lower()
    filename = os.path.basename(rel_path)

    # 1. KiCad Footprints (.kicad_mod)
    if ext == ".kicad_mod":
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
Dieser Footprint wird in `pcbnew` und SKiDL über den Footprint-Bezeichner `{fp_name}` adressiert.
"""
        return category_tag, markdown_content

    # 2. KiCad Schaltplan-Symbole (.kicad_sym)
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

    # 3. KiCad Custom DRC Rules (.kicad_dru)
    elif ext == ".kicad_dru":
        category_tag = "KICAD_DRC_RULES"
        rules_found = re.findall(r'\(rule\s+"([^"]+)"', raw_text)
        rule_names = ", ".join(rules_found) if rules_found else "Standard-DRC Regelset"

        markdown_content = f"""[TAG: KICAD_DRC_RULES]

# KiCad Custom DRC Rules Definition: {filename}

- **Dateipfad:** `{rel_path}`
- **Enthaltene Regeln:** `{rule_names}`

## Regel-Syntax & Einschränkungen (KiCad DRC)
```lisp
{raw_text[:8000]}
```
"""
        return category_tag, markdown_content

    else:
        category_tag = "KICAD_EDA"
        markdown_content = f"[TAG: KICAD_EDA]\n\n# KiCad Datei: {filename}\n- **Dateipfad:** `{rel_path}`"
        return category_tag, markdown_content

def smart_eda_parser(rel_path: str, raw_text: str, active_model: str) -> tuple[str, str] | tuple[None, None]:
    """Intelligenter Hybrid-Parser für SKiDL, KiCad Python (pcbnew), Custom DRC und FreeRouting Code."""
    filename = os.path.basename(rel_path)
    rel_lower = rel_path.lower()
    ext = os.path.splitext(rel_path)[1].lower()

    # 1. Aussortieren von Rauschen (Build-Skripte, Doku, Setup-Dateien)
    if filename in IGNORED_FILENAMES or any(p in rel_lower for p in IGNORED_PATH_PARTS):
        return None, None

    # Code-Dateien über 60 KB überspringen (oft autogenerierte riesige C++ oder Layout-Dumps)
    if len(raw_text) > 60000:
        return None, None

    # --- DOMÄNEN-IDENTIFIKATION ---
    category_tag = None
    enrichment_instructions = ""

    # FALL A: KiCad Custom DRC Rules / Python DRC Scripts
    if ext == ".kicad_dru" or "drc" in rel_lower or "erc" in rel_lower or "check_rule" in raw_text.lower():
        category_tag = "KICAD_DRC_RULES"
        enrichment_instructions = """Analysiere diese DRC/ERC Layout-Regeln oder Prüfskript:
1. Regelzweck: (Welche Abstände, Layer, Vias oder PCB-Fertigungsgrenzen werden geprüft?)
2. Parameter & Constraints: (Konkrete Werte wie min_clearance, hole_size, track_width etc.)
3. 3 Entwicklerfragen: (Drei Fragen zu PCB-Designregeln, die dieser Code beantwortet)"""

    # FALL B: FreeRouting Rules / DSN Autorouter Config
    elif ext in {".dsn", ".rules"} or "freerouting" in rel_lower:
        category_tag = "FREEROUTING_RULES"
        enrichment_instructions = """Analysiere diese FreeRouting Autorouter Regelkonfiguration:
1. Zweck: (Welche Netzklassen, Clearance-Matrizen oder Routing-Passes werden definiert?)
2. Routing-Parameter: (Spurbreiten, Via-Typen, Layer-Richtungen)
3. 3 Entwicklerfragen: (Drei typische Fragen zur automatischen Entflechtung)"""

    # FALL C: Python-Code Differenzierung (SKiDL vs. KiCad pcbnew)
    elif ext == ".py":
        try:
            tree = ast.parse(raw_text)
        except SyntaxError:
            return None, None  # Fehlerhaften Code sofort verwerfen

        code_lower = raw_text.lower()
        
        # C1: SKiDL Code Unterscheidung (API / Internals VS. Golden Example Schaltung)
        if "skidl" in code_lower:
            # Prüfe, ob es sich um den SKiDL-Bibliotheks-Quellcode selbst handelt (z.B. src/skidl/...)
            is_internal_api = any(p in rel_lower for p in ["/src/", "skidl/skidl/", "package"])
            
            # Prüfe via AST, ob echte Bauteile/Netze erzeugt werden
            has_circuit_elements = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    if func_name in ["Part", "Net", "Bus", "generate_netlist", "generate_pcb"]:
                        has_circuit_elements = True
                        break

            if is_internal_api or not has_circuit_elements:
                category_tag = "SKIDL_API"
                enrichment_instructions = """Analysiere dieses SKiDL SDK/API Modul:
1. Zweck & Klasse: (Welche SKiDL-Kernklasse oder Hilfsfunktion wird bereitgestellt?)
2. Methoden & Parameter: (Wichtigste Schnittstellen für Entwickler)
3. 3 Entwicklerfragen: (Fragen zur Erweiterung oder Verwendung der SKiDL API)"""
            else:
                category_tag = "SKIDL_GOLDEN_EXAMPLE"
                enrichment_instructions = """Analysiere diesen funktionsfähigen SKiDL Schaltungs-Code:
1. Schaltungszweck: (Was baut diese Schaltung z.B. ESP32 Power-Management, USB ESD Schutz?)
2. Bauteile & Verschaltung: (Verwendete ICs, Transistoren, Widerstände und Busse)
3. 3 Entwicklerfragen: (Drei konkrete Hardware-Design-Fragen, die dieser Code als Vorlage löst)"""

        # C2: KiCad PCBNEW Automation / Action Plugin Python Code
        elif "pcbnew" in code_lower or "board" in code_lower:
            category_tag = "KICAD_PCBNEW_API"
            enrichment_instructions = """Analysiere dieses KiCad pcbnew Python-Layout-Skript:
1. Zweck: (Welche Layout-Automatisierung wird durchgeführt, z.B. Bounding-Box, Via-Stitching, Panelisierung?)
2. Verwendete pcbnew API Methoden: (z.B. GetBoard(), TRACK(), Add(), FOOTPRINT)
3. 3 Entwicklerfragen: (Fragen zur Python-Skriptierung in KiCad pcbnew)"""

    if not category_tag:
        return None, None  # Datei passt in keine relevante EDA-Kategorie

    # --- LLM ANREICHERUNG (GENERIERUNG VON METADATEN & QA-PAAREN) ---
    enrichment_prompt = f"""{enrichment_instructions}

QUELLCODE / REGELN:
{raw_text[:4500]}

VERÄNDERE DEN CODE NICHT. ANTWORTE AUF DEUTSCH IM MARKDOWN-FORMAT."""

    try:
        response = ollama_client.chat(
            model=active_model,
            messages=[{'role': 'user', 'content': enrichment_prompt}],
            options={"num_ctx": DEFAULT_NUM_CTX}
        )
        enrichment_text = response['message']['content']
    except Exception:
        enrichment_text = f"**Zweck:** EDA Modul ({rel_path})\n**Kategorie:** {category_tag}"

    code_snippet = raw_text[:10000] + ("\n# ... [Inhalt gekürzt wegen Dateigröße]" if len(raw_text) > 10000 else "")

    markdown_content = f"""[TAG: {category_tag}]

# {category_tag}: {os.path.basename(rel_path)}

- **Dateipfad / Quelle:** `{rel_path}`

## Analyse & Entwickler-Dokumentation
{enrichment_text}

## Original Quellcode / Regel-Spezifikation
{TB}{ext.replace('.', '')}
{code_snippet}
{TB}
"""
    return category_tag, markdown_content

# --- THREAD-SICHERER TASK MANAGER ---
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
                    # ROUTE A: KiCad Strukturdaten (Symbole, Footprints, Custom DRC)
                    if ext in {".kicad_mod", ".kicad_sym", ".kicad_dru"}:
                        self.append_log(f"[{idx}/{self.total_files}] Fast-Pass S-Expr Parse: {rel_path}")
                        category_tag, processed_md = fast_parse_kicad(rel_path, raw_text)

                    # ROUTE B: Smart EDA Parser (SKiDL, pcbnew, DRC Prüfskripte, FreeRouting)
                    elif mode == "repo_mining" or ext in {".py", ".dsn", ".rules"}:
                        self.append_log(f"[{idx}/{self.total_files}] Smart EDA Hybrid Parse: {rel_path}")
                        category_tag, processed_md = smart_eda_parser(rel_path, raw_text, active_model)
                        if not processed_md:
                            self.append_log(f"   ⚠️ Nicht-relevante oder Build-Datei übersprungen: {rel_path}")
                            continue

                    # ROUTE C: Allgemeines LLM-Parsing
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

                    # Abgesicherter Embedding-Aufruf mit Fallback bei Überlänge (Token-Limit Guardrail)
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
                            placeholder="https://gitlab.com/kicad/libraries/kicad-symbols.git",
                            scale=4
                        )
                        scan_repo_btn = gr.Button("🔍 Scannen", variant="secondary", scale=1, elem_classes=["full-height-btn"])

                    with gr.Row():
                        folder_checkboxes = gr.CheckboxGroup(label="Ordnerstruktur", choices=[], visible=False, interactive=True, scale=1)
                        ext_checkboxes = gr.CheckboxGroup(label="Dateiformate Filter", choices=[], visible=False, interactive=True, scale=1)

                with gr.Tab("⭐ EDA & Hardware Mining"):
                    mining_repo_input = gr.Dropdown(
                        choices=[
                            ("SKiDL Haupt-Repository (Offiziell & Examples)", "https://github.com/xesscorp/skidl"),
                            ("KiCad Python Action Plugins & Scripts", "https://github.com/KiCad/kicad-python"),
                            ("KiCad Custom DRC Rules Examples", "https://github.com/KiCad/kicad-custom-rules"),
                            ("KiCad CLI & ERC/DRC Automation Tools", "https://github.com/maia-hdl/kicad-cli-tools"),
                            ("Freerouting Java Core & Rules", "https://github.com/freerouting/freerouting")
                        ],
                        value="https://github.com/xesscorp/skidl",
                        label="Ziel-Repository für Mining (SKiDL, KiCad, DRC/ERC, FreeRouting)",
                        allow_custom_value=True,
                        interactive=True
                    )
                    start_mining_btn = gr.Button("⛏️ EDA Git-Mining & Hybrid Ingestion Starten", variant="primary")

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
```eoc

### Key Improvements in this Release:

1. **Intelligent Multi-Domain Classification (`smart_eda_parser`)**:
   - **`[TAG: SKIDL_GOLDEN_EXAMPLE]`**: Assigned only to scripts that actively instantiate circuit elements (`Part`, `Net`, `Bus`).
   - **`[TAG: SKIDL_API]`**: Automatically routes internal library core files (e.g., `src/skidl/interface.py`) here, asking the LLM for developer SDK documentation instead of treating it as a circuit.
   - **`[TAG: KICAD_PCBNEW_API]`**: Handles KiCad Python layout automation and Action Plugins (`pcbnew.GetBoard()`).
   - **`[TAG: KICAD_DRC_RULES]`**: Parses KiCad custom DRC rule files (`.kicad_dru`) and Python DRC verification scripts.
   - **`[TAG: FREEROUTING_RULES]`**: Handles FreeRouting `.dsn` files and routing rule definitions.

2. **Expanded Repository Mining Presets**:
   - Added standard presets for SKiDL, KiCad Python Action Plugins, KiCad Custom DRC Rules, KiCad CLI tools, and FreeRouting Core.