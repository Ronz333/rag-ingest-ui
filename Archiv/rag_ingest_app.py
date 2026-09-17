import os
import re
import uuid
import time
import shutil
import zipfile
import hashlib
import json
import ast
import warnings
import subprocess
import multiprocessing
import gradio as gr
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader

# --- KONFIGURATION & KONSTANTEN ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q5_K_M")
CONFIG_FILE = "/tmp/rag_ingest_config.json"
TB = "```"

IGNORED_FILENAMES = {"setup.py", "conftest.py", "__init__.py"}
IGNORED_PATH_PARTS = ["/docs/", "/tests/", "/build/", "/dist/", "/.git/"]

# Ergänzt um Java-, Gradle-, Properties- und Specctra DSN/SES Formate für FreeRouting
TEXT_EXTENSIONS = {
    ".kicad_sym", ".kicad_mod", ".kicad_pcb", ".kicad_sch", ".kicad_prj", ".kicad_dru",
    ".sch", ".net", ".cir", ".lib", ".mod", ".sym", ".spice", ".sub", ".mcb", ".dxf",
    ".rules", ".dsn", ".ses",
    ".java", ".gradle", ".properties", ".xml",
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".c", ".h", ".cpp", ".hpp",
    ".js", ".ts", ".html", ".css", ".rst", ".csv", ".ini", ".conf", ".sh",
    ".pdf"
}

CATEGORIES = {
    "⚡ PCB & Hardware Design": {
        "collection": "pcb_knowledge_base",
        "system_prompt": """Du bist ein Ingestion-Agent für ein EDA/PCB-RAG-System zur Unterstützung eines autonomen AI-PCB-Designers.
Analysiere den Inhalt (z. B. SKiDL-Code, KiCad-Footprints/Symbols, DRC-Rules, FreeRouting-Java-Code, Specctra DSN/SES) strikt faktengetreu und erstelle ein strukturiertes Markdown-Dokument.

STRIKTE REGELN:
1. ERSTE ZEILE: Zwingend ein exaktes Kategorie-Schlagwort in eckigen Klammern, z. B.:
   [TAG: SKIDL_API], [TAG: FREEROUTING_JAVA], [TAG: KICAD_FOOTPRINT], [TAG: KICAD_SYM], [TAG: KICAD_DRC_RULES], [TAG: FREEROUTING_RULES], [TAG: SPECCTRA_DSN].
2. ABSOLUTES HALLUZINATIONSVERBOT: Verarbeite Dateinamen und Inhalt strikt faktengetreu.
3. INHALTS-INTEGRITÄT: Bette bereitgestellten Quellcode, S-Expressions oder Routing-Regeln 1:1 im passenden Codeblock ein.
4. EXHAUSTIVE AGENTEN-FRAGEN: Erstelle eine VOLLSTÄNDIGE, UNBEGRENZTE Liste von Entwickler- und Agenten-Steuerungsfragen.
   WICHTIG: Formuliere AUSSCHLIESSLICH "Wie steuere / nutze / erstelle / vergleiche / konfiguriere ich X mit dieser API oder Spezifikation?"-Fragen (z. B. zu Methoden, Parametern, Flags, Operatoren, Context-Managern, Scoping, State-Reset, Footprint-Formaten oder Clearance-Limits). Stelle KEINE Fragen zum temporären Zustand einer konkreten Beispielschaltung (z. B. NICHT: "Wie viele Bauteile sind in der Schaltung enthalten?")."""
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

# --- LOGGING HELPER MIT LOKALER ZEITZEILE ---
def log_msg(log_list, text: str):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    log_list.append(f"[{timestamp}] {text}")

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

def get_indexed_hashes_set(qdrant_client, collection_name: str) -> set:
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

def ensure_qdrant_collection(qdrant_client, collection_name: str, vector_size: int):
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

def unload_ollama_model(ollama_client, model_name: str):
    """Zwingt Ollama dazu, ein geladenes Modell umgehend aus dem VRAM zu entladen."""
    try:
        ollama_client.chat(model=model_name, messages=[], keep_alive=0)
    except Exception:
        pass

def calculate_dynamic_num_ctx(text_len: int, max_limit: int = 32768) -> int:
    """Berechnet ein optimales Kontextfenster (Power of 2) basierend auf der Textlänge."""
    estimated_tokens = int(text_len / 3.0) + 512
    num_ctx = 2048
    while num_ctx < estimated_tokens and num_ctx < max_limit:
        num_ctx *= 2
    return min(num_ctx, max_limit)

def smart_markdown_chunking(text: str, max_chars: int = 12000, overlap_chars: int = 1200) -> list[str]:
    """Trennt Markdown strukturbewusst an Überschriften/Absätzen und hält Kontext via Overlap."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r'(\n(?=#{1,4} )|\n\n+)', text)
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if not p:
            continue
        if len(current_chunk) + len(p) <= max_chars:
            current_chunk += p
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            overlap_start = max(0, len(current_chunk) - overlap_chars)
            overlap_text = current_chunk[overlap_start:]
            
            if len(p) > max_chars:
                p_chunks = [p[i:i + max_chars] for i in range(0, len(p), max_chars - overlap_chars)]
                for pc in p_chunks:
                    chunks.append(pc.strip())
                current_chunk = ""
            else:
                current_chunk = overlap_text + p

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# --- SPEZIALPARSER FÜR KICAD, DRC, RULES & SPECCTRA ---
def fast_parse_kicad(rel_path: str, raw_text: str, max_code_len: int) -> tuple[str, str]:
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
{raw_text[:max_code_len]}
{TB}
"""
        return category_tag, markdown_content

    elif ext in {".rules", ".dsn", ".ses"}:
        category_tag = "FREEROUTING_RULES" if ext == ".rules" else "SPECCTRA_DSN"
        markdown_content = f"""[TAG: {category_tag}]

# Specctra / FreeRouting Datei: {filename}

- **Dateipfad:** `{rel_path}`

## Struktur-Definition
{TB}lisp
{raw_text[:max_code_len]}
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

# --- FEINGLIEDRIGER JAVA PARSER (FREEROUTING / EDA CORE) ---
def hybrid_java_llm_parse(rel_path: str, raw_code: str, active_model: str, ollama_client, num_ctx: int, max_code_len: int) -> tuple[str, str] | tuple[None, None]:
    filename = os.path.basename(rel_path)
    rel_lower = rel_path.lower()

    if filename in IGNORED_FILENAMES or any(p in rel_lower for p in IGNORED_PATH_PARTS):
        return None, None

    if len(raw_code) > 300000:
        return None, None

    # Structural Extraction für Java
    package_match = re.search(r'package\s+([a-zA-Z0-9_.]+);', raw_code)
    package_name = package_match.group(1) if package_match else "default"

    classes = re.findall(r'(?:public|protected|private)?\s*(?:static\s+)?(?:class|interface|enum)\s+([A-Za-z0-9_]+)', raw_code)
    methods = re.findall(r'(?:public|protected)\s+(?:[A-Za-z0-9_<>\[\]]+\s+)+([A-Za-z0-9_]+)\s*\([^\)]*\)', raw_code)

    extracted_elements = set()
    for c in classes:
        extracted_elements.add(f"class/interface:{c}")
    for m in methods:
        if m not in {"if", "for", "while", "switch", "catch"}:
            extracted_elements.add(f"method:{m}()")

    is_freerouting = "freerouting" in rel_lower or "autoroute" in rel_lower or "board" in rel_lower or "routing" in rel_lower
    category_tag = "FREEROUTING_JAVA" if is_freerouting else "JAVA_SOURCE"
    tag_title = f"FreeRouting Java Modul: {filename}" if is_freerouting else f"Java Modul: {filename}"

    elements_checklist = ", ".join(sorted(list(extracted_elements))) if extracted_elements else "Keine spezifischen Methoden extrahiert"
    full_code_for_llm = raw_code[:max_code_len]

    enrichment_prompt = f"""Analysiere diesen Java-Quellcode (FreeRouting Autorouter / EDA Core) für ein autonomes AI-PCB-Design-System:

{full_code_for_llm}

ERSTELLE FOLGENDE ABSCHNITTE AUF DEUTSCH:
1. Zweck: (Strukturierte Zusammenfassung der Funktion, Modulrolle, Datenstruktur oder Routing-Algorithmus)
2. Hauptkomponenten & Schnittstellen: (Liste aller Klassen, Interfaces, Hauptmethoden, Parameter, Datenstrukturen)
3. Autonome Agenten- & API-Anwendungsfragen: Erstelle eine VOLLSTÄNDIGE, ERSCHÖPFENDE Liste präziser Steuerungs- und Programmierfragen.

STRIKTE REGELN FÜR ABSCHNITT 3:
a) ERZWUNGENE JAVA-ELEMENT-ABDECKUNG: Generiere ZWINGEND für JEDES der folgenden im Code erkannten Elemente mindestens eine spezifische Steuerungs-/Anwendungsfrage:
   [{elements_checklist}]
b) MANDATORISCHE STEUERUNGSDIMENSIONEN: Deckt gezielt folgende Bereiche ab, sofern im Code vorhanden:
   - Routing-Algorithmen & Heuristiken (z. B. Maze Routing, Push & Shove, Via-Placement, Clearance Check)
   - Board- & Geometrie-Datenstrukturen (z. B. Board, Item, Net, Trace, Padstack, Shape)
   - Konfiguration & Parameterisierung (z. B. Autoroute Settings, DRC Rules, Layer Rules)
   - Import & Export Schnittstellen (z. B. Specctra DSN Import, SES Export, Rules Parsing)
c) FORMULIERUNG: Verwende AUSSCHLIESSLICH "Wie steuere / nutze / erstelle / vergleiche / konfiguriere / starte ich X mit dieser Java API?"-Fragen.

Verändere den Quelltext/Inhalt NICHT."""

    try:
        response = ollama_client.chat(
            model=active_model,
            messages=[{'role': 'user', 'content': enrichment_prompt}],
            options={"num_ctx": num_ctx}
        )
        enrichment_text = response['message']['content']
    except Exception:
        enrichment_text = f"**Zweck:** Java Modul (`{rel_path}`)\n**Package:** `{package_name}`"

    code_snippet = raw_code[:max_code_len] + ("\n// ... [Code gekürzt wegen Dateigröße]" if len(raw_code) > max_code_len else "")

    markdown_content = f"""[TAG: {category_tag}]

# {tag_title}

- **Quelle/Dateipfad:** `{rel_path}`
- **Package:** `{package_name}`

## Code-Analyse & Dokumentation
{enrichment_text}

## Validierter Quellcode (Java)
{TB}java
{code_snippet}
{TB}
"""
    return category_tag, markdown_content

# --- FEINGLIEDRIGER PYTHON AST PARSER ---
def hybrid_ast_llm_parse(rel_path: str, raw_code: str, active_model: str, ollama_client, num_ctx: int, max_code_len: int) -> tuple[str, str] | tuple[None, None]:
    filename = os.path.basename(rel_path)
    rel_lower = rel_path.lower()

    if filename in IGNORED_FILENAMES or any(p in rel_lower for p in IGNORED_PATH_PARTS):
        return None, None

    if len(raw_code) > 300000:
        return None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(raw_code)
    except Exception:
        return None, None

    extracted_elements = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("__") or node.name in ["__enter__", "__exit__", "__iadd__", "__isub__", "__init__"]:
                extracted_elements.add(f"method:{node.name}()")
        elif isinstance(node, ast.ClassDef):
            extracted_elements.add(f"class:{node.name}")

    docstring = ast.get_docstring(tree) or "Kein Modul-Docstring vorhanden"
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    if "skidl" in rel_lower:
        category_tag = "SKIDL_API"
        tag_title = f"SKiDL API Modul: {filename}"
    elif "pcbnew" in rel_lower or "kicad" in rel_lower:
        category_tag = "KICAD_PCBNEW"
        tag_title = f"KiCad / PCBNew Plugin: {filename}"
    else:
        category_tag = "PYTHON_MODULE"
        tag_title = f"Python Modul: {filename}"

    elements_checklist = ", ".join(sorted(list(extracted_elements))) if extracted_elements else "Keine spezifischen Methoden extrahiert"
    full_code_for_llm = raw_code[:max_code_len]

    enrichment_prompt = f"""Analysiere diesen Quellcode / diese EDA-Spezifikation für ein autonomes AI-PCB-Design-System:

{full_code_for_llm}

ERSTELLE FOLGENDE ABSCHNITTE AUF DEUTSCH:
1. Zweck: (Strukturierte Zusammenfassung der Funktion, Modulrolle oder Regeldefinition)
2. Hauptkomponenten & Schnittstellen: (Liste aller Klassen, Methoden, Operatoren, Parameter, Pads, Pin-Belegungen oder Layout-Regeln)
3. Autonome Agenten- & API-Anwendungsfragen: Erstelle eine VOLLSTÄNDIGE, ERSCHÖPFENDE Liste präziser Steuerungs- und Programmierfragen.

STRIKTE REGELN FÜR ABSCHNITT 3:
a) ERZWUNGENE AST-ABDECKUNG: Generiere ZWINGEND für JEDES der folgenden im Code erkannten Elemente mindestens eine spezifische Steuerungs-/Anwendungsfrage:
   [{elements_checklist}]
b) FORMULIERUNG: Verwende AUSSCHLIESSLICH "Wie steuere / nutze / erstelle / vergleiche / konfiguriere ich X mit dieser API?"-Fragen.

Verändere den Quelltext/Inhalt NICHT."""

    try:
        response = ollama_client.chat(
            model=active_model,
            messages=[{'role': 'user', 'content': enrichment_prompt}],
            options={"num_ctx": num_ctx}
        )
        enrichment_text = response['message']['content']
    except Exception:
        enrichment_text = f"**Zweck:** Python Modul ({rel_path})\n**Docstring:** {docstring}"

    code_snippet = raw_code[:max_code_len] + ("\n# ... [Code gekürzt wegen Dateigröße]" if len(raw_code) > max_code_len else "")

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

# --- PROZESS WORKER MIT DISPATCH-PARSING, BATCHING & STRUCTURAL CHUNKING ---
def worker_process_entry(log_list, status_dict, files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model, mode, mining_repo_url, num_ctx, max_embed_chars, batch_size):
    temp_work_dir = None
    try:
        ollama_worker = ollama.Client(host=OLLAMA_HOST)
        qdrant_worker = QdrantClient(url=QDRANT_HOST)

        category_info = CATEGORIES.get(category_key, CATEGORIES["⚡ PCB & Hardware Design"])
        target_collection = category_info["collection"]
        system_prompt = category_info["system_prompt"]
        active_model = selected_model if selected_model else DEFAULT_MODEL

        session_id = str(uuid.uuid4())[:8]
        temp_work_dir = os.path.join("/tmp", f"rag_ingest_{session_id}")
        os.makedirs(temp_work_dir, exist_ok=True)

        files_to_process = []

        if mode == "repo_mining":
            clean_repo_url = sanitize_url(mining_repo_url)
            if not clean_repo_url:
                log_msg(log_list, "❌ Keine valide Repository-URL für das Mining angegeben.")
                status_dict["header"] = "🔴 Status: BEENDET (Ungültige URL)"
                return

            log_msg(log_list, f"⛏️ Starte Git-Mining via `git clone`: {clean_repo_url}")
            mined_repo_dir = os.path.join(temp_work_dir, "mined_repo")
            
            res = subprocess.run(
                ["git", "clone", "--depth", "1", clean_repo_url, mined_repo_dir],
                capture_output=True, text=True
            )

            if res.returncode != 0:
                log_msg(log_list, f"❌ Git-Clone fehlgeschlagen: {res.stderr[:300]}")
                status_dict["header"] = "🔴 Status: GIT CLONE FEHLER"
                return

            log_msg(log_list, "   ↳ Repository erfolgreich geklont. Scanne Dateien...")
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

        else:
            if files:
                for file_item in files:
                    fpath = file_item.name if hasattr(file_item, 'name') else (file_item.get('name') if isinstance(file_item, dict) else str(file_item))
                    fname = os.path.basename(fpath)
                    ext = os.path.splitext(fname)[1].lower()

                    if ext == ".zip":
                        log_msg(log_list, f"📦 Entpacke ZIP: {fname}...")
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
            log_msg(log_list, "❌ Keine verwertbaren Dateien gefunden.")
            status_dict["header"] = "🔴 Status: BEENDET (Keine Dateien)"
            return

        total_files = len(files_to_process)
        total_batches = (total_files + batch_size - 1) // batch_size
        log_msg(log_list, f"📊 Gesamt: {total_files} Datei(en) bereit zur Indizierung.")
        log_msg(log_list, f"⚙️ Modus: Phase-Batching ({batch_size} Dateien/Batch -> {total_batches} Batches total)")
        log_msg(log_list, f"⚙️ Konfiguration: Context={num_ctx} Tokens | Max Embed Chars={max_embed_chars}")
        
        existing_hashes = get_indexed_hashes_set(qdrant_worker, target_collection)
        log_msg(log_list, f"   ↳ {len(existing_hashes)} bereits indizierte Datei(en) in '{target_collection}' übersprungen.\n")

        # Outer Loop: Batching nach Benutzerkonfiguration
        for batch_idx in range(0, total_files, batch_size):
            chunk = files_to_process[batch_idx : batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1

            log_msg(log_list, f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            log_msg(log_list, f"📦 STARTE BATCH {current_batch_num}/{total_batches} ({len(chunk)} Dateien in diesem Durchlauf)")
            log_msg(log_list, f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            batch_prepared_items = []
            used_llm_in_this_batch = False

            # --- PHASE A: TEXTANALYSE & MARKDOWN GENERIERUNG ---
            log_msg(log_list, f"🧠 Phase A [Batch {current_batch_num}]: Analysiere und strukturiere Dokumente...")
            for idx_in_chunk, (rel_path, file_path) in enumerate(chunk, 1):
                global_idx = batch_idx + idx_in_chunk
                status_dict["header"] = f"🟢 Status: LÄUFT (Batch {current_batch_num}/{total_batches} | Datei {global_idx}/{total_files})"

                raw_text = extract_text_from_file(file_path)
                if not raw_text.strip():
                    continue

                content_hash = calculate_sha256(raw_text)
                if (rel_path, content_hash) in existing_hashes:
                    log_msg(log_list, f"[{global_idx}/{total_files}] ⏭️ Unverändert übersprungen: {rel_path}")
                    continue

                ext = os.path.splitext(rel_path)[1].lower()
                parse_start_time = time.time()

                try:
                    # Determinismus nach Dateiendungen (Kein vorschnelles Infiltrations-Skipping)
                    if ext == ".py":
                        log_msg(log_list, f"[{global_idx}/{total_files}] Hybrid AST-LLM Parse (Python): {rel_path}")
                        used_llm_in_this_batch = True
                        category_tag, processed_md = hybrid_ast_llm_parse(rel_path, raw_text, active_model, ollama_worker, num_ctx, max_embed_chars)

                    elif ext == ".java":
                        log_msg(log_list, f"[{global_idx}/{total_files}] Hybrid AST-LLM Parse (Java/FreeRouting): {rel_path}")
                        used_llm_in_this_batch = True
                        category_tag, processed_md = hybrid_java_llm_parse(rel_path, raw_text, active_model, ollama_worker, num_ctx, max_embed_chars)

                    elif ext in {".kicad_mod", ".kicad_sym", ".kicad_pcb", ".kicad_sch", ".kicad_dru", ".rules", ".dsn", ".ses"}:
                        log_msg(log_list, f"[{global_idx}/{total_files}] Fast-Pass Parsing (EDA/Specctra): {rel_path}")
                        category_tag, processed_md = fast_parse_kicad(rel_path, raw_text, max_embed_chars)

                    else:
                        log_msg(log_list, f"[{global_idx}/{total_files}] Verarbeite via LLM ({active_model}): {rel_path}")
                        used_llm_in_this_batch = True
                        full_user_prompt = f"DATEIPFAD: {rel_path}\nINHALT:\n{raw_text[:12000]}"
                        response = ollama_worker.chat(
                            model=active_model,
                            messages=[
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': full_user_prompt}
                            ],
                            options={"num_ctx": num_ctx}
                        )
                        processed_md = response['message']['content']
                        tag_match = re.search(r'\[(?:TAG|PAGE|CATEGORY):\s*([A-Z0-9_]+)\]', processed_md, re.IGNORECASE)
                        category_tag = tag_match.group(1).upper() if tag_match else "GENERAL"

                    if not processed_md:
                        log_msg(log_list, f"   ⚠️ Übersprungen (Datei-Filter/Größenlimit): {rel_path}")
                        continue

                    parse_duration = time.time() - parse_start_time
                    batch_prepared_items.append({
                        "rel_path": rel_path,
                        "content_hash": content_hash,
                        "category_tag": category_tag,
                        "processed_md": processed_md,
                        "parse_duration": parse_duration,
                        "global_idx": global_idx
                    })

                except Exception as parse_err:
                    log_msg(log_list, f"   ❌ Analyse-Fehler bei {rel_path}: {str(parse_err)}")

            # --- VRAM RELEASE VOR EMBEDDING ---
            if used_llm_in_this_batch:
                log_msg(log_list, f"🔄 Phase A abgeschlossen. Entlade Anwendungs-LLM ({active_model}) aus dem VRAM...")
                unload_ollama_model(ollama_worker, active_model)
                time.sleep(1.5)

            if not batch_prepared_items:
                log_msg(log_list, f"ℹ️ Batch {current_batch_num} enthält keine neu zu indizierenden Dateien.\n")
                continue

            # --- PHASE B: VEKTORISIERUNG MIT STRUCTURAL OVERLAP CHUNKING ---
            log_msg(log_list, f"📐 Phase B [Batch {current_batch_num}]: Erzeuge Embeddings ({EMBED_MODEL}) & speichere in Qdrant...")
            for prep_item in batch_prepared_items:
                rel_path = prep_item["rel_path"]
                content_hash = prep_item["content_hash"]
                category_tag = prep_item["category_tag"]
                processed_md = prep_item["processed_md"]
                parse_dur = prep_item["parse_duration"]
                global_idx = prep_item["global_idx"]

                embed_start_time = time.time()

                # Strukturbewusstes Chunking mit Overlap
                md_chunks = smart_markdown_chunking(processed_md, max_chars=12000, overlap_chars=1200)

                for chunk_idx, md_chunk in enumerate(md_chunks):
                    chunk_success = False
                    retry_count = 0
                    current_chars = len(md_chunk)

                    while not chunk_success and retry_count < 3:
                        try:
                            chunk_ctx = calculate_dynamic_num_ctx(current_chars, max_limit=32768)

                            embed_res = ollama_worker.embeddings(
                                model=EMBED_MODEL, 
                                prompt=md_chunk[:current_chars],
                                options={"num_ctx": chunk_ctx}
                            )

                            vector = embed_res['embedding']
                            ensure_qdrant_collection(qdrant_worker, target_collection, len(vector))

                            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{target_collection}_{rel_path}_chunk_{chunk_idx}"))

                            qdrant_worker.upsert(
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
                                            "chunk_index": chunk_idx,
                                            "total_chunks": len(md_chunks),
                                            "content": md_chunk[:current_chars]
                                        }
                                    )
                                ]
                            )
                            chunk_success = True

                        except Exception as embed_err:
                            retry_count += 1
                            log_msg(log_list, f"   ⚠️ Vulkan-Reset bei {rel_path} (Chunk {chunk_idx+1}/{len(md_chunks)}): {embed_err}")
                            unload_ollama_model(ollama_worker, EMBED_MODEL)
                            time.sleep(2.5)
                            current_chars = int(current_chars * 0.85)

                embed_dur = time.time() - embed_start_time
                total_file_dur = parse_dur + embed_dur
                log_msg(
                    log_list, 
                    f"[{global_idx}/{total_files}] ✅ Indiziert in {total_file_dur:.2f}s "
                    f"({len(md_chunks)} Chunk(s) | Analyse: {parse_dur:.2f}s | Embed: {embed_dur:.2f}s) | **#{category_tag}**: {rel_path}"
                )

            # --- VRAM CLEANUP VOR NÄCHSTEM BATCH ---
            log_msg(log_list, f"🔄 Phase B abgeschlossen. Entlade Embedding-Modell ({EMBED_MODEL}) aus dem VRAM...")
            unload_ollama_model(ollama_worker, EMBED_MODEL)
            time.sleep(1.5)
            log_msg(log_list, f"✅ Batch {current_batch_num}/{total_batches} vollständig verankert!\n")

        log_msg(log_list, f"🎉 Ingestion erfolgreich beendet! Alle Dokumente sind in Collection '{target_collection}' verfügbar.")
        status_dict["header"] = f"✅ Status: ABGESCHLOSSEN ({total_files}/{total_files})"

    except Exception as top_e:
        log_msg(log_list, f"\n❌ Unerwarteter Systemfehler: {str(top_e)}")
        status_dict["header"] = "🔴 Status: FEHLER"
    finally:
        status_dict["running"] = False
        if temp_work_dir and os.path.exists(temp_work_dir):
            shutil.rmtree(temp_work_dir, ignore_errors=True)
        log_msg(log_list, "✨ System ist wieder inaktiv und bereit für neue Anfragen.")

# --- INGEST TASK PROCESS MANAGER ---
class IngestProcessManager:
    def __init__(self):
        self.process = None
        self.manager = multiprocessing.Manager()
        self.log_list = self.manager.list()
        log_msg(self.log_list, "Inaktiv. Bereit für neuen Ingestion-Job.")
        self.status_dict = self.manager.dict({"header": "⚪ Status: Inaktiv", "running": False})

    def append_log(self, text: str):
        log_msg(self.log_list, text)

    def set_status(self, header: str):
        self.status_dict["header"] = header

    def get_ui_snapshot(self):
        full_log = "\n".join(list(self.log_list))
        header = self.status_dict.get("header", "⚪ Status: Inaktiv")
        return full_log, f"### {header}"

    def is_alive(self):
        return self.process is not None and self.process.is_alive()

    def request_cancel(self):
        """Beendet den Ingest-Prozess hart per SIGTERM/SIGKILL."""
        if self.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.kill()
            
            self.status_dict["running"] = False
            self.status_dict["header"] = "🔴 Status: ABGEBROCHEN"
            log_msg(self.log_list, "\n🛑 Abbruch-Signal ausgeführt: Prozess wurde umgehend beendet.")
            log_msg(self.log_list, "✨ System ist wieder inaktiv und bereit für neue Anfragen.")
        else:
            self.status_dict["header"] = "⚪ Status: Inaktiv"
            self.status_dict["running"] = False
            log_msg(self.log_list, "ℹ️ Kein aktiver Job zum Abbrechen vorhanden.")

        return "\n".join(list(self.log_list)), f"### {self.status_dict['header']}"

    def start_background_job(self, files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model, mode="standard", mining_repo_url="", num_ctx=32768, max_embed_chars=50000, batch_size=25):
        if self.is_alive():
            log_msg(self.log_list, "⚠️ Ein Ingestion-Job läuft derzeit noch. Bitte erst abbrechen...")
            return f"### {self.status_dict['header']}"

        del self.log_list[:]
        self.status_dict["header"] = "🟢 Status: WIRD GESTARTET..."
        self.status_dict["running"] = True
        
        save_config({
            "last_model": selected_model, 
            "last_category": category_key,
            "last_num_ctx": num_ctx,
            "last_max_embed_chars": max_embed_chars,
            "last_batch_size": batch_size
        })

        self.process = multiprocessing.Process(
            target=worker_process_entry,
            args=(
                self.log_list, self.status_dict, files, scanned_repo_path, 
                selected_folders, selected_exts, category_key, selected_model, 
                mode, mining_repo_url, num_ctx, max_embed_chars, batch_size
            ),
            daemon=True
        )
        self.process.start()
        return f"### {self.status_dict['header']}"

task_manager = IngestProcessManager()

# --- HILFSFUNKTIONEN UI & GIT ---
def get_ollama_models():
    config = load_config()
    saved_model = config.get("last_model", DEFAULT_MODEL)

    try:
        ollama_client = ollama.Client(host=OLLAMA_HOST)
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

    log_msg_text = f"✅ Repository gescannt! {len(folder_choices)-1} Ordner und {len(ext_choices)} Dateiformate erkannt."
    task_manager.append_log(log_msg_text)
    task_manager.set_status("⚪ Status: Inaktiv (Scan bereit)")

    yield (
        gr.update(choices=folder_choices, value=["ALL_REPO"], visible=True),
        gr.update(choices=ext_choices, value=[e[1] for e in ext_choices], visible=True),
        repo_dir,
        log_msg_text
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
initial_num_ctx = saved_cfg.get("last_num_ctx", 32768)
initial_max_embed_chars = saved_cfg.get("last_max_embed_chars", 50000)
initial_batch_size = saved_cfg.get("last_batch_size", 25)

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

            with gr.Accordion("⚙️ Kontext- & Performance-Einstellungen", open=True):
                batch_size_slider = gr.Slider(
                    minimum=5, 
                    maximum=200, 
                    step=5, 
                    value=initial_batch_size, 
                    label="Batch-Größe (Dateien pro Wechsel)",
                    info="Legt fest, nach wie vielen analysierten Dateien VRAM entladen und Embeddings in Qdrant gespeichert werden."
                )
                num_ctx_slider = gr.Slider(
                    minimum=4096, 
                    maximum=32768, 
                    step=2048, 
                    value=initial_num_ctx, 
                    label="Ollama Kontextfenster (num_ctx Tokens)",
                    info="Erhöht die max. Verarbeitungsmenge für LLM & Embedding (32.768 max für Qwen3)."
                )
                embed_chars_slider = gr.Slider(
                    minimum=4000, 
                    maximum=80000, 
                    step=2000, 
                    value=initial_max_embed_chars, 
                    label="Max. Embedding Input Zeichen (Chars)",
                    info="Regelt den Schnitt-Punkt vor der Einreichung beim Vektormodell."
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
                        value="[https://github.com/freerouting/freerouting](https://github.com/freerouting/freerouting)",
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
        fn=lambda files, repo, f_cb, e_cb, cat, mod, ctx, chars, batch: task_manager.start_background_job(
            files, repo, f_cb, e_cb, cat, mod, mode="standard", num_ctx=ctx, max_embed_chars=chars, batch_size=batch
        ),
        inputs=[file_input, repo_state, folder_checkboxes, ext_checkboxes, category_dropdown, model_dropdown, num_ctx_slider, embed_chars_slider, batch_size_slider],
        outputs=[status_banner]
    )

    start_mining_btn.click(
        fn=lambda cat, mod, url, ctx, chars, batch: task_manager.start_background_job(
            None, None, None, None, cat, mod, mode="repo_mining", mining_repo_url=url, num_ctx=ctx, max_embed_chars=chars, batch_size=batch
        ),
        inputs=[category_dropdown, model_dropdown, mining_repo_input, num_ctx_slider, embed_chars_slider, batch_size_slider],
        outputs=[status_banner]
    )

    stop_btn.click(
        fn=task_manager.request_cancel,
        outputs=[status_output, status_banner]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7861,
        css=custom_css,
        js=autoscroll_js
    )
