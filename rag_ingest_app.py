import os
import re
import uuid
import time
import shutil
import zipfile
import hashlib
import json
import subprocess
import multiprocessing
import gradio as gr
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader

# Dynamisches Plugin-System importieren
from processors.processor_registry import registry

# --- KONFIGURATION & KONSTANTEN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q5_K_M")
CONFIG_FILE = "/tmp/rag_ingest_config.json"
REPO_FILTERS_FILE = os.path.join(BASE_DIR, "repo_filters.json")
OSHW_SOURCES_FILE = os.path.join(BASE_DIR, "oshw_sources.json")

CATEGORIES = registry.get_categories_dict()
TEXT_EXTENSIONS = registry.get_all_supported_extensions()

# --- DEFAULT OSHW BEZUGSQUELLEN (FALLBACK) ---
DEFAULT_OSHW_SOURCES = [
    {"name": "🌐 Olimex ESP32-GATEWAY (Industrial IoT, Ethernet, Power)", "url": "https://github.com/OLIMEX/ESP32-GATEWAY"},
    {"name": "🔌 Olimex ESP32-EVB (Ethernet, Relays, CAN Bus, Power)", "url": "https://github.com/OLIMEX/ESP32-EVB"},
    {"name": "⚡ Olimex ESP32-PoE (Power-over-Ethernet, LiPo Charge)", "url": "https://github.com/OLIMEX/ESP32-PoE"},
    {"name": "⚡ Adafruit Feather M4 Express (Power, MCU, USB, LiPo)", "url": "https://github.com/adafruit/Adafruit-Feather-M4-Express-PCB"},
    {"name": "📡 Adafruit ESP32-S3 Feather (LiPo Charging, USB-C, Power)", "url": "https://github.com/adafruit/Adafruit-ESP32-S3-Feather-PCB"},
    {"name": "🧩 SparkFun MicroMod MainBoard (Modular Interfaces)", "url": "https://github.com/sparkfun/SparkFun_MicroMod_MainBoard_Single_Hardware"},
    {"name": "🔋 SparkFun RedBoard Qwiic (USB-C, Power & Logic Shifting)", "url": "https://github.com/sparkfun/SparkFun_RedBoard_Qwiic_Hardware"},
    {"name": "📶 SparkFun Thing Plus ESP32 WROOM (LiPo, USB-C, I2C)", "url": "https://github.com/sparkfun/SparkFun_Thing_Plus_ESP32_WROOM_Hardware"},
    {"name": "📡 Seeed Studio KiCad Library (Sensor & Breakout Schematics)", "url": "https://github.com/Seeed-Studio/Seeed_KiCad_Lib"},
    {"name": "⚙️ Arduino AVR Reference Boards (ATmega, Power, Serial)", "url": "https://github.com/arduino/ArduinoCore-avr"},
    {"name": "🍇 Raspberry Pi Official HAT Specifications & Reference", "url": "https://github.com/raspberrypi/hats"},
    {"name": "🔧 Pine64 Pinecil (USB-PD Power Electronics & DC/DC)", "url": "https://github.com/pine64/Pinecil"},
    {"name": "📡 Great Scott Gadgets HackRF One (RF & Analog Reference)", "url": "https://github.com/greatscottgadgets/hackrf"},
    {"name": "🦘 PocketBeagle (High-Density System & Power Reference)", "url": "https://github.com/beagleboard/pocketbeagle"},
    {"name": "🐍 SKiDL Circuits As Code (Offizielle SKiDL Subcircuit-Bibliothek)", "url": "https://github.com/devbisme/circuitsascode"},
    {"name": "🏗️ Atopile Language & Hardware Library (Deklarative PCB-Module)", "url": "https://github.com/atopile/atopile"},
    {"name": "🤖 KiCad Tools für LLM Agenten (Pure Python Parser, DRC & Router)", "url": "https://github.com/rjwalters/kicad-tools"},
    {"name": "🐍 PyKiCad (Python-Bibliothek für KiCad PCB-Generierung)", "url": "https://github.com/dvc94ch/pykicad"},
    {"name": "🔌 KiCad Python IPC API Bindings", "url": "https://github.com/atopile/kicad-python"},
    {"name": "🔥 Procedural PCB Geometry Generator (KiCad Trace Generierung)", "url": "https://github.com/Trilys/PCB_Heater_KiCad"},
    {"name": "⚡ OrthoRoute (GPU-Beschleunigter KiCad Autorouter)", "url": "https://github.com/bbenchoff/OrthoRoute"},
    {"name": "🤖 KiBot Automated KiCad Preflights & DRC Verification", "url": "https://github.com/MicroType-Engineering/KiBot"}
]

# --- PERSISTENT OSHW SOURCES MANAGEMENT ---
def load_oshw_sources() -> list[dict]:
    """Lädt die OSHW-Quellen aus der JSON-Datei. Bei Lesefehlern oder leerer Datei greift das DEFAULT-Fallback."""
    if os.path.exists(OSHW_SOURCES_FILE):
        if not os.path.isfile(OSHW_SOURCES_FILE):
            print(f"[ERROR] {OSHW_SOURCES_FILE} existiert, ist aber ein ORDNER anstelle einer Datei!")
            return DEFAULT_OSHW_SOURCES

        try:
            with open(OSHW_SOURCES_FILE, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, list) and len(data) > 0:
                        print(f"[SUCCESS] {len(data)} OSHW-Quellen erfolgreich geladen.")
                        return data
                    else:
                        print(f"[WARN] {OSHW_SOURCES_FILE} enthält kein gültiges JSON-Array oder ist leer.")
        except Exception as e:
            print(f"[ERROR] Fehler beim Lesen von {OSHW_SOURCES_FILE}: {e}")
    else:
        print(f"[WARN] Datei nicht gefunden unter: {OSHW_SOURCES_FILE}")

    print("[INFO] Nutze DEFAULT_OSHW_SOURCES als Fallback.")
    return DEFAULT_OSHW_SOURCES

def save_oshw_sources(sources_list: list[dict]):
    try:
        os.makedirs(os.path.dirname(OSHW_SOURCES_FILE), exist_ok=True)
        with open(OSHW_SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump(sources_list, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] {len(sources_list)} Quellen in {OSHW_SOURCES_FILE} gespeichert.")
    except Exception as e:
        print(f"[ERROR] Fehler beim Speichern der OSHW-Quellen: {e}")

def get_oshw_dropdown_choices() -> list[tuple[str, str]]:
    sources = load_oshw_sources()
    return [(s.get("name", s.get("url")), s.get("url")) for s in sources if s.get("url")]

def oshw_sources_to_text(sources_list: list[dict]) -> str:
    lines = []
    for s in sources_list:
        lines.append(f"{s.get('name', '')} | {s.get('url', '')}")
    return "\n".join(lines)

def text_to_oshw_sources(text: str) -> list[dict]:
    sources = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            parts = line.split("|", 1)
            name = parts[0].strip()
            url = parts[1].strip()
        else:
            url = line.strip()
            name = url.split("/")[-1]
        if url:
            sources.append({"name": name, "url": url})
    return sources

def handle_save_oshw_editor(raw_text: str):
    new_sources = text_to_oshw_sources(raw_text)
    save_oshw_sources(new_sources)
    choices = get_oshw_dropdown_choices()
    default_vals = [c[1] for c in choices[:2]] if len(choices) >= 2 else ([c[1] for c in choices] if choices else [])
    return (
        gr.update(choices=choices, value=default_vals),
        f"✅ {len(new_sources)} Quellen erfolgreich gespeichert und Menü aktualisiert!"
    )

# --- LOGGING HELPER MIT LOKALER ZEITZEILE ---
def log_msg(log_list, text: str):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    log_list.append(f"[{timestamp}] {text}")

# --- HELPER: NON-INTERACTIVE GIT ENVIRONMENT ---
def get_git_env():
    """Erstellt eine Prozessumgebung, die Git-Prompts (Username/Passwort) strikt unterbindet."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "echo"
    return env

# --- FILTER PRESETS MANAGEMENT (ERWEITERT UM BUILD-ARTEFAKTE UND EDA-RAW-FILTER) ---
def get_default_filter_presets() -> dict:
    return {
        "default": [
            "/fixtures/", "/tests/", "/test/", "/benchmarks/", "/.github/", "/.agents/",
            "/build/", "/CMakeFiles/", "TargetDirectories.txt", "CMakeCache.txt",
            ".kicad_pcb", ".pro", ".kicad_sch", ".net"
        ],
        "freerouting": [
            "/api/security/", "/api/dev/", "/analytics/", "/api/mcp/", "/util/gson/",
            "Analytics", "RateLimit", "ApiKey", "ExceptionMapper", "MessageBody", "WebSocketConfigurator", "Mocked",
            "/build/", "/CMakeFiles/", "TargetDirectories.txt", "CMakeCache.txt",
            ".kicad_pcb", ".pro", ".kicad_sch", ".net"
        ],
        "adafruit": [
            "/.github/", "/build/", "README.md", ".kicad_pcb", ".pro", ".kicad_sch", ".net"
        ],
        "sparkfun": [
            "/.github/", "/Firmware/", "/Software/", ".kicad_pcb", ".pro", ".kicad_sch", ".net"
        ]
    }

def get_repo_filters_dict() -> dict:
    if os.path.exists(REPO_FILTERS_FILE):
        try:
            with open(REPO_FILTERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return get_default_filter_presets()

def get_filter_preset_for_url(repo_url_or_path) -> str:
    if isinstance(repo_url_or_path, list):
        repo_url_or_path = repo_url_or_path[0] if repo_url_or_path else ""

    if not repo_url_or_path:
        return "\n".join(get_default_filter_presets()["default"])
        
    filters_dict = get_repo_filters_dict()
    url_lower = str(repo_url_or_path).lower()
    
    for key, patterns in filters_dict.items():
        if key != "default" and key in url_lower:
            return "\n".join(patterns)
            
    return "\n".join(filters_dict.get("default", []))

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
    try:
        ollama_client.chat(model=model_name, messages=[], keep_alive=0)
    except Exception:
        pass

def calculate_dynamic_num_ctx(text_len: int, max_limit: int = 32768) -> int:
    estimated_tokens = int(text_len / 3.0) + 512
    num_ctx = 2048
    while num_ctx < estimated_tokens and num_ctx < max_limit:
        num_ctx *= 2
    return min(num_ctx, max_limit)

def smart_markdown_chunking(text: str, max_chars: int = 4000, overlap_chars: int = 400) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    blocks = re.split(r'(\n(?=#{1,4}\s)|\n(?=---\n)|\n\n+)', text)
    chunks = []
    current_chunk = ""

    for b in blocks:
        if not b:
            continue
        if len(current_chunk) + len(b) <= max_chars:
            current_chunk += b
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            if len(b) > max_chars:
                lines = b.splitlines(keepends=True)
                sub_chunk = ""
                for line in lines:
                    if len(sub_chunk) + len(line) <= max_chars:
                        sub_chunk += line
                    else:
                        if sub_chunk.strip():
                            chunks.append(sub_chunk.strip())
                        sub_chunk = line
                current_chunk = sub_chunk
            else:
                current_chunk = b

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    sanitized_chunks = []
    for chunk in chunks:
        backtick_matches = re.findall(r'```', chunk)
        if len(backtick_matches) % 2 != 0:
            chunk += "\n```"
        sanitized_chunks.append(chunk)

    return sanitized_chunks

# --- PROZESS WORKER MIT NON-INTERACTIVE GIT ENVIRONMENT ---
def worker_process_entry(log_list, status_dict, files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model, mode, mining_repo_url, num_ctx, max_embed_chars, batch_size, custom_filters_raw=""):
    temp_work_dir = None
    try:
        ollama_worker = ollama.Client(host=OLLAMA_HOST)
        qdrant_worker = QdrantClient(url=QDRANT_HOST)

        category_info = CATEGORIES.get(category_key, CATEGORIES[list(CATEGORIES.keys())[0]])
        target_collection = category_info["collection"]
        active_model = selected_model if selected_model else DEFAULT_MODEL

        active_custom_filters = [
            line.strip() for line in (custom_filters_raw or "").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        llm_options = {
            "num_ctx": num_ctx,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.05
        }

        session_id = str(uuid.uuid4())[:8]
        temp_work_dir = os.path.join("/tmp", f"rag_ingest_{session_id}")
        os.makedirs(temp_work_dir, exist_ok=True)

        git_env = get_git_env()
        files_to_process = []

        if mode in ["repo_mining", "oshw_mining"]:
            repo_urls = mining_repo_url if isinstance(mining_repo_url, list) else [mining_repo_url]
            clean_repo_urls = [sanitize_url(u) for u in repo_urls if sanitize_url(u)]

            if not clean_repo_urls:
                log_msg(log_list, "❌ Keine validen Repository-URLs für das Mining angegeben.")
                status_dict["header"] = "🔴 Status: BEENDET (Ungültige URL)"
                return

            log_msg(log_list, f"⛏️ Starte Batch Git-Mining [{mode.upper()}] für {len(clean_repo_urls)} Repository/Repositories...")

            for repo_idx, clean_repo_url in enumerate(clean_repo_urls, 1):
                log_msg(log_list, f"   [Repo {repo_idx}/{len(clean_repo_urls)}] Klone: {clean_repo_url}")
                mined_repo_dir = os.path.join(temp_work_dir, f"mined_repo_{repo_idx}")
                
                res = subprocess.run(
                    ["git", "clone", "--depth", "1", clean_repo_url, mined_repo_dir],
                    capture_output=True, text=True, env=git_env
                )

                if res.returncode != 0:
                    err_brief = res.stderr.strip()[:200] if res.stderr else "Unbekannter Fehler"
                    log_msg(log_list, f"   ⚠️ Git-Clone fehlgeschlagen für {clean_repo_url}: {err_brief}")
                    continue

                repo_base_name = os.path.basename(clean_repo_url.rstrip("/"))
                file_count = 0
                for root, _, filenames in os.walk(mined_repo_dir):
                    if ".git" in root:
                        continue
                    for f in filenames:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in TEXT_EXTENSIONS:
                            full_p = os.path.join(root, f)
                            rel_p = os.path.relpath(full_p, mined_repo_dir)
                            files_to_process.append((f"{repo_base_name}/{rel_p}", full_p))
                            file_count += 1
                
                log_msg(log_list, f"   ↳ Repo '{repo_base_name}' geklont ({file_count} relevante Dateien).")

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
            log_msg(log_list, "❌ Keine verwertbaren Dateien in allen ausgewählten Quellen gefunden.")
            status_dict["header"] = "🔴 Status: BEENDET (Keine Dateien)"
            return

        total_files = len(files_to_process)
        log_msg(log_list, f"📊 Gesamt: {total_files} Datei(en) bereit zur Prüfung.")
        log_msg(log_list, f"⚙️ Konfiguration: Target Batch Size = {batch_size} verarbeitete Dateien | Max Embed Chars = {max_embed_chars}")
        log_msg(log_list, f"🛡️ Aktive Webpanel-Filter: {len(active_custom_filters)} Regel(n) geladen.")
        
        existing_hashes = get_indexed_hashes_set(qdrant_worker, target_collection)
        log_msg(log_list, f"   ↳ {len(existing_hashes)} bereits indizierte Datei(en) in '{target_collection}' übersprungen.\n")

        batch_prepared_items = []
        batch_count = 0
        used_llm_in_current_batch = False

        for global_idx, (rel_path, file_path) in enumerate(files_to_process, 1):
            current_buffer_count = len(batch_prepared_items)
            status_dict["header"] = f"🟢 Status: LÄUFT (Datei {global_idx}/{total_files} | Im Batch-Puffer: {current_buffer_count}/{batch_size})"

            raw_text = extract_text_from_file(file_path)
            if not raw_text.strip():
                continue

            content_hash = calculate_sha256(raw_text)
            if (rel_path, content_hash) in existing_hashes:
                log_msg(log_list, f"[{global_idx}/{total_files}] ⏭️ Unverändert übersprungen: {rel_path}")
                continue

            log_msg(log_list, f"[{global_idx}/{total_files}] ⏳ Starte Phase A Analyse [Batch-Puffer: {current_buffer_count}/{batch_size}]: {rel_path}")
            parse_start_time = time.time()

            try:
                category_tag, processed_md = registry.dispatch_parse(
                    rel_path, raw_text, active_model, ollama_worker, llm_options, max_embed_chars, 
                    selected_category=category_key, custom_filters=active_custom_filters
                )

                if processed_md == "SKIP" or not processed_md:
                    log_msg(log_list, f"[{global_idx}/{total_files}] ⏭️ Übersprungen (Gefiltert/Build-Doc): {rel_path}")
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
                used_llm_in_current_batch = True

                new_buffer_count = len(batch_prepared_items)
                log_msg(
                    log_list, 
                    f"[{global_idx}/{total_files}] ✅ Phase A Analyse abgeschlossen in {parse_duration:.2f}s "
                    f"[Batch-Puffer: {new_buffer_count}/{batch_size}] | **#{category_tag}**: {rel_path}"
                )

            except Exception as parse_err:
                log_msg(log_list, f"   ❌ Analyse-Fehler bei {rel_path}: {str(parse_err)}")

            is_last_file = (global_idx == total_files)
            if len(batch_prepared_items) >= batch_size or (is_last_file and batch_prepared_items):
                batch_count += 1
                log_msg(log_list, f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                log_msg(log_list, f"📦 STARTE VEKTORISIERUNG FÜR BATCH {batch_count} ({len(batch_prepared_items)} Dateien)")
                log_msg(log_list, f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                if used_llm_in_current_batch:
                    log_msg(log_list, f"🔄 Phase A abgeschlossen. Entlade Anwendungs-LLM ({active_model}) aus VRAM...")
                    unload_ollama_model(ollama_worker, active_model)
                    time.sleep(1.5)

                log_msg(log_list, f"📐 Phase B [Batch {batch_count}]: Erzeuge Embeddings ({EMBED_MODEL}) & speichere in Qdrant...")
                overlap_val = max(200, int(max_embed_chars * 0.10))

                for prep_item in batch_prepared_items:
                    r_path = prep_item["rel_path"]
                    c_hash = prep_item["content_hash"]
                    c_tag = prep_item["category_tag"]
                    p_md = prep_item["processed_md"]
                    p_dur = prep_item["parse_duration"]
                    g_idx = prep_item["global_idx"]

                    log_msg(log_list, f"[{g_idx}/{total_files}] ⏳ Starte Phase B Vektorisierung: {r_path}")
                    embed_start_time = time.time()
                    raw_md_chunks = smart_markdown_chunking(p_md, max_chars=max_embed_chars, overlap_chars=overlap_val)

                    # --- QUALITÄTSFILTER FÜR PCB-GENERATOR (MIN-LÄNGE & NOISE-FILTER) ---
                    md_chunks = []
                    for chk in raw_md_chunks:
                        cleaned_chk = chk.strip()
                        if len(cleaned_chk) < 60:
                            continue
                        if cleaned_chk.startswith("(") and cleaned_chk.endswith(")") and len(cleaned_chk.split()) <= 2:
                            continue
                        md_chunks.append(chk)

                    if not md_chunks:
                        log_msg(log_list, f"[{g_idx}/{total_files}] ⏭️ Keine qualifizierten Chunks nach Längen- & Müllfilterung: {r_path}")
                        continue

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

                                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{target_collection}_{r_path}_chunk_{chunk_idx}"))

                                # --- OPTIMIERTER PAYLOAD OHNE SPEICHERVERSCHWENDUNG (NUR 'content', KEINE REDUNDANTEN REPLIKATE) ---
                                qdrant_worker.upsert(
                                    collection_name=target_collection,
                                    points=[
                                        PointStruct(
                                            id=point_id,
                                            vector=vector,
                                            payload={
                                                "title": os.path.basename(r_path),
                                                "filename": os.path.basename(r_path),
                                                "file_path": r_path,
                                                "content_hash": c_hash,
                                                "category_tag": c_tag,
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
                                log_msg(log_list, f"   ⚠️ Vulkan-Reset bei {r_path} (Chunk {chunk_idx+1}/{len(md_chunks)}): {embed_err}")
                                unload_ollama_model(ollama_worker, EMBED_MODEL)
                                time.sleep(2.5)
                                current_chars = int(current_chars * 0.80)

                    embed_dur = time.time() - embed_start_time
                    total_file_dur = p_dur + embed_dur
                    log_msg(
                        log_list, 
                        f"[{g_idx}/{total_files}] ✅ Phase B Vektorisierung abgeschlossen in {embed_dur:.2f}s "
                        f"({len(md_chunks)} Chunk(s) | Gesamtzeit: {total_file_dur:.2f}s) | **#{c_tag}**: {r_path}"
                    )

                log_msg(log_list, f"🔄 Phase B abgeschlossen. Entlade Embedding-Modell ({EMBED_MODEL}) aus VRAM...\n")
                unload_ollama_model(ollama_worker, EMBED_MODEL)
                time.sleep(1.5)

                batch_prepared_items.clear()
                used_llm_in_current_batch = False

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

    def start_background_job(self, files, scanned_repo_path, selected_folders, selected_exts, category_key, selected_model, mode="standard", mining_repo_url="", num_ctx=32768, max_embed_chars=50000, batch_size=25, custom_filters_raw=""):
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
                mode, mining_repo_url, num_ctx, max_embed_chars, batch_size, custom_filters_raw
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

    git_env = get_git_env()

    res = subprocess.run(
        ["git", "clone", "--depth", "1", clean_url, repo_dir],
        capture_output=True, text=True, env=git_env
    )

    if res.returncode != 0:
        err_brief = res.stderr.strip()[:200] if res.stderr else "Unbekannter Fehler"
        task_manager.set_status("🔴 Status: SCAN FEHLGESCHLAGEN")
        task_manager.append_log(f"❌ Git-Clone fehlgeschlagen: {err_brief}")
        yield (
            gr.update(choices=[], value=[], visible=False),
            gr.update(choices=[], value=[], visible=False),
            "",
            f"❌ Git-Clone fehlgeschlagen:\n{err_brief}"
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
initial_max_embed_chars = saved_cfg.get("last_max_embed_chars", 4000)
initial_batch_size = saved_cfg.get("last_batch_size", 25)

initial_oshw_choices = get_oshw_dropdown_choices()
initial_oshw_defaults = [c[1] for c in initial_oshw_choices[:2]] if len(initial_oshw_choices) >= 2 else ([c[1] for c in initial_oshw_choices] if initial_oshw_choices else [])

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
                label="Knowledge Collection / Plugin Category",
                interactive=True
            )

            with gr.Accordion("⚙️ Kontext- & Performance-Einstellungen", open=True):
                batch_size_slider = gr.Slider(
                    minimum=5, 
                    maximum=200, 
                    step=5, 
                    value=initial_batch_size, 
                    label="Batch-Größe (Gültig aufbereitete Dateien pro Wechsel)",
                    info="Legt fest, nach wie vielen echten, verarbeiteten Dateien VRAM entladen und Embeddings in Qdrant gespeichert werden."
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
                    minimum=2000, 
                    maximum=16000, 
                    step=500, 
                    value=initial_max_embed_chars, 
                    label="Max. Embedding Chunk-Größe (Zeichen)",
                    info="Regelt die Chunk-Größe vor der Einreichung beim Vektormodell (4.000 Chars ideal für Vulkan)."
                )

            with gr.Accordion("🛡️ Dynamic Repository Filter-Rules (Live Edit)", open=False):
                custom_filters_input = gr.Textbox(
                    label="Ausschlussmuster & Keywords (Ein Muster pro Zeile)",
                    placeholder="/api/security/\nRateLimit\n/fixtures/",
                    lines=6,
                    value=get_filter_preset_for_url("freerouting"),
                    info="Dateien, deren Pfad oder Name ein solches Muster enthält, werden im Ingest sofort übersprungen."
                )

            with gr.Tabs():
                with gr.Tab("🔌 OSHW Library Mining"):
                    gr.Markdown("### 🏛️ Vorkonfigurierte Open-Source Hardware Repositories")
                    oshw_preset_dropdown = gr.Dropdown(
                        choices=initial_oshw_choices,
                        value=initial_oshw_defaults,
                        label="OSHW-Bezugsquellen auswählen (Mehrfachauswahl möglich)",
                        multiselect=True,
                        interactive=True
                    )
                    start_oshw_btn = gr.Button("🔌 OSHW Sub-Circuits Ingestieren (Batch)", variant="primary")

                    with gr.Accordion("📝 OSHW-Quellen verwalten & speichern (Persistent)", open=False):
                        gr.Markdown("Format pro Zeile: `Anzeigename | Repository-URL` oder nur `Repository-URL`")
                        oshw_editor_input = gr.Textbox(
                            label="OSHW Quellen-Liste (Bearbeitbar)",
                            value=oshw_sources_to_text(load_oshw_sources()),
                            lines=10,
                            interactive=True
                        )
                        save_oshw_sources_btn = gr.Button("💾 Quellen-Liste speichern", variant="secondary")
                        oshw_save_status = gr.Markdown("")

                with gr.Tab("⭐ EDA & Rule Mining"):
                    mining_repo_input = gr.Dropdown(
                        choices=[
                            ("SKiDL Haupt-Repository (Offiziell)", "https://github.com/xesscorp/skidl"),
                            ("KiCad Python Action Plugins", "https://github.com/KiCad/kicad-python"),
                            ("Freerouting Java / Config Core", "https://github.com/freerouting/freerouting")
                        ],
                        value="https://github.com/freerouting/freerouting",
                        label="Ziel-Repository für EDA Mining",
                        allow_custom_value=True,
                        interactive=True
                    )
                    start_mining_btn = gr.Button("⛏️ Mining & Hybrid Ingestion Starten", variant="primary")

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

                with gr.Tab("📁 Dateiupload / ZIP"):
                    file_input = gr.File(
                        label="Dateien oder ZIP-Archiv hochladen", 
                        file_count="multiple"
                    )

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

    # Persistent OSHW Editor Callbacks
    save_oshw_sources_btn.click(
        fn=handle_save_oshw_editor,
        inputs=[oshw_editor_input],
        outputs=[oshw_preset_dropdown, oshw_save_status]
    )

    # Dynamic Filter Presets beim Wechsel von Repositories laden
    mining_repo_input.change(fn=get_filter_preset_for_url, inputs=[mining_repo_input], outputs=[custom_filters_input])
    oshw_preset_dropdown.change(fn=get_filter_preset_for_url, inputs=[oshw_preset_dropdown], outputs=[custom_filters_input])
    github_input.change(fn=get_filter_preset_for_url, inputs=[github_input], outputs=[custom_filters_input])

    scan_repo_btn.click(
        fn=scan_github_repository,
        inputs=[github_input, repo_state],
        outputs=[folder_checkboxes, ext_checkboxes, repo_state, status_output]
    )

    folder_checkboxes.change(fn=handle_folder_selection, inputs=[folder_checkboxes], outputs=[folder_checkboxes])

    start_btn.click(
        fn=lambda files, repo, f_cb, e_cb, cat, mod, ctx, chars, batch, filters: task_manager.start_background_job(
            files, repo, f_cb, e_cb, cat, mod, mode="standard", num_ctx=ctx, max_embed_chars=chars, batch_size=batch, custom_filters_raw=filters
        ),
        inputs=[file_input, repo_state, folder_checkboxes, ext_checkboxes, category_dropdown, model_dropdown, num_ctx_slider, embed_chars_slider, batch_size_slider, custom_filters_input],
        outputs=[status_banner]
    )

    start_mining_btn.click(
        fn=lambda cat, mod, url, ctx, chars, batch, filters: task_manager.start_background_job(
            None, None, None, None, cat, mod, mode="repo_mining", mining_repo_url=url, num_ctx=ctx, max_embed_chars=chars, batch_size=batch, custom_filters_raw=filters
        ),
        inputs=[category_dropdown, model_dropdown, mining_repo_input, num_ctx_slider, embed_chars_slider, batch_size_slider, custom_filters_input],
        outputs=[status_banner]
    )

    start_oshw_btn.click(
        fn=lambda cat, mod, urls, ctx, chars, batch, filters: task_manager.start_background_job(
            None, None, None, None, "⚡ PCB & Hardware Design", mod, mode="oshw_mining", mining_repo_url=urls, num_ctx=ctx, max_embed_chars=chars, batch_size=batch, custom_filters_raw=filters
        ),
        inputs=[category_dropdown, model_dropdown, oshw_preset_dropdown, num_ctx_slider, embed_chars_slider, batch_size_slider, custom_filters_input],
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
