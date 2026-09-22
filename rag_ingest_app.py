import os
import json
import streamlit as st
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Importieren der überarbeiteten Prozessoren
from processors.processor_registry import ProcessorRegistry


# ==========================================
# SEITENKONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="RAG Ingest Pipeline - PCB Generator",
    page_icon="⚡",
    layout="wide"
)

FILTER_FILE_PATH = "repo_filters.json"


# ==========================================
# HELFERFUNKTIONEN FÜR FILTER & FILE-TRAVERSAL
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
        st.error(f"Fehler beim Laden von {filters_path}: {e}")
        return []


def save_repo_filters(filters: List[str], filters_path: str = FILTER_FILE_PATH) -> bool:
    """Speichert die Blacklist-Muster in die JSON-Konfigurationsdatei."""
    try:
        with open(filters_path, "w", encoding="utf-8") as f:
            json.dump({"ignore_patterns": filters}, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Fehler beim Speichern von {filters_path}: {e}")
        return False


def is_blacklisted(file_path: str, ignore_patterns: List[str]) -> bool:
    """Prüft, ob ein Dateipfad ein Blacklist-Muster aus repo_filters.json enthält."""
    normalized_path = file_path.replace("\\", "/")
    for pattern in ignore_patterns:
        if pattern in normalized_path:
            return True
    return False


# ==========================================
# STREAMLIT UI & APPLIKATIONS-LOGIK
# ==========================================
def main():
    st.title("⚡ RAG Ingest UI - Autonomous PCB Generator Pipeline")
    st.markdown(
        "Diese Pipeline verarbeitet Hardware-Repositories, filtert physikalischen Layout-Müll "
        "und generiert schlanke, hochwertige Payloads für Qdrant."
    )

    # Sidebar: Einstellungen
    st.sidebar.header("⚙️ Qdrant & Ingestion Konfiguration")
    qdrant_url = st.sidebar.text_input("Qdrant URL", value="http://localhost:6333")
    collection_name = st.sidebar.text_input("Collection Name", value="hardware_docs")
    vector_size = st.sidebar.number_input("Vektor-Dimension", value=384, step=1)
    
    st.sidebar.subheader("Chunking Parameters")
    chunk_size = st.sidebar.number_input("Chunk Size (Chars)", value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap (Chars)", value=100, step=10)
    min_chunk_len = st.sidebar.number_input("Min Chunk Length (Chars)", value=60, step=10)

    # Tabs für Hauptfunktionen
    tab_ingest, tab_filters, tab_preview = st.tabs(["🚀 Ingestion Run", "🛡️ Filterliste (repo_filters.json)", "🔍 Chunk Vorschau"])

    # --------------------------------------------------
    # TAB 1: INGESTION RUN
    # --------------------------------------------------
    with tab_ingest:
        target_dir = st.text_input("Ziel-Verzeichnis zum Scannen", value="./src")
        
        col1, col2 = st.columns(2)
        with col1:
            dry_run = st.checkbox("Dry Run (Nur Analyse, kein Qdrant-Upload)", value=False)
        
        if st.button("🚀 Ingestion Pipeline Starten", type="primary"):
            if not os.path.exists(target_dir):
                st.error(f"Das Verzeichnis `{target_dir}` wurde nicht gefunden.")
                return

            ignore_patterns = load_repo_filters()
            registry = ProcessorRegistry()
            
            # Prozessoren mit Benutzer-Parametern konfigurieren
            registry.default_processor.min_chunk_len = min_chunk_len
            registry.default_processor.chunk_size = chunk_size
            registry.default_processor.chunk_overlap = chunk_overlap

            st.info("Scanne Dateien und wende Filterregeln an...")
            
            all_payloads = []
            file_stats = {"total_files": 0, "processed_files": 0, "skipped_files": 0}
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Alle relevanten Dateipfade sammeln
            file_paths = []
            for root, _, files in os.walk(target_dir):
                for f in files:
                    file_paths.append(os.path.join(root, f))
            
            file_stats["total_files"] = len(file_paths)

            for idx, full_path in enumerate(file_paths):
                # UI Status-Update
                progress_bar.progress((idx + 1) / max(1, len(file_paths)))
                status_text.text(f"Verarbeite ({idx+1}/{len(file_paths)}): {os.path.basename(full_path)}")

                # 1. Blacklist Check
                if is_blacklisted(full_path, ignore_patterns):
                    file_stats["skipped_files"] += 1
                    continue

                # 2. Prozessor zuweisen
                processor = registry.get_processor(full_path)

                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # 3. Datei verarbeiten (Filtert auch Chunks < min_chunk_len)
                    payloads = processor.process(content, full_path)
                    
                    if payloads:
                        all_payloads.extend(payloads)
                        file_stats["processed_files"] += 1
                    else:
                        file_stats["skipped_files"] += 1

                except Exception as e:
                    st.warning(f"Fehler bei Datei `{full_path}`: {e}")
                    file_stats["skipped_files"] += 1

            st.session_state["latest_payloads"] = all_payloads

            st.success("✅ Verarbeitungsphase abgeschlossen!")
            
            # Statistik-Metriken anzeigen
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Gesamte Dateien", file_stats["total_files"])
            m2.metric("Verarbeitete Dateien", file_stats["processed_files"])
            m3.metric("Übersprungene Dateien", file_stats["skipped_files"])
            m4.metric("Generierte Chunks", len(all_payloads))

            # 4. Qdrant Upsert Phase (falls kein Dry Run)
            if not dry_run and all_payloads:
                st.subheader("📤 Upload nach Qdrant")
                try:
                    client = QdrantClient(url=qdrant_url)
                    
                    # Collection prüfen / erstellen
                    collections = [c.name for c in client.get_collections().collections]
                    if collection_name not in collections:
                        st.info(f"Erstelle Collection `{collection_name}` mit Vektor-Dimension {vector_size}...")
                        client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                        )

                    st.info("Generiere Vektoren und lade Punkte nach Qdrant hoch...")
                    upsert_progress = st.progress(0)
                    
                    # Batch Upserting
                    batch_size = 50
                    points = []
                    
                    for p_idx, payload in enumerate(all_payloads):
                        # Platzhalter für Embedding-Erzeugung (z.B. via Ollama oder FastEmbed)
                        dummy_vector = [0.0] * vector_size
                        
                        points.append(
                            PointStruct(
                                id=p_idx,
                                vector=dummy_vector,
                                payload=payload  # Schlankes Payload OHNE redundante 'text' / 'document' Felder
                            )
                        )

                        if len(points) >= batch_size or p_idx == len(all_payloads) - 1:
                            client.upsert(collection_name=collection_name, points=points)
                            points = []
                            upsert_progress.progress((p_idx + 1) / len(all_payloads))

                    st.success(f"🎉 Erreicht! {len(all_payloads)} Chunks wurden erfolgreich nach Qdrant hochgeladen.")

                except Exception as e:
                    st.error(f"Fehler bei der Verbindung zu Qdrant: {e}")

    # --------------------------------------------------
    # TAB 2: FILTERLISTE EDITOR
    # --------------------------------------------------
    with tab_filters:
        st.subheader("🛡️ Filterregeln verwalten (`repo_filters.json`)")
        current_filters = load_repo_filters()
        
        filter_text_area = st.text_area(
            "Musterliste (Ein Eintrag pro Zeile):",
            value="\n".join(current_filters),
            height=300
        )

        if st.button("💾 Filterliste Speichern"):
            updated_filters = [line.strip() for line in filter_text_area.split("\n") if line.strip()]
            if save_repo_filters(updated_filters):
                st.success("Filterliste erfolgreich in `repo_filters.json` gespeichert!")

    # --------------------------------------------------
    # TAB 3: CHUNK VORSCHAU & PAYLOAD INSPEKTION
    # --------------------------------------------------
    with tab_preview:
        st.subheader("🔍 Vorschau der generierten Chunks & Payload-Struktur")
        if "latest_payloads" in st.session_state and st.session_state["latest_payloads"]:
            payloads = st.session_state["latest_payloads"]
            chunk_num = st.number_input("Chunk Index auswählen", min_value=0, max_value=len(payloads)-1, value=0)
            
            selected_payload = payloads[chunk_num]
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("**Inhalt (`content`):**")
                st.code(selected_payload.get("content", ""), language="markdown")
            
            with col_b:
                st.markdown("**Payload JSON (An Qdrant gesendet):**")
                st.json(selected_payload)
        else:
            st.info("Führe zuerst einen Ingestion Run durch, um die Chunks hier zu überprüfen.")


if __name__ == "__main__":
    main()
