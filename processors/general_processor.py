"""
GENERAL DOCUMENTATION PROCESSOR
-------------------------------
Verarbeitet allgemeine Dokumentationsdateien (.md, .txt, .rst).
Filtert Meta- und Verwaltungsdateien (AUTHORS, CONTRIBUTING, HISTORY, LICENSE, etc.)
rigoros heraus, insbesondere im PCB-/Hardware-Kontext.
"""

import os
from typing import Tuple, Optional
from .base_processor import BaseProcessor


class GeneralDocumentProcessor(BaseProcessor):
    category_key = "📚 Allgemeines Wissen & Dokumente"
    collection_name = "general_knowledge_base"
    supported_extensions = {".md", ".txt", ".rst"}

    PCB_CATEGORY = "⚡ PCB & Hardware Design"

    META_FILENAMES = {
        "authors", "authors.md", "authors.txt",
        "claude.md", "claude.txt",
        "contributing", "contributing.md", "contributing.txt",
        "history", "history.md", "history.txt",
        "changelog", "changelog.md", "changelog.txt",
        "license", "license.md", "license.txt", "licence",
        "code_of_conduct", "code_of_conduct.md",
        "security.md", "governance.md", "todo.md"
    }

    PCB_KEYWORDS = {
        "skidl", "kicad", "pcb", "pcbnew", "footprint", "symbol", "netlist",
        "schematic", "circuit", "gerber", "spice", "pin", "part", "erc", "drc", "hardware", "freerouting"
    }

    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        return ext in self.supported_extensions

    def parse(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_model: str, 
        ollama_client, 
        num_ctx: int, 
        max_code_len: int,
        selected_category: str = ""
    ) -> Tuple[Optional[str], Optional[str]]:
        filename = os.path.basename(rel_path).lower()

        # 1. Meta-Verwaltungsdateien immer ignorieren
        if filename in self.META_FILENAMES or any(meta in filename for meta in ["changelog", "contributing", "code_of_conduct"]):
            return "IGNORED", "SKIP"

        # 2. Im PCB-Modus: Prüfen, ob das Dokument wirklich relevanten PCB-/API-Bezug hat
        if selected_category == self.PCB_CATEGORY:
            rel_lower = rel_path.lower()
            text_snippet = raw_text[:3000].lower()

            is_relevant = any(kw in rel_lower for kw in self.PCB_KEYWORDS) or any(kw in text_snippet for kw in self.PCB_KEYWORDS)

            if not is_relevant and filename != "readme.md":
                return "IGNORED", "SKIP"

        # 3. Verarbeiten von echter Dokumentation (z. B. README.md oder API-Guides)
        enrichment_prompt = f"""Analysiere diese Dokumentation für ein RAG-System:

DATEIPFAD: {rel_path}
INHALT:
{raw_text[:max_code_len]}

Erstelle eine präzise, strukturierte Zusammenfassung auf Deutsch mit Fokus auf Funktionsweise, Verwendung und Steuerungsfragen."""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options={"num_ctx": num_ctx}
            )
            processed_md = response['message']['content']
            return "DOCUMENTATION", processed_md
        except Exception:
            return "DOCUMENTATION", f"# Dokument: {filename}\n\n{raw_text[:max_code_len]}"
