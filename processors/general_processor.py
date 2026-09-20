"""
GENERAL DOCUMENTATION PROCESSOR
-------------------------------
Verarbeitet allgemeine Dokumentationsdateien (.md, .txt, .rst).
"""

import os
from typing import Tuple, Optional, List
from .base_processor import BaseProcessor


class GeneralDocumentProcessor(BaseProcessor):
    category_key = "📚 Allgemeines Wissen & Dokumente"
    collection_name = "general_knowledge_base"
    supported_extensions = {".md", ".txt", ".rst"}

    PCB_CATEGORY = "⚡ PCB & Hardware Design"

    IGNORED_PATH_PARTS = [
        "/_sources/", "/html/", "/_static/", "/_templates/", 
        "/docs/api/", "/build/", "/dist/", "/.git/", "/site-packages/",
        "/.agents/", "/.github/", "/.gitlab/", "/.vscode/", "/.idea/",
        "/code-quality", "/skills/", "/pre-commit", "/hooks/",
        "/fixtures/", "/tests/", "/test/", "/benchmarks/", "/scripts/benchmark/",
        "/docs/research/backlog", "/docs/reference/gradle"
    ]

    META_FILENAMES = {
        "authors", "authors.md", "authors.txt",
        "claude.md", "claude.txt",
        "contributing", "contributing.md", "contributing.txt",
        "history", "history.md", "history.txt",
        "changelog", "changelog.md", "changelog.txt",
        "license", "license.md", "license.txt", "licence",
        "code_of_conduct", "code_of_conduct.md",
        "security.md", "governance.md", "todo.md", "skill.md", "report.txt",
        "developer.md", "gradle_quick_reference.md", "backlog_triage_and_investigation_notes.md"
    }

    PCB_KEYWORDS = {
        "skidl", "kicad", "pcb", "pcbnew", "footprint", "symbol", "netlist",
        "schematic", "circuit", "gerber", "spice", "pin", "part", "erc", "drc", "hardware", "freerouting"
    }

    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        rel_lower = rel_path.lower()
        if any(p in rel_lower for p in self.IGNORED_PATH_PARTS):
            return False
        return ext in self.supported_extensions

    def parse(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_model: str, 
        ollama_client, 
        num_ctx,
        max_code_len: int,
        selected_category: str = "",
        custom_filters: List[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        filename = os.path.basename(rel_path).lower()
        rel_lower = rel_path.lower()
        custom_filters = custom_filters or []

        # 1. Standard-Ausschlüsse
        if any(p in rel_lower for p in self.IGNORED_PATH_PARTS):
            return "IGNORED", "SKIP"

        # 2. Meta-Verwaltungsdateien & Developer-Dokus
        if filename in self.META_FILENAMES or any(meta in filename for meta in ["changelog", "contributing", "code_of_conduct"]):
            return "IGNORED", "SKIP"

        # 3. Dynamische Webpanel-Filter prüfen
        for rule in custom_filters:
            rule_lower = rule.lower()
            if rule_lower in rel_lower or rule_lower in filename:
                return "IGNORED", "SKIP"

        # 4. Im PCB-Modus: Relevant-Prüfung
        if selected_category == self.PCB_CATEGORY:
            text_snippet = raw_text[:3000].lower()
            is_relevant = any(kw in rel_lower for kw in self.PCB_KEYWORDS) or any(kw in text_snippet for kw in self.PCB_KEYWORDS)

            if not is_relevant and filename != "readme.md":
                return "IGNORED", "SKIP"

        options_dict = num_ctx if isinstance(num_ctx, dict) else {"num_ctx": num_ctx}

        enrichment_prompt = f"""Analysiere diese Dokumentation für ein RAG-System:

DATEIPFAD: {rel_path}
INHALT:
{raw_text[:max_code_len]}

Erstelle eine präzise, strukturierte Zusammenfassung auf Deutsch mit Fokus auf Funktionsweise, Verwendung und Steuerungsfragen."""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options=options_dict
            )
            processed_md = response['message']['content']
            return "DOCUMENTATION", processed_md
        except Exception:
            return "DOCUMENTATION", f"# Dokument: {filename}\n\n{raw_text[:max_code_len]}"
