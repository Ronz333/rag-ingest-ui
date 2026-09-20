"""
JAVA CODE PROCESSOR
-------------------
Verarbeitet Java-Quellcode (.java), extrahiert Paketstrukturen und Klassen
und wendet allgemeine sowie dynamische Webpanel-Filter an.
"""

import os
import re
from typing import Tuple, Optional, List
from .base_processor import BaseProcessor


class JavaProcessor(BaseProcessor):
    category_key = "⚡ PCB & Hardware Design"
    collection_name = "pcb_knowledge_base"
    supported_extensions = {".java"}

    # Nur noch generische Build-Ordner fest verankert!
    GENERAL_IGNORED_PARTS = [
        "/.git/", "/build/", "/target/", "/.gradle/", "/.idea/", "/.settings/"
    ]

    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        rel_lower = rel_path.lower()
        if any(p in rel_lower for p in self.GENERAL_IGNORED_PARTS):
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
        
        rel_lower = rel_path.lower()
        filename = os.path.basename(rel_path)
        custom_filters = custom_filters or []

        # 1. Allgemeine Build-Pfade prüfen
        if any(p in rel_lower for p in self.GENERAL_IGNORED_PARTS):
            return "IGNORED", "SKIP"

        # 2. DYNAMISCHE FILTER AUS DEM WEBPANEL PRÜFEN (Pfad ODER Dateiname)
        for rule in custom_filters:
            rule_lower = rule.lower()
            if rule_lower in rel_lower or rule_lower in filename.lower():
                return "IGNORED", "SKIP"

        options_dict = num_ctx if isinstance(num_ctx, dict) else {"num_ctx": num_ctx}

        enrichment_prompt = f"""Analysiere diesen Java-Code für ein RAG-System:

DATEIPFAD: {rel_path}
CODE:
{raw_text[:max_code_len]}

Erstelle eine präzise Zusammenfassung auf Deutsch:
1. Zweck der Klasse / Schnittstelle.
2. Konkrete Steuerungsfragen zur Nutzung dieser Java-API."""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options=options_dict
            )
            processed_md = response['message']['content']
            return "JAVA_CODE", processed_md
        except Exception:
            return "JAVA_CODE", f"# Java Datei: {filename}\n\n```java\n{raw_text[:max_code_len]}\n```"
