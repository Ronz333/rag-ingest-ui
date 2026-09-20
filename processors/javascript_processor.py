"""
JAVASCRIPT CODE PROCESSOR
-------------------------
Verarbeitet JavaScript/TypeScript-Quellcode (.js, .ts, .jsx, .tsx).
"""

import os
from typing import Tuple, Optional, List
from .base_processor import BaseProcessor


class JavascriptProcessor(BaseProcessor):
    category_key = "⚡ PCB & Hardware Design"
    collection_name = "pcb_knowledge_base"
    supported_extensions = {".js", ".ts", ".jsx", ".tsx"}

    IGNORED_PATH_PARTS = [
        "/node_modules/", "/dist/", "/build/", "/.next/", "/coverage/",
        "/fixtures/", "/tests/", "/test/", "/.git/", "/.agents/", "/.github/"
    ]

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
        rel_lower = rel_path.lower()
        filename = os.path.basename(rel_path)
        custom_filters = custom_filters or []

        if any(p in rel_lower for p in self.IGNORED_PATH_PARTS):
            return "IGNORED", "SKIP"

        for rule in custom_filters:
            rule_lower = rule.lower()
            if rule_lower in rel_lower or rule_lower in filename.lower():
                return "IGNORED", "SKIP"

        options_dict = num_ctx if isinstance(num_ctx, dict) else {"num_ctx": num_ctx}

        enrichment_prompt = f"""Analysiere diesen JavaScript/TypeScript-Code für ein RAG-System:

DATEIPFAD: {rel_path}
CODE:
{raw_text[:max_code_len]}

Erstelle eine präzise Zusammenfassung auf Deutsch:
1. Zweck und Rolle im Gesamtsystem.
2. Wichtige exportierte Funktionen/Klassen und deren Nutzung."""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options=options_dict
            )
            processed_md = response['message']['content']
            return "JS_CODE", processed_md
        except Exception:
            return "JS_CODE", f"# JavaScript/TypeScript Datei: {filename}\n\n```javascript\n{raw_text[:max_code_len]}\n```"
