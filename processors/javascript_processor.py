"""
JAVASCRIPT & TYPESCRIPT INGESTION PROCESSOR
-------------------------------------------
Analysiert JS/TS Codebases. Reagiert im PCB-Modus nur auf spezifische EDA-Plugins.
"""

import os
import re
from typing import Tuple, Optional
from .base_processor import BaseProcessor

TB = "```"


class JavaScriptProcessor(BaseProcessor):
    category_key = "🌐 JavaScript & TypeScript"
    collection_name = "programming_knowledge_base"
    supported_extensions = {".js", ".jsx", ".ts", ".tsx", ".mjs"}

    PCB_CATEGORY = "⚡ PCB & Hardware Design"
    PCB_KEYWORDS = {"easyeda", "kicad", "pcb", "gerber", "bom", "eda"}

    system_prompt = """Du bist ein Ingestion-Agent für JavaScript/TypeScript-Code.
Analysiere den Code strikt faktengetreu und erstelle ein strukturiertes Markdown-Dokument."""

    IGNORED_PATH_PARTS = ["/node_modules/", "/dist/", "/build/", "/.next/", "/.git/"]

    def _is_pcb_relevant(self, rel_path: str, raw_code: str) -> bool:
        rel_lower = rel_path.lower()
        if any(kw in rel_lower for kw in self.PCB_KEYWORDS):
            return True
        code_snippet = raw_code[:3000].lower()
        return any(kw in code_snippet for kw in self.PCB_KEYWORDS)

    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        rel_lower = rel_path.lower()
        if any(p in rel_lower for p in self.IGNORED_PATH_PARTS):
            return False
        return ext in self.supported_extensions

    def parse(
        self, 
        rel_path: str, 
        raw_code: str, 
        active_model: str, 
        ollama_client, 
        num_ctx: int, 
        max_code_len: int,
        selected_category: str = ""
    ) -> Tuple[Optional[str], Optional[str]]:
        if selected_category == self.PCB_CATEGORY and not self._is_pcb_relevant(rel_path, raw_code):
            return "IGNORED", "SKIP"

        if len(raw_code) > 300000:
            return "IGNORED", "SKIP"

        filename = os.path.basename(rel_path)

        functions = re.findall(r'(?:function\s+([A-Za-z0-9_]+)|const\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)', raw_code)
        classes = re.findall(r'class\s+([A-Za-z0-9_]+)', raw_code)

        extracted = set()
        for f_tuple in functions:
            name = f_tuple[0] or f_tuple[1]
            if name:
                extracted.add(f"function:{name}()")
        for c in classes:
            extracted.add(f"class:{c}")

        elements_checklist = ", ".join(sorted(list(extracted))) if extracted else "Keine Funktionen extrahiert"
        category_tag = "TYPESCRIPT_MODULE" if rel_path.endswith(('.ts', '.tsx')) else "JAVASCRIPT_MODULE"

        enrichment_prompt = f"""Analysiere diesen JS/TS-Quellcode für ein RAG-System:

{raw_code[:max_code_len]}

ERSTELLE FOLGENDE ABSCHNITTE AUF DEUTSCH:
1. Zweck: (Zusammenfassung der Modulrolle)
2. Hauptkomponenten & Schnittstellen: (Klassen, Exportierte Funktionen, Komponenten)
3. Autonome Agenten- & API-Anwendungsfragen: Erstelle eine vollständige Liste von Steuerungsfragen.

STRIKTE REGELN:
a) ERZWUNGENE ABDECKUNG: Generiere ZWINGEND für JEDES dieser Elemente mindestens eine Steuerungsfrage:
   [{elements_checklist}]
b) FORMULIERUNG: Verwende "Wie steuere / nutze / erstelle / konfiguriere ich X mit dieser API?"-Fragen."""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options={"num_ctx": num_ctx}
            )
            enrichment_text = response['message']['content']
        except Exception:
            enrichment_text = f"**Zweck:** JS/TS Modul (`{rel_path}`)"

        code_snippet = raw_code[:max_code_len] + ("\n// ... [Code gekürzt]" if len(raw_code) > max_code_len else "")

        markdown_content = f"""[TAG: {category_tag}]

# JS/TS Modul: {filename}

- **Quelle/Dateipfad:** `{rel_path}`

## Code-Analyse & Dokumentation
{enrichment_text}

## Validierter Quellcode (JavaScript/TypeScript)
{TB}javascript
{code_snippet}
{TB}
"""
        return category_tag, markdown_content
