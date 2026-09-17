"""
JAVA & FREEROUTING INGESTION PROCESSOR
--------------------------------------
Analysiert Java-Codebases (z. B. FreeRouting Autorouter Core).
Nutzt Regex-basiertes AST-Structural-Parsing zur Methodenerfassung.
"""

import os
import re
from typing import Tuple, Optional
from .base_processor import BaseProcessor

TB = "```"


class JavaProcessor(BaseProcessor):
    category_key = "⚡ PCB & Hardware Design"
    collection_name = "pcb_knowledge_base"
    supported_extensions = {".java"}

    system_prompt = """Du bist ein Ingestion-Agent für Java-Quellcode und FreeRouting Autorouting-Kernmodule.
Analysiere den Code strikt faktengetreu und erstelle ein strukturiertes Markdown-Dokument."""

    IGNORED_PATH_PARTS = ["/docs/", "/tests/", "/build/", "/dist/", "/.git/"]

    def can_handle(self, rel_path: str, ext: str) -> bool:
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
        max_code_len: int
    ) -> Tuple[Optional[str], Optional[str]]:
        if len(raw_code) > 300000:
            return None, None

        filename = os.path.basename(rel_path)
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

        category_tag = "FREEROUTING_JAVA" if "freerouting" in rel_path.lower() else "JAVA_SOURCE"
        tag_title = f"FreeRouting Java Modul: {filename}" if category_tag == "FREEROUTING_JAVA" else f"Java Modul: {filename}"

        elements_checklist = ", ".join(sorted(list(extracted_elements))) if extracted_elements else "Keine Methoden extrahiert"

        enrichment_prompt = f"""Analysiere diesen Java-Quellcode (FreeRouting Core / EDA) für ein RAG-System:

{raw_code[:max_code_len]}

ERSTELLE FOLGENDE ABSCHNITTE AUF DEUTSCH:
1. Zweck: (Strukturierte Zusammenfassung der Routing-Logik / Datenstruktur)
2. Hauptkomponenten & Schnittstellen: (Klassen, Interfaces, Hauptmethoden)
3. Autonome Agenten- & API-Anwendungsfragen: Erstelle eine vollständige Liste von Steuerungsfragen.

STRIKTE REGELN:
a) ERZWUNGENE JAVA-ELEMENT-ABDECKUNG: Generiere ZWINGEND für JEDES dieser Elemente mindestens eine Frage:
   [{elements_checklist}]
b) FORMULIERUNG: Verwende "Wie steuere / nutze / konfiguriere ich X mit dieser Java API?"-Fragen."""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options={"num_ctx": num_ctx}
            )
            enrichment_text = response['message']['content']
        except Exception:
            enrichment_text = f"**Zweck:** Java Modul (`{rel_path}`)\n**Package:** `{package_name}`"

        code_snippet = raw_code[:max_code_len] + ("\n// ... [Code gekürzt]" if len(raw_code) > max_code_len else "")

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
