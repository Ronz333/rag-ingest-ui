"""
PYTHON & SKIDL INGESTION PROCESSOR
----------------------------------
Analysiert Python-Quellcode. Filtert im PCB-Modus strikt auf EDA-, SKiDL- 
und KiCad-Relevanz, um allgemeine Python-Skripte im PCB-Kontext zu ignorieren.
"""

import os
import ast
import warnings
from typing import Tuple, Optional
from .base_processor import BaseProcessor

TB = "```"


class PythonProcessor(BaseProcessor):
    category_key = "💻 Programmiersprachen & Software"
    collection_name = "programming_knowledge_base"
    supported_extensions = {".py"}

    PCB_CATEGORY = "⚡ PCB & Hardware Design"
    PCB_KEYWORDS = {
        "skidl", "kicad", "pcb", "pcbnew", "footprint", "symbol", "netlist", 
        "schematic", "circuit", "gerber", "spice", "pin", "part", "erc", "drc", "hardware"
    }

    system_prompt = """Du bist ein Ingestion-Agent für Python-Quellcode und SKiDL/KiCad-APIs.
Analysiere den Code strikt faktengetreu und erstelle ein strukturiertes Markdown-Dokument.
Erstelle eine exhaustive Liste von Entwickler- und Agenten-Steuerungsfragen ("Wie steuere/nutze/konfiguriere ich X?")."""

    IGNORED_FILENAMES = {"setup.py", "conftest.py"}
    IGNORED_PATH_PARTS = ["/docs/", "/tests/", "/build/", "/dist/", "/.git/"]

    def _is_pcb_relevant(self, rel_path: str, raw_code: str) -> bool:
        """Prüft, ob eine Python-Datei PCB/EDA-Relevanz aufweist."""
        rel_lower = rel_path.lower()
        if any(kw in rel_lower for kw in self.PCB_KEYWORDS):
            return True
        
        # Inhaltsprüfung auf Importe und Begriffe
        code_snippet = raw_code[:3000].lower()
        return any(kw in code_snippet for kw in self.PCB_KEYWORDS)

    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        filename = os.path.basename(rel_path)
        rel_lower = rel_path.lower()
        if filename in self.IGNORED_FILENAMES or any(p in rel_lower for p in self.IGNORED_PATH_PARTS):
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
        # PCB-Kategorie Filter: Allgemeiner Code ohne PCB-Bezug wird übersprungen
        if selected_category == self.PCB_CATEGORY and not self._is_pcb_relevant(rel_path, raw_code):
            return "IGNORED", "SKIP"

        if len(raw_code) > 300000:
            return "IGNORED", "SKIP"

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

        rel_lower = rel_path.lower()
        filename = os.path.basename(rel_path)

        if "skidl" in rel_lower:
            category_tag = "SKIDL_API"
            tag_title = f"SKiDL API Modul: {filename}"
        elif "pcbnew" in rel_lower or "kicad" in rel_lower:
            category_tag = "KICAD_PCBNEW"
            tag_title = f"KiCad / PCBNew Plugin: {filename}"
        else:
            category_tag = "PYTHON_MODULE"
            tag_title = f"Python Modul: {filename}"

        elements_checklist = ", ".join(sorted(list(extracted_elements))) if extracted_elements else "Keine Methoden extrahiert"

        enrichment_prompt = f"""Analysiere diesen Python-Quellcode für ein RAG-System:

{raw_code[:max_code_len]}

ERSTELLE FOLGENDE ABSCHNITTE AUF DEUTSCH:
1. Zweck: (Strukturierte Zusammenfassung)
2. Hauptkomponenten & Schnittstellen: (Klassen, Methoden, Parameter)
3. Autonome Agenten- & API-Anwendungsfragen: Erstelle eine vollständige Liste präziser Fragen.

STRIKTE REGELN:
a) ERZWUNGENE AST-ABDECKUNG: Generiere ZWINGEND für JEDES dieser Elemente mindestens eine Steuerungsfrage:
   [{elements_checklist}]
b) FORMULIERUNG: Verwende AUSSCHLIESSLICH "Wie steuere / nutze / erstelle / vergleiche / konfiguriere ich X mit dieser API?"-Fragen."""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options={"num_ctx": num_ctx}
            )
            enrichment_text = response['message']['content']
        except Exception:
            enrichment_text = f"**Zweck:** Python Modul ({rel_path})\n**Docstring:** {docstring}"

        code_snippet = raw_code[:max_code_len] + ("\n# ... [Code gekürzt]" if len(raw_code) > max_code_len else "")

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
