"""
PYTHON CODE PROCESSOR
---------------------
Verarbeitet Python-Quellcode (.py), führt AST-Analysen durch und ignoriert
Test-Fixtures sowie Build-Artefakte.
"""

import os
import ast
from typing import Tuple, Optional
from .base_processor import BaseProcessor


class PythonProcessor(BaseProcessor):
    category_key = "⚡ PCB & Hardware Design"
    collection_name = "pcb_knowledge_base"
    supported_extensions = {".py"}

    IGNORED_PATH_PARTS = [
        "/fixtures/", "/tests/", "/test/", "/benchmarks/", "/scripts/benchmark/",
        "/.git/", "/build/", "/dist/", "/.agents/", "/.github/", "/__pycache__/"
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
        selected_category: str = ""
    ) -> Tuple[Optional[str], Optional[str]]:
        rel_lower = rel_path.lower()
        if any(p in rel_lower for p in self.IGNORED_PATH_PARTS):
            return "IGNORED", "SKIP"

        filename = os.path.basename(rel_path)
        
        # AST-Analyse zur Extraktion von Klassen und Funktionen
        ast_elements = []
        try:
            tree = ast.parse(raw_text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    ast_elements.append(f"class:{node.name}")
                elif isinstance(node, ast.FunctionDef):
                    ast_elements.append(f"method:{node.name}()")
        except Exception:
            pass

        ast_summary = ", ".join(ast_elements[:30]) if ast_elements else "Keine Klassen/Methoden erkannt."
        options_dict = num_ctx if isinstance(num_ctx, dict) else {"num_ctx": num_ctx}

        enrichment_prompt = f"""Analysiere diesen Python-Code für ein PCB/Hardware-RAG-System:

DATEIPFAD: {rel_path}
EXAKTE AST-METHODEN: {ast_summary}
CODE-AUSSCHNITT:
{raw_text[:max_code_len]}

Erstelle eine präzise Zusammenfassung auf Deutsch:
1. Zweck und Rolle im Hardware-/Layout-System.
2. Für jede relevante Methode eine konkrete Steuerungsfrage für einen PCB-Agenten."""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options=options_dict
            )
            processed_md = response['message']['content']
            return "SKIDL_API" if "skidl" in rel_lower else "PYTHON_CODE", processed_md
        except Exception:
            return "PYTHON_CODE", f"# Python Datei: {filename}\n\nAST: {ast_summary}\n\n```python\n{raw_text[:max_code_len]}\n```"
