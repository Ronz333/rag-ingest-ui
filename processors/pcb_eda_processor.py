"""
PCB & EDA PROCESSOR
-------------------
Verarbeitet Hardware-, Layout- und EDA-Dateien (.dsn, .kicad_pcb, .kicad_mod, .kicad_sym, .sch, .pro).
"""

import os
import re
from typing import Tuple, Optional, List
from .base_processor import BaseProcessor


class PcbEdaProcessor(BaseProcessor):
    category_key = "⚡ PCB & Hardware Design"
    collection_name = "pcb_knowledge_base"
    supported_extensions = {".dsn", ".kicad_pcb", ".kicad_mod", ".kicad_sym", ".sch", ".pro"}

    IGNORED_PATH_PARTS = [
        "/fixtures/", "/tests/", "/test/", "/benchmarks/", "/scripts/benchmark/",
        "/.git/", "/build/", "/dist/", "/.agents/", "/.github/"
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

        ext = os.path.splitext(rel_path)[1].lower()

        if ext == ".dsn":
            cleaned_dsn = self._clean_dsn_content(raw_text)
            if not cleaned_dsn.strip():
                return "IGNORED", "SKIP"

            md_content = f"# Specctra DSN Design-Regeln: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{rel_path}`\n\n"
            md_content += f"```lisp\n{cleaned_dsn[:max_code_len]}\n```"
            return "SPECCTRA_DSN", md_content

        elif ext in {".kicad_pcb", ".kicad_mod", ".kicad_sym"}:
            rules_and_headers = self._extract_kicad_metadata(raw_text)
            md_content = f"# KiCad EDA Definition: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{rel_path}`\n\n"
            md_content += f"```lisp\n{rules_and_headers[:max_code_len]}\n```"
            return "KICAD_EDA", md_content

        else:
            md_content = f"# PCB EDA Datei: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{rel_path}`\n\n"
            md_content += f"```text\n{raw_text[:max_code_len]}\n```"
            return "PCB_EDA", md_content

    def _clean_dsn_content(self, dsn_text: str) -> str:
        cleaned = re.sub(r'\(placement\s*\(.*?\)\s*\)', '', dsn_text, flags=re.DOTALL)
        cleaned = re.sub(r'\(plane\s+.*?\)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\(polygon\s+.*?\)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\(place\s+.*?\)', '', cleaned)
        cleaned = re.sub(r'\(wire\s+.*?\)', '', cleaned)
        cleaned = re.sub(r'\(path\s+.*?\)', '', cleaned)
        
        lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
        return "\n".join(lines)

    def _extract_kicad_metadata(self, kicad_text: str) -> str:
        lines = []
        for line in kicad_text.splitlines():
            line_str = line.strip()
            if any(kw in line_str for kw in ["(version", "(generator", "(setup", "(clearance", "(trace_min", "(via_size", "(layer", "(net"]):
                lines.append(line)
            if len(lines) >= 150:
                break
        return "\n".join(lines) if lines else kicad_text[:2000]
