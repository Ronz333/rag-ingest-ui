"""
KICAD & SPECCTRA EDA PROCESSOR
------------------------------
Verarbeitet KiCad Footprints, Symbols, DRC-Regeln und Specctra DSN/SES.
Ist im Modus "Programmiersprachen & Software" deaktiviert.
"""

import os
import re
from typing import Tuple, Optional
from .base_processor import BaseProcessor

TB = "```"


class PcbEdaProcessor(BaseProcessor):
    category_key = "⚡ PCB & Hardware Design"
    collection_name = "pcb_knowledge_base"
    supported_extensions = {".kicad_mod", ".kicad_sym", ".kicad_pcb", ".kicad_sch", ".kicad_dru", ".rules", ".dsn", ".ses"}

    SOFTWARE_CATEGORY = "💻 Programmiersprachen & Software"

    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        # KiCad-Footprints/DSN sind keine allgemeinen Programmiersprachen
        if selected_category == self.SOFTWARE_CATEGORY:
            return False
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
        ext = os.path.splitext(rel_path)[1].lower()
        filename = os.path.basename(rel_path)

        if ext == ".kicad_dru":
            category_tag = "KICAD_DRC_RULES"
            rules_found = re.findall(r'\(rule\s+"([^"]+)"', raw_text)
            rules_str = ", ".join(rules_found) if rules_found else "Custom DRC Rules"
            markdown_content = f"""[TAG: KICAD_DRC_RULES]

# KiCad Custom DRC Rules: {filename}

- **Dateipfad:** `{rel_path}`
- **Erkannte Regeln:** `{rules_str}`

## Regel-Definition
{TB}lisp
{raw_text[:max_code_len]}
{TB}
"""
            return category_tag, markdown_content

        elif ext in {".rules", ".dsn", ".ses"}:
            category_tag = "FREEROUTING_RULES" if ext == ".rules" else "SPECCTRA_DSN"
            markdown_content = f"""[TAG: {category_tag}]

# Specctra / FreeRouting Datei: {filename}

- **Dateipfad:** `{rel_path}`

## Struktur-Definition
{TB}lisp
{raw_text[:max_code_len]}
{TB}
"""
            return category_tag, markdown_content

        elif ext == ".kicad_mod":
            category_tag = "KICAD_FOOTPRINT"
            fp_match = re.search(r'\(footprint\s+"?([^"\s)]+)"?', raw_text)
            fp_name = fp_match.group(1) if fp_match else filename
            pads = re.findall(r'\(pad\s+"([^"]+)"\s+([^\s]+)\s+([^\s]+)', raw_text)
            pad_info = f"Gesamt: {len(pads)} Pads" if pads else "Keine Pads"

            markdown_content = f"""[TAG: KICAD_FOOTPRINT]

# KiCad Footprint: {fp_name}

- **Dateipfad:** `{rel_path}`
- **Bauteil-Name:** `{fp_name}`
- **Anschlüsse / Pads:** {pad_info}

## Verwendung für Layout (`pcbnew`)
Dieser Footprint wird über `{fp_name}` adressiert.
"""
            return category_tag, markdown_content

        elif ext == ".kicad_sym":
            category_tag = "KICAD_SYM"
            sym_matches = re.findall(r'\(symbol\s+"([^"]+)"', raw_text)
            main_sym = sym_matches[0] if sym_matches else filename
            markdown_content = f"""[TAG: KICAD_SYM]

# KiCad Symbol: {main_sym}

- **Dateipfad:** `{rel_path}`
- **Symbol-Name:** `{main_sym}`

## Verwendung für SKiDL / Schaltungssynthese
Steht unter `{main_sym}` bereit.
"""
            return category_tag, markdown_content

        return "KICAD_EDA", f"[TAG: KICAD_EDA]\n\n# KiCad Datei: {filename}\n- **Dateipfad:** `{rel_path}`"
