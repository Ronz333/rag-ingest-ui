import os
import re
from .base_processor import BaseProcessor


class PcbEdaProcessor(BaseProcessor):
    """
    Processor für PCB-EDA-, Specctra DSN- und Freerouting-Dateien.
    Extrahiert ausschließlich Design-Regeln, Netclasses, Layer-Setups und Routing-Constraints.
    Filtert physische Trassen-Koordinaten und Bauteil-Platzierungen heraus.
    """

    def __init__(self):
        super().__init__()
        self.category_key = "⚡ PCB & Hardware Design"
        self.supported_extensions = [".dsn", ".rules", ".kicad_pcb"]

    def can_handle(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions

    def parse(
        self, 
        file_path: str, 
        raw_content: str, 
        model_name: str, 
        ollama_client, 
        llm_options: dict, 
        max_embed_chars: int, 
        custom_filters: list = None
    ) -> tuple[str, str]:
        # Custom Filter Check
        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern in fp_clean:
                    return "PCB_EDA", "SKIP"

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if ext == ".dsn":
            cleaned_dsn = self._extract_dsn_rules(raw_content)
            if not cleaned_dsn.strip():
                return "PCB_EDA", "SKIP"

            md_content = f"# Specctra DSN & Freerouting Design-Regeln: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{file_path}`\n\n"
            md_content += f"```lisp\n{cleaned_dsn[:max_embed_chars]}\n```"
            return "FREEROUTING_DSN", md_content

        elif ext == ".rules":
            # Direct Freerouting Rules File
            md_content = f"# Freerouting Rules Definition: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{file_path}`\n\n"
            md_content += f"```text\n{raw_content[:max_embed_chars]}\n```"
            return "FREEROUTING_RULES", md_content

        elif ext == ".kicad_pcb":
            drc_metadata = self._extract_kicad_drc_rules(raw_content)
            if not drc_metadata.strip():
                return "PCB_EDA", "SKIP"

            md_content = f"# KiCad DRC & Netclass Rules: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{file_path}`\n\n"
            md_content += f"```lisp\n{drc_metadata[:max_embed_chars]}\n```"
            return "KICAD_DRC", md_content

        return "PCB_EDA", "SKIP"

    def _extract_dsn_rules(self, dsn_text: str) -> str:
        """
        Extrahiert aus DSN-Dateien nur die logischen Abschnitte (parser, resolution, structure, 
        placement-rules, netclasses, rules). Entsorgt alle physischen Koordinaten (placement, wiring, path).
        """
        # Entferne explizite Routing-Pfade und Platzierungskoordinaten
        cleaned = re.sub(r'\(placement\s*\(.*?\)\s*\)', '', dsn_text, flags=re.DOTALL)
        cleaned = re.sub(r'\(wiring\s*\(.*?\)\s*\)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\(wire\s+.*?\)', '', cleaned)
        cleaned = re.sub(r'\(path\s+.*?\)', '', cleaned)
        cleaned = re.sub(r'\(place\s+.*?\)', '', cleaned)
        cleaned = re.sub(r'\(polygon\s+.*?\)', '', cleaned)

        lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
        return "\n".join(lines)

    def _extract_kicad_drc_rules(self, kicad_text: str) -> str:
        """
        Extrahiert aus .kicad_pcb ausschließlich DRC-Regeln, Netclasses und Setup-Einstellungen.
        """
        retained_lines = []
        for line in kicad_text.splitlines():
            line_str = line.strip()
            
            # Ignoriere Footprints, Pads, Traces & Zonen-Koordinaten
            if any(line_str.startswith(kw) for kw in [
                "(module", "(footprint", "(fp_line", "(fp_text", "(fp_circle", "(fp_arc", 
                "(pad", "(segment", "(via", "(gr_line", "(gr_text", "(zone"
            ]):
                continue
                
            # Behalte DRC-Setups, Netclasses & Abstandsregeln
            if any(kw in line_str for kw in [
                "(kicad_pcb", "(version", "(setup", "(trace_min", "(via_size", 
                "(clearance", "(netclass", "(uvia", "(tracks", "(vias"
            ]):
                retained_lines.append(line)
                
            if len(retained_lines) >= 300:
                break
                
        return "\n".join(retained_lines)
