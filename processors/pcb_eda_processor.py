import os
import re
from .base_processor import BaseProcessor


class PcbEdaProcessor(BaseProcessor):
    """
    Processor für PCB-EDA-, Specctra DSN- und Freerouting-Dateien.
    Extrahiert ausschließlich Design-Regeln, Netclasses, Layer-Setups und Routing-Constraints.
    Vergibt den Payload-Tag 'DESIGN_RULE'.
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
        custom_filters: list = None,
    ) -> tuple[str, str]:
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern in fp_clean:
                    return "DESIGN_RULE", "SKIP"

        if ext in [".o", ".obj", ".elf", ".log", ".cmakecache.txt"]:
            return "DESIGN_RULE", "SKIP"

        if ext == ".dsn":
            cleaned_dsn = self._extract_dsn_rules(raw_content)
            if not cleaned_dsn.strip():
                return "DESIGN_RULE", "SKIP"

            md_content = f"# Specctra DSN & Freerouting Design-Regeln: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{file_path}`\n\n"
            md_content += f"```lisp\n{cleaned_dsn[:max_embed_chars]}\n```"
            return "DESIGN_RULE", md_content

        elif ext == ".rules":
            md_content = f"# Freerouting Rules Definition: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{file_path}`\n\n"
            md_content += f"```text\n{raw_content[:max_embed_chars]}\n```"
            return "DESIGN_RULE", md_content

        elif ext == ".kicad_pcb":
            drc_metadata = self._extract_kicad_drc_rules(raw_content)
            if not drc_metadata.strip():
                return "DESIGN_RULE", "SKIP"

            md_content = f"# KiCad DRC & Netclass Rules: {filename}\n\n"
            md_content += f"- **Dateipfad:** `{file_path}`\n\n"
            md_content += f"```lisp\n{drc_metadata[:max_embed_chars]}\n```"
            return "DESIGN_RULE", md_content

        return "DESIGN_RULE", "SKIP"

    def _extract_dsn_rules(self, dsn_text: str) -> str:
        cleaned = re.sub(r'\(placement\s*\(.*?\)\s*\)', '', dsn_text, flags=re.DOTALL)
        cleaned = re.sub(r'\(wiring\s*\(.*?\)\s*\)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\(wire\s+.*?\)', '', cleaned)
        cleaned = re.sub(r'\(path\s+.*?\)', '', cleaned)
        cleaned = re.sub(r'\(place\s+.*?\)', '', cleaned)
        cleaned = re.sub(r'\(polygon\s+.*?\)', '', cleaned)

        lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
        return "\n".join(lines)

    def _extract_kicad_drc_rules(self, kicad_text: str) -> str:
        retained_lines = []
        for line in kicad_text.splitlines():
            line_str = line.strip()

            if any(line_str.startswith(kw) for kw in [
                "(module", "(footprint", "(fp_line", "(fp_text", "(fp_circle", "(fp_arc",
                "(pad", "(segment", "(via", "(gr_line", "(gr_text", "(zone"
            ]):
                continue

            if any(kw in line_str for kw in [
                "(kicad_pcb", "(version", "(setup", "(trace_min", "(via_size",
                "(clearance", "(netclass", "(uvia", "(tracks", "(vias"
            ]):
                retained_lines.append(line)

            if len(retained_lines) >= 300:
                break

        return "\n".join(retained_lines)
