import os
import re
from .base_processor import BaseProcessor


class PcbEdaProcessor(BaseProcessor):
    """
    Processor für PCB-EDA-, Specctra DSN- und Freerouting-Dateien (.dsn, .rules, .kicad_pcb).
    Nutzt deterministische Regel-Extraktion und bei Syntax-Fehlern eine 3-stufige Selbstkorrektur mit temp=0.0.
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
        from quality_control import QualityControl

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern in fp_clean:
                    return "DESIGN_RULE", "SKIP"

        if ext == ".dsn":
            extracted_body = self._extract_dsn_rules(raw_content)
            doc_type = "Specctra DSN & Freerouting Design-Regeln"
            lang_tag = "lisp"
        elif ext == ".rules":
            extracted_body = raw_content.strip()
            doc_type = "Freerouting Rules Definition"
            lang_tag = "text"
        elif ext == ".kicad_pcb":
            extracted_body = self._extract_kicad_drc_rules(raw_content)
            doc_type = "KiCad DRC & Netclass Rules"
            lang_tag = "lisp"
        else:
            return "DESIGN_RULE", "SKIP"

        if not extracted_body.strip():
            return "DESIGN_RULE", "SKIP"

        initial_md = f"# {doc_type}: {filename}\n\n"
        initial_md += f"- **Dateipfad:** `{file_path}`\n\n"
        initial_md += f"```{lang_tag}\n{extracted_body[:max_embed_chars]}\n```"

        is_valid, err_msg = QualityControl.validate("DESIGN_RULE", initial_md, file_path)
        if is_valid:
            return "DESIGN_RULE", initial_md

        # Self-Correction-Loop mit temp=0.0
        phase_a_options = dict(llm_options)
        phase_a_options["temperature"] = 0.0

        max_attempts = 3
        current_md = initial_md
        last_error = err_msg

        for attempt in range(1, max_attempts + 1):
            repair_system_prompt = (
                "Du bist ein EDA-Software-Spezialist für Specctra DSN, KiCad und Freerouting.\n"
                "Deine Aufgabe ist es, den vorliegenden Markdown-Block mit Design-Regeln zu reparieren.\n\n"
                "REGELN:\n"
                "1. Korrigiere alle S-Expression-Klammerfehler.\n"
                "2. Stelle sicher, dass keine Trassen-Koordinaten ((path, (wire, (placement) enthalten sind.\n"
                "3. Antworte mit 'SKIP', wenn der Inhalt keine DSN/DRC-Regeln enthält.\n\n"
                f"⚠️ KORREKTUR-AUFFORDERUNG (VERSUCH {attempt}/{max_attempts}):\n"
                f"Fehler: {last_error}"
            )

            user_message = f"Datei: {filename}\n\nDefekter Inhalt:\n{current_md}"

            try:
                response = ollama_client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": repair_system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    options=phase_a_options,
                )
                repaired_md = response["message"]["content"].strip()

                if repaired_md == "SKIP" or repaired_md.startswith("SKIP"):
                    return "DESIGN_RULE", "SKIP"

                is_valid, err_msg = QualityControl.validate("DESIGN_RULE", repaired_md, file_path)
                if is_valid:
                    return "DESIGN_RULE", repaired_md

                last_error = err_msg
                current_md = repaired_md

            except Exception as e:
                last_error = str(e)

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
