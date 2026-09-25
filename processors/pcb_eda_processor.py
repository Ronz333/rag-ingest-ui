import os
import re
from .base_processor import BaseProcessor
from quality_control import QualityControl


class PcbEdaProcessor(BaseProcessor):
    """
    Processor für PCB-EDA-, Specctra DSN- und Freerouting-Dateien.
    Extrahiert ausschließlich Design-Regeln, Netclasses, Layer-Setups und Routing-Constraints.
    Filtert physische Trassen-Koordinaten und Platzierungsblöcke heraus.
    Nutzt einen 3-stufigen Selbstkorrektur-Loop mit QualityControl zur Validierung von S-Expressions.
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

        # 1. Custom Filter Check
        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern in fp_clean:
                    return "DESIGN_RULE", "SKIP"

        # 2. Strikter Ausschluss von Binär-, Build- und Logdateien
        if ext in [".o", ".obj", ".elf", ".log", ".cmakecache.txt"]:
            return "DESIGN_RULE", "SKIP"

        # 3. Deterministische Extraktion je nach Dateityp
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

        # Erster Markdown-Entwurf erzeugen
        initial_md = f"# {doc_type}: {filename}\n\n"
        initial_md += f"- **Dateipfad:** `{file_path}`\n\n"
        initial_md += f"```{lang_tag}\n{extracted_body[:max_embed_chars]}\n```"

        # 4. Qualitätskontrolle (QC) direkt durchführen
        is_valid, err_msg = QualityControl.validate("DESIGN_RULE", initial_md, file_path)
        if is_valid:
            return "DESIGN_RULE", initial_md

        # 5. SELBSTKORREKTUR-SCHLEIFE (UP TO 3 ATTEMPTS) FÜR SYNTAX-REPARATUR
        max_attempts = 3
        current_md = initial_md
        last_error = err_msg

        for attempt in range(1, max_attempts + 1):
            repair_system_prompt = (
                "Du bist ein EDA-Software-Spezialist für Specctra DSN, KiCad und Freerouting.\n"
                "Deine Aufgabe ist es, den vorliegenden Markdown-Block mit Design-Regeln zu reparieren.\n\n"
                "REGELN:\n"
                "1. Korrigiere alle S-Expression-Klammerfehler (öffnende und schließende Klammern müssen exakt übereinstimmen).\n"
                "2. Stelle sicher, dass keine Trassen-Koordinaten ((path, (wire, (placement) enthalten sind.\n"
                "3. Gib ausschließlich den korrigierten Markdown-Block aus.\n\n"
                f"⚠️ KORREKTUR-AUFFORDERUNG (VERSUCH {attempt}/{max_attempts}):\n"
                f"Der vorherige Entwurf schlug fehl mit folgendem Fehler:\n-> {last_error}"
            )

            user_message = f"Datei: {filename}\n\nDefekter Inhalt:\n{current_md}"

            try:
                response = ollama_client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": repair_system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    options=llm_options,
                )
                repaired_md = response["message"]["content"].strip()

                is_valid, err_msg = QualityControl.validate("DESIGN_RULE", repaired_md, file_path)
                if is_valid:
                    return "DESIGN_RULE", repaired_md

                last_error = err_msg
                current_md = repaired_md

            except Exception as e:
                last_error = str(e)

        # Falls auch nach 3 Versuchen kein valider DRC/DSN-Chunk entstand: Verwerfen
        return "DESIGN_RULE", "SKIP"

    def _extract_dsn_rules(self, dsn_text: str) -> str:
        """
        Extrahiert aus DSN-Dateien nur die logischen Abschnitte (parser, resolution, structure, 
        rules, netclasses). Entsorgt alle physischen Koordinaten (placement, wiring, path, wire, place, polygon).
        """
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
        Verwirft Traces, Vias, Footprints und Zeichnungen.
        """
        retained_lines = []
        for line in kicad_text.splitlines():
            line_str = line.strip()

            # Ignoriere Footprint-Zeichnungen, Pads, Traces & Zonen
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
