import os
import re
import json
from .base_processor import BaseProcessor


class GeneralProcessor(BaseProcessor):
    """
    Processor für Quellcode, Header, SKiDL-Klassen, Freerouting-Rules und Doku.
    Nutzt Cloudflare Clef-27B als ultraschnelles JSON Decision Gate (Phase A1)
    und ein Coder-LLM für die deterministische Synthese (Phase A2).
    """

    def __init__(self):
        super().__init__()
        self.category_key = "🌐 General Knowledge Base"
        self.supported_extensions = [
            ".cpp", ".c", ".h", ".hpp", ".py", ".md", ".rst", 
            ".txt", ".json", ".yaml", ".yml", ".dsn", ".rules"
        ]

    def can_handle(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions

    def parse(
        self,
        file_path: str,
        raw_content: str,
        model_name: str,
        clef_model_name: str,
        ollama_client,
        llm_options: dict,
        max_embed_chars: int,
        custom_filters: list = None,
    ) -> tuple[str, str]:
        filename = os.path.basename(file_path)

        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern.strip() and pattern.strip() in fp_clean:
                    if pattern.strip().startswith("/") or pattern.strip().endswith("/"):
                        return "GENERAL", "SKIP"

        # =========================================================================
        # PHASE A1: CLEF-27B DECISION GATE (Zero-Token JSON Klassifizierung)
        # =========================================================================
        clef_prompt = (
            "Du bist ein praeziser Hardware-Decision-Head (Cloudflare Clef Gate).\n"
            "Deine Aufgabe ist die sofortige Klassifizierung von Dateien für ein autonomes PCB-RAG-System.\n\n"
            "RELEVANZ-KRITERIEN (is_hardware_relevant = true):\n"
            "1. SKiDL & Python Hardware-Generierung (Pin-Klassen, Footprints, Subcircuits, Netze, SKiDL Core Engine).\n"
            "2. KiCad Bibliotheken & Symbole (Symbol-Definitionen, Pinouts, Footprint- Zuordnungen).\n"
            "3. Freerouting & Design Rules (DSN-Grammatik, Clearance, Via-Rules, Trace-Widths, Net-Classes).\n"
            "4. Embedded C/C++ Treiber & Header mit Hardware-Registern, Pinouts, Bus-Protokollen (Ethernet PHY/MAC, I2C, SPI, CAN, UART, Clocks).\n"
            "5. Technische Datenblätter und Hardware-Dokumentation mit Pinbelegungen oder Layout-Vorgaben.\n\n"
            "IRRELEVANT (is_hardware_relevant = false):\n"
            "- GUI-Code, Bildverarbeitung, reine Web-Frontends, Build-Müll, unvollständige Fragmente ohne Hardware-Bezug.\n\n"
            "ANTWORTE AUSSCHLIESSLICH IM FOLGENDEN JSON-FORMAT:\n"
            "{\n"
            '  "is_hardware_relevant": true,\n'
            '  "category_tag": "SKIDL_SUBCIRCUIT" | "KICAD_SYMBOL" | "DESIGN_RULE" | "GENERAL"\n'
            "}"
        )

        clef_user_msg = f"Datei: {filename}\nPfad: {file_path}\n\nInhalt:\n{raw_content[:4000]}"

        clef_options = dict(llm_options)
        clef_options["temperature"] = 0.0

        target_tag = "GENERAL"

        try:
            clef_res = ollama_client.chat(
                model=clef_model_name,
                messages=[
                    {"role": "system", "content": clef_prompt},
                    {"role": "user", "content": clef_user_msg}
                ],
                format="json",
                options=clef_options
            )
            clef_json = json.loads(clef_res["message"]["content"].strip())

            is_relevant = clef_json.get("is_hardware_relevant", False)
            target_tag = clef_json.get("category_tag", "GENERAL")

            if not is_relevant:
                return target_tag, "SKIP"

        except Exception:
            # Fallback falls Clef-27B nicht erreichbar ist: Nicht abbrechen, sondern verarbeiten
            pass

        # =========================================================================
        # PHASE A2: GENERATIVE LLM SYNTHESE (Deterministische Extraktion)
        # =========================================================================
        synth_system_prompt = (
            "Du bist ein deterministischer Hardware-Extraktor und SKiDL/PCB-Synthesizer für einen autonomen PCB-Generator.\n\n"
            "DEINE AUFGABE:\n"
            "Extrahiere die Hardware-Fakten, Pinouts, Routing-Regeln oder SKiDL-Definitionen strikt nach untenstehendem Schema.\n\n"
            "STRENGSTE REGELN:\n"
            "❌ KEINE PROSA: Nie 'Der Datei-Inhalt beschreibt...', 'Zusammenfassung:' oder 'Diese Datei ist wichtig' schreiben!\n"
            "❌ KEINE LIZENZHINWEISE: Ignoriere Kopfteil-Kommentare zu Lizenzen völlig.\n\n"
            "AUSGABE-SCHEMA:\n\n"
            "### 1. Pinout & Hardware-Mapping\n"
            "| Signal / Funktion | Hardware-Pin / GPIO | Schnittstelle / Bus | Bemerkung / Pegel |\n"
            "|---|---|---|---|\n"
            "| ... | ... | ... | ... |\n\n"
            "### 2. Synthetisierter SKiDL Python Code / PCB Rules\n"
            "```python\n"
            "from skidl import *\n"
            "# Exakte SKiDL Subcircuits, Netz-Verbindungen oder Freerouting/KiCad Parameter\n"
            "```\n\n"
            "### 3. Elektrische Parameter & Constraints\n"
            "- **Spannungsebenen:** [z.B. 3.3V, 5V]\n"
            "- **Takt / Frequenz:** [z.B. 50 MHz REF_CLK, 100 kHz I2C]\n"
            "- **Schnittstellen & Pins:** [Genaue Liste]\n"
        )

        user_message = f"Datei: {filename}\nPfad: {file_path}\n\nInhalt:\n{raw_content[:max_embed_chars]}"

        try:
            response = ollama_client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": synth_system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options=llm_options,
            )
            generated_md = response["message"]["content"].strip()

            if generated_md == "SKIP" or generated_md.startswith("SKIP"):
                return target_tag, "SKIP"

            cleaned_md = re.sub(r'^(?:RELEVANT[\:\s]*)+', '', generated_md, flags=re.IGNORECASE).strip()
            return target_tag, cleaned_md

        except Exception:
            return target_tag, "SKIP"
