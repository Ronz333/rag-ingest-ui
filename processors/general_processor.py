import os
import re
from .base_processor import BaseProcessor


class GeneralProcessor(BaseProcessor):
    """
    Processor für Quellcode, Header, SKiDL-Python-Klassen und Dokumentation (.c, .h, .cpp, .py, .md, .json, .yaml).
    Extrahiert deterministisch Pinouts, Register-Maps und Hardware-Konfigurationen.
    """

    def __init__(self):
        super().__init__()
        self.category_key = "🌐 General Knowledge Base"
        self.supported_extensions = [".cpp", ".c", ".h", ".hpp", ".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"]

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
        filename = os.path.basename(file_path)

        # Exakter Pfadfilter: Nur filtern, wenn ein relativer Pfad explizit gematcht wird
        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern.strip() and pattern.strip() in fp_clean:
                    # Sicherstellen, dass nicht versehentlich allgemeine Wörter gematcht werden
                    if pattern.strip().startswith("/") or pattern.strip().endswith("/"):
                        return "GENERAL", "SKIP"

        phase_a_options = dict(llm_options)
        phase_a_options["temperature"] = 0.0

        system_prompt = (
            "Du bist ein deterministischer Hardware-Extraktor und SKiDL-Synthesizer für ein autonomes PCB-RAG-System.\n\n"
            "AUFGABE & RELEVANZ-KRITERIEN:\n"
            "Analysiere den vorliegenden Datei-Inhalt (.c, .h, .py, .md, etc.).\n"
            "Eine Datei ist RELEVANT und DARF NICHT UEBERSEHEN WERDEN, wenn sie mindestens eines der folgenden Elemente enthält:\n"
            "- GPIO-Pinbelegungen, Register-Definitions, Peripheral-Initialisierungen (Ethernet PHY/MAC, I2C, SPI, CAN, UART, Clock, PWM, Relais, Buttons).\n"
            "- C/C++ Treiber-Code oder Header mit Pinouts, Hardware-Konfigurationen oder Register-Maps.\n"
            "- SKiDL Python-Code (Pins, Parts, Nets, Subcircuits, Klassen-Definitionen wie pin.py, part.py).\n"
            "- Technische Dokumentation (Pinouts, Signalbeschreibungen, Layout-Hinweise, Board-Revisionen).\n\n"
            "WANN SOLL 'SKIP' GEWÄHLT WERDEN?\n"
            "- AUSSCHLIESSLICH bei reinem GUI-Code, Bildverarbeitungs-Logik, CI/CD-Pipelines oder völlig inhaltsleeren Platzhaltern.\n"
            "- WICHTIG: Lizenzhinweise oder Header-Kommentare (z.B. 'Licensed under Apache/MIT') sind KEIN Grund zum Skippen! "
            "Wenn die Datei Code, Pinouts oder Hardware-Klassen enthält, verarbeite sie UNBEDINGT!\n\n"
            "VERBOTENE PROSA & META-SPRACHE (STRENGSTENS UNTERSAGT):\n"
            "❌ NIE schreiben: 'Der Datei-Inhalt beschreibt...', 'Zusammenfassung:', 'Diese Datei ist wichtig für...'\n\n"
            "AUSGABE-SCHEMA FÜR RELEVANTE DATEIEN:\n\n"
            "### 1. Pinout & Hardware-Mapping\n"
            "| Signal / Funktion | Hardware-Pin / GPIO | Schnittstelle / Bus | Bemerkung / Pegel |\n"
            "|---|---|---|---|\n"
            "| ... | ... | ... | ... |\n\n"
            "### 2. Synthetisierter SKiDL Python Code / Code-Struktur\n"
            "```python\n"
            "from skidl import *\n"
            "# Exakte Verbindungen, SKiDL-Definitionen oder Hardware-Interface\n"
            "```\n\n"
            "### 3. Elektrische Parameter & Constraints\n"
            "- **Spannungsebenen:** [z.B. 3.3V, 5V]\n"
            "- **Takt / Frequenz:** [z.B. 50 MHz REF_CLK, 100 kHz I2C]\n"
            "- **Schnittstellen & Pins:** [Genaue Pin-Liste]\n"
        )

        user_message = f"Datei: {filename}\nPfad: {file_path}\n\nInhalt:\n{raw_content[:max_embed_chars]}"

        try:
            response = ollama_client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options=phase_a_options,
            )
            generated_md = response["message"]["content"].strip()

            if generated_md == "SKIP" or generated_md.startswith("SKIP"):
                return "GENERAL", "SKIP"

            cleaned_md = re.sub(r'^(?:RELEVANT[\:\s]*)+', '', generated_md, flags=re.IGNORECASE).strip()

            return "GENERAL", cleaned_md

        except Exception:
            return "GENERAL", "SKIP"
