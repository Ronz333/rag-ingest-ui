import os
import re
from .base_processor import BaseProcessor


class GeneralProcessor(BaseProcessor):
    """
    Processor für Quellcode, Header und Dokumentation (.c, .h, .cpp, .md, .json, .yaml).
    Wandelt technische Treiber- und Hardware-Informationen direkt in deterministischen
    SKiDL-Code, Pin-Mapping-Tabellen und elektrische Parameter um.
    Verbietet jegliche Prosa-Zusammenfassungen.
    """

    def __init__(self):
        super().__init__()
        self.category_key = "🌐 General Knowledge Base"
        self.supported_extensions = [".cpp", ".c", ".h", ".hpp", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"]

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

        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern in fp_clean:
                    return "GENERAL", "SKIP"

        phase_a_options = dict(llm_options)
        phase_a_options["temperature"] = 0.0

        system_prompt = (
            "Du bist ein deterministischer Hardware-Extraktor und SKiDL-Synthesizer für einen autonomen PCB-Generator.\n\n"
            "STUFE 1: RELEVANZ-PRÜFUNG\n"
            "Ist die Datei relevant für PCB-Design, Schaltungen, Pinouts, ICs, Busse (I2C, SPI, Ethernet, CAN, UART) oder Signale?\n"
            "- NEIN -> Antworte AUSSCHLIESSLICH mit: SKIP\n"
            "- JA   -> Extrahiere die Hardware-Fakten strikt nach untenstehendem Schema.\n\n"
            "VERBOTENE PROSA & META-SPRACHE (STRENGSTENS UNTERSAGT):\n"
            "❌ NIE schreiben: 'Der Datei-Inhalt beschreibt...', 'Zusammenfassung:', 'Diese Datei ist wichtig für...'\n"
            "❌ NIE den Inhalt mit Fließtext umschreiben. Wir brauchen harte, maschinenlesbare Daten!\n\n"
            "AUSGABE-SCHEMA FÜR RELEVANTE DATEIEN (Strikte Pflicht):\n\n"
            "### 1. Pinout & Hardware-Mapping\n"
            "| Signal / Funktion | Hardware-Pin / GPIO | Schnittstelle / Bus | Bemerkung / Pegel |\n"
            "|---|---|---|---|\n"
            "| ... | ... | ... | ... |\n\n"
            "### 2. Synthetisierter SKiDL Python Code\n"
            "```python\n"
            "from skidl import *\n\n"
            "@subcircuit\n"
            "def hardware_interface_subcircuit(net_dict):\n"
            "    # Exakte Verbindungen basierend auf den Datei-Informationen\n"
            "    pass\n"
            "```\n\n"
            "### 3. Elektrische Parameter & Constraints\n"
            "- **Spannungsebenen:** [z.B. 3.3V, 5V]\n"
            "- **Takt / Frequenz:** [z.B. 50 MHz REF_CLK, 100 kHz I2C]\n"
            "- **Erforderliche Bauteile:** [z.B. 4.7k Pull-Ups an SDA/SCL, 100nF Abblockkondensator]\n"
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

            # Entfernen eventueller Rest-Präfixe wie 'RELEVANT'
            cleaned_md = re.sub(r'^(?:RELEVANT[\:\s]*)+', '', generated_md, flags=re.IGNORECASE).strip()

            return "GENERAL", cleaned_md

        except Exception:
            return "GENERAL", "SKIP"
