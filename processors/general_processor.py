import os
from .base_processor import BaseProcessor


class GeneralProcessor(BaseProcessor):
    """
    Fallback-Processor für allgemeine Quellcode- und Dokumentations-Dateien (.cpp, .c, .h, .md, .rst, etc.).
    Setzt das semantische LLM-Gate (temp=0.0) mit Few-Shot-Beispielen ein, um projektfremde Dokumente auszufiltern.
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

        # Determination: Temperatur auf 0.0 erzwingen
        phase_a_options = dict(llm_options)
        phase_a_options["temperature"] = 0.0

        system_prompt = (
            "Du bist ein hochspezialisierter Filter-Assistent für ein PCB-Automatisierungs-RAG.\n\n"
            "Deine Aufgabe: Entscheide, ob der vorliegende Datei-Inhalt RELEVANT für PCB-Design, Elektronik, SKiDL, "
            "KiCad, Freerouting, Pinouts, Signalintegrität oder Bus-Protokolle ist.\n\n"
            "FEW-SHOT BEISPIELE:\n"
            "1. POSITIV (RELEVANT):\n"
            "   - Technische Spezifikationen zu Bussen (I2C, SPI, Ethernet, CAN, USB, UART).\n"
            "   - Pinout-Beschreibungen, Signal-Layout-Guides, Impedanz-Anforderungen.\n"
            "   - C/C++ Treiber mit genauen Hardware-Register-Maps und Pin-Belegungen.\n"
            "   - Freerouting-Spezifikationen, DSN-Grammatik-Dokumentation.\n\n"
            "2. NEGATIV (UNRELEVANT - sofort 'SKIP'):\n"
            "   - GUI-Code (Qt, Swing, Web-UI), Bildverarbeitung (SVG, Canvas, OpenGL).\n"
            "   - Lizenzen, ChangeLogs, READMEs ohne technische Schaltungsspezifikationen.\n"
            "   - Unit-Test-Frameworks, Mocking-Bibliotheken, CI/CD-Skripte, Build-Systeme (CMake, Makefiles).\n\n"
            "ANWEISUNG:\n"
            "Wenn die Datei UNRELEVANT ist, antworte AUSSCHLIESSLICH mit dem einzelnen Wort:\n"
            "SKIP\n\n"
            "Wenn die Datei RELEVANT ist, fass den Inhalt präzise und strukturiert für das RAG-System zusammen."
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

            return "GENERAL", generated_md

        except Exception as e:
            return "GENERAL", "SKIP"
