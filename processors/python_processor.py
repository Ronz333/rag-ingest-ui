import os
from .base_processor import BaseProcessor


class PythonProcessor(BaseProcessor):
    """
    Processor für Python-Quellcode (.py).
    Filtert mittels semantischem LLM-Gate (temp=0.0) GUI-Code, Vektor-Tools,
    Test-Frameworks und CI-Skripte heraus und extrahiert nur PCB/SKiDL-relevante Logik.
    """

    def __init__(self):
        super().__init__()
        self.category_key = "⚡ PCB & Hardware Design"
        self.supported_extensions = [".py"]

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
                    return "SKIDL_SUBCIRCUIT", "SKIP"

        # Determination: Temperatur auf 0.0 erzwingen
        phase_a_options = dict(llm_options)
        phase_a_options["temperature"] = 0.0

        system_prompt = (
            "Du bist ein hochspezialisierter Filter-Assistent und Python-Analyst für ein PCB-Automatisierungs-RAG.\n\n"
            "Deine Aufgabe: Entscheide, ob der Python-Code RELEVANT für die Erstellung, Routing, Generierung oder Konfiguration "
            "von gedruckten Schaltungen (PCB), SKiDL-Subcircuits, KiCad-Automatisierung oder Freerouting ist.\n\n"
            "FEW-SHOT BEISPIELE:\n"
            "1. POSITIV (RELEVANT):\n"
            "   - SKiDL Schaltungs-Code: Scripte mit @subcircuit, Part(), Net(), generate_netlist().\n"
            "   - KiCad Action-Plugins: Scripting-Code für PCB-Layout, DRC-Checks, Track-Generierung.\n"
            "   - Hardware-Berechnungen: Filter-Design, Trace-Breiten-Rechner, Pinout-Generatoren.\n\n"
            "2. NEGATIV (UNRELEVANT - sofort 'SKIP'):\n"
            "   - GUI-Utilities: wxPython, PyQt, Tkinter, Web-Server, UI-Dialoge, Event-Handler.\n"
            "   - Bild & Vektor: SVG-Export-Tools, PNG/PDF-Converter, Plotter-Grafik-Frameworks.\n"
            "   - Tests & Mocks: pytest, unittest, Fixtures, Mock-Klassen, Test-Runner.\n"
            "   - Build & Setup: setup.py, CI/CD pipelines, Release-Skripte.\n\n"
            "ANWEISUNG:\n"
            "Wenn der Code UNRELEVANT ist, antworte AUSSCHLIESSLICH mit dem einzelnen Wort:\n"
            "SKIP\n\n"
            "Wenn der Code RELEVANT ist, bereite ihn als strukturiertes Markdown mit Code-Block auf:\n"
            "## Python Hardware/SKiDL Modul: [Name]\n"
            "- **Dateipfad:** `[Pfad]`\n"
            "- **Zweck:** [Kurze Beschreibung]\n\n"
            "```python\n"
            "[Relevanter Code-Ausschnitt]\n"
            "```"
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
                return "SKIDL_SUBCIRCUIT", "SKIP"

            return "SKIDL_SUBCIRCUIT", generated_md

        except Exception as e:
            return "SKIDL_SUBCIRCUIT", "SKIP"
