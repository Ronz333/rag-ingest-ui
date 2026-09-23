import os
import re
from processors.base_processor import BaseProcessor


class OSHWCircuitProcessor(BaseProcessor):
    """
    Processor für OSHW-Schaltpläne. Generiert kanonische SKiDL Few-Shot Muster (Säule 2).
    """

    def __init__(self):
        super().__init__()
        self.category_key = "⚡ PCB & Hardware Design"
        self.supported_extensions = [".sch", ".kicad_sch", ".net"]

    def can_handle(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions

    def parse(self, file_path: str, raw_content: str, model_name: str, ollama_client, llm_options: dict, max_embed_chars: int, custom_filters: list = None) -> tuple[str, str]:
        # Filter-Prüfung
        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern in fp_clean:
                    return "OSHW_SUBCIRCUIT", "SKIP"

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".kicad_pcb", ".pro", ".kicad_pro", ".txt", ".ninja"]:
            return "OSHW_SUBCIRCUIT", "SKIP"

        system_prompt = (
            "Du bist ein Senior PCB Electronics Architect.\n"
            "Analysiere den Schaltplan/Netzliste und erstelle eine hochreine SKiDL-Referenzvorlage.\n\n"
            "REGELN FÜR DIE CODE-SYNTHESE:\n"
            "1. Kapselung: Erstelle für logische Blöcke (Power, MCU, PHY, ESD) wiederverwendbare Funktionen mit @subcircuit.\n"
            "2. Bibliotheken: Verwende für Passivbauteile ausschließlich 'Device' (z.B. Part('Device', 'R', ...), Part('Device', 'LED', ...)).\n"
            "3. Topologie: Achte auf korrekt in Reihe geschaltete LED-Vorwiderstände und antiparallel geschaltete TVS-Dioden.\n\n"
            "FORMAT-VORGABE:\n"
            "## 1. Funktionale Beschreibung\n"
            "- Kurze Stichpunkte\n\n"
            "## 2. SKiDL Sub-Circuit Code\n"
            "```python\n"
            "# Valider SKiDL Code\n"
            "```\n\n"
            "## 3. Signal-Mapping\n"
            "| Bauteil | Pin | Funktion | Net |\n"
        )

        user_message = f"Schaltplan: {file_path}\n\nInhalt:\n{raw_content[:max_embed_chars]}"

        try:
            response = ollama_client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                options=llm_options
            )
            return "OSHW_SUBCIRCUIT", response['message']['content'].strip()
        except Exception as e:
            raise RuntimeError(f"Fehler in OSHWCircuitProcessor ({file_path}): {str(e)}")
