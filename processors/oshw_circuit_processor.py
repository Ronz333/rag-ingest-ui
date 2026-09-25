import os
import re
from .base_processor import BaseProcessor


class OSHWCircuitProcessor(BaseProcessor):
    """
    Processor für OSHW-Schaltpläne (.sch, .kicad_sch, .pdf).
    Generiert strukturierte SKiDL-Teilschaltungen mit ZWINGENDEN KiCad Footprint-Zuordnungen.
    Vergibt den Payload-Tag 'SKIDL_SUBCIRCUIT'.
    """

    def __init__(self):
        super().__init__()
        self.category_key = "⚡ PCB & Hardware Design"
        self.supported_extensions = [".sch", ".kicad_sch", ".pdf"]

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

        # Blacklist Filter-Check
        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern in fp_clean:
                    return "SKIDL_SUBCIRCUIT", "SKIP"

        # Strikte Dateiendungs-Ausschlüsse
        if ext in [".kicad_pcb", ".pro", ".kicad_pro", ".txt", ".ninja", ".o", ".obj", ".elf", ".log", ".cmakecache.txt"]:
            return "SKIDL_SUBCIRCUIT", "SKIP"

        system_prompt = (
            "Du bist ein Senior PCB Electronics Architect und SKiDL-Code-Synthesizer.\n"
            "Deine Aufgabe ist es, aus dem Schaltplan/BOM/Dokument ein wiederverwendbares, hochpräzises SKiDL Entwurfsmuster (Subcircuit) zu extrahieren.\n\n"
            "WICHTIGE ANFORDERUNGEN FÜR FREEROUTING & KICAD WORKFLOW:\n"
            "1. ZWINGENDE FOOTPRINT-ZUWEISUNG:\n"
            "   Jedes Bauteil MUSS ein explizites 'footprint=' Attribut enthalten (z. B. footprint='Package_TO_SOT_SMD:SOT-23-5' oder part.footprint = 'Resistor_SMD:R_0603_1608Metric').\n"
            "   SKiDL benötigt diese Angaben zwingend, um ein valides Netzlisten- und Specctra .dsn-Format für Freerouting zu generieren!\n"
            "2. SCHALTUNGSKAPSELUNG:\n"
            "   Verwende stets den @subcircuit Dekorator für modulare Funktionsblöcke (z.B. Buck Converter, LDO, Ethernet PHY, ESD Protection).\n"
            "3. DEVICE-BIBLIOTHEKEN:\n"
            "   Nutze für Passivbauteile die KiCad Symbol-Bibliothek 'Device' (Part('Device', 'R', ...), Part('Device', 'C', ...), Part('Device', 'D_TVS', ...)).\n\n"
            "GIB DEINE ANTWORT AUSSCHLIESSLICH IM FOLGENDEN STRUKTURIERTEN MARKDOWN-FORMAT AUS:\n\n"
            "## 1. Entwurfsmuster / Teilschaltung\n"
            "- **Name & Funktion:** [z. B. Buck Converter 5V zu 3.3V]\n"
            "- **Kurzbeschreibung:** [Funktionsweise & Spezifikationen]\n\n"
            "## 2. Vollständiger SKiDL Python-Block\n"
            "```python\n"
            "from skidl import *\n\n"
            "@subcircuit\n"
            "def my_subcircuit(v_in, v_out, gnd):\n"
            "    # SKiDL Code mit expliziten Footprints\n"
            "```\n\n"
            "## 3. KiCad Footprint-Zuordnungen\n"
            "| Bauteil | Ref / Typ | KiCad Footprint Library & Name | Pinning / Bemerkung |\n"
            "|---|---|---|---|\n"
        )

        user_message = f"Schaltplan/Dokument: {filename}\nPfad: {file_path}\n\nInhalt:\n{raw_content[:max_embed_chars]}"

        try:
            response = ollama_client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options=llm_options,
            )
            return "SKIDL_SUBCIRCUIT", response["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(
                f"Fehler in OSHWCircuitProcessor ({file_path}): {str(e)}"
            )
