import os
import re
from .base_processor import BaseProcessor


class OSHWCircuitProcessor(BaseProcessor):
    """
    Processor für OSHW-Schaltpläne (.sch, .kicad_sch, .pdf).
    Generiert SKiDL-Subcircuits mit einem 3-stufigen Selbstkorrektur-Loop bei Laufzeitfehlern.
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
        # Lazy Import zur Vermeidung von zirkulären Import-Abhängigkeiten
        from quality_control import QualityControl

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        # Blacklist Filter-Check
        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern in fp_clean:
                    return "SKIDL_SUBCIRCUIT", "SKIP"

        if ext in [".kicad_pcb", ".pro", ".kicad_pro", ".txt", ".ninja", ".o", ".obj", ".elf", ".log", ".cmakecache.txt"]:
            return "SKIDL_SUBCIRCUIT", "SKIP"

        base_system_prompt = (
            "Du bist ein Senior PCB Electronics Architect und SKiDL-Code-Synthesizer.\n"
            "Deine Aufgabe ist es, aus dem Schaltplan/BOM/Dokument ein wiederverwendbares, hochpräzises SKiDL Entwurfsmuster (Subcircuit) zu extrahieren.\n\n"
            "WICHTIGE ANFORDERUNGEN FÜR FREEROUTING & KICAD WORKFLOW:\n"
            "1. ZWINGENDE FOOTPRINT-ZUWEISUNG:\n"
            "   Jedes Bauteil MUSS ein explizites 'footprint=' Attribut enthalten (z. B. footprint='Package_TO_SOT_SMD:SOT-23-5' oder part.footprint = 'Resistor_SMD:R_0603_1608Metric').\n"
            "2. SCHALTUNGSKAPSELUNG:\n"
            "   Verwende stets den @subcircuit Dekorator für modulare Funktionsblöcke.\n"
            "3. DEVICE-BIBLIOTHEKEN:\n"
            "   Nutze für Passivbauteile ausschließlich 'Device' (Part('Device', 'R', ...), Part('Device', 'C', ...), Part('Device', 'D_TVS', ...)).\n\n"
            "FORMAT-VORGABE:\n"
            "## 1. Entwurfsmuster / Teilschaltung\n"
            "- **Name & Funktion:** [Name]\n\n"
            "## 2. Vollständiger SKiDL Python-Block\n"
            "```python\n"
            "from skidl import *\n\n"
            "@subcircuit\n"
            "def my_subcircuit(v_in, v_out, gnd):\n"
            "    # Code\n"
            "```\n\n"
            "## 3. KiCad Footprint-Zuordnungen\n"
            "| Bauteil | Ref / Typ | KiCad Footprint Library & Name | Pinning |\n"
        )

        user_message = f"Schaltplan/Dokument: {filename}\nPfad: {file_path}\n\nInhalt:\n{raw_content[:max_embed_chars]}"

        max_attempts = 3
        last_error = ""

        # --- SELBSTKORREKTUR-SCHLEIFE (UP TO 3 ATTEMPTS) ---
        for attempt in range(1, max_attempts + 1):
            system_prompt = base_system_prompt

            if last_error:
                system_prompt += (
                    f"\n\n⚠️ KORREKTUR-AUFFORDERUNG (VERSUCH {attempt}/{max_attempts}):\n"
                    f"Dein vorheriger SKiDL-Code-Entwurf schlug bei der SKiDL-Laufzeitprüfung fehl mit folgendem Fehler:\n"
                    f"-> {last_error}\n\n"
                    f"Bitte korrigiere den SKiDL-Code und stelle sicher, dass alle Bibliotheken und Pins korrekt deklariert sind!"
                )

            try:
                response = ollama_client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    options=llm_options,
                )
                generated_md = response["message"]["content"].strip()

                # Laufzeit-Validierung des generierten Chunks
                is_valid, err_msg = QualityControl.validate_skidl_runtime(generated_md)

                if is_valid:
                    return "SKIDL_SUBCIRCUIT", generated_md

                last_error = err_msg

            except Exception as e:
                last_error = str(e)

        # Wenn auch nach 3 Versuchen kein ausführbarer SKiDL-Code entstand: Verwerfen
        return "SKIDL_SUBCIRCUIT", "SKIP"


# Alias bereitstellen, damit alle Import-Schreibweisen abgedeckt sind
OshwCircuitProcessor = OSHWCircuitProcessor
