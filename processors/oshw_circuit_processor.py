import os
import re
from .base_processor import BaseProcessor


class OSHWCircuitProcessor(BaseProcessor):
    """
    Processor für OSHW-Schaltpläne (.sch, .kicad_sch).
    Generiert SKiDL-Subcircuits, säubert Gate-Präfixe und prüft die Ausführbarkeit via QualityControl.
    """

    def __init__(self):
        super().__init__()
        self.category_key = "⚡ PCB & Hardware Design"
        self.supported_extensions = [".sch", ".kicad_sch"]

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
                    return "SKIDL_SUBCIRCUIT", "SKIP"

        phase_a_options = dict(llm_options)
        phase_a_options["temperature"] = 0.0

        base_system_prompt = (
            "Du bist ein hochspezialisierter Filter-Assistent und SKiDL-Code-Synthesizer für ein PCB-Automatisierungs-RAG.\n\n"
            "STUFE 1: SEMANTISCHE RELEVANZ-PRÜFUNG\n"
            "Entscheide, ob der vorliegende Datei-Inhalt RELEVANT für PCB-Design, Elektronik-Topologien, Schaltungsfunktionalität oder SKiDL ist.\n\n"
            "FEW-SHOT BEISPIELE:\n"
            "1. RELEVANT (Verarbeiten):\n"
            "   - Schaltpläne mit konkreten ICs, Widerständen, Kondensatoren, Bussen (I2C, SPI, Ethernet, USB) und Netzen.\n"
            "   - Wiederverwendbare Schaltungs-Topologien (z. B. Step-Down-Regler, Mikrokontroller-Grundbeschaltungen).\n"
            "2. UNRELEVANT (sofort mit 'SKIP' antworten):\n"
            "   - Leere Schaltpläne, unvollständige Fragmente, Doku-Skizzen ohne Bauteile.\n\n"
            "ANWEISUNG:\n"
            "- Wenn die Datei UNRELEVANT ist, antworte AUSSCHLIESSLICH mit dem einzelnen Wort: SKIP\n"
            "- Wenn die Datei RELEVANT ist, synthetisiere daraus ein SKiDL Entwurfsmuster.\n"
            "WICHTIG: Schreibe UNTER KEINEN UMSTÄNDEN das Wort 'RELEVANT' in deine Antwort! Beginne direkt mit der Markdown-Ausgabe.\n\n"
            "FORMAT-VORGABE FÜR RELEVANTE DATEIEN:\n"
            "## 1. Entwurfsmuster / Teilschaltung\n"
            "- **Name & Funktion:** [Name]\n\n"
            "## 2. Vollständiger SKiDL Python-Block\n"
            "```python\n"
            "from skidl import *\n\n"
            "@subcircuit\n"
            "def my_subcircuit(v_in, v_out, gnd):\n"
            "    # Code\n"
            "```\n"
        )

        user_message = f"Schaltplan: {filename}\nPfad: {file_path}\n\nInhalt:\n{raw_content[:max_embed_chars]}"

        max_attempts = 3
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            system_prompt = base_system_prompt
            if last_error:
                system_prompt += (
                    f"\n\n⚠️ KORREKTUR-AUFFORDERUNG (VERSUCH {attempt}/{max_attempts}):\n"
                    f"Dein vorheriger SKiDL-Code-Entwurf schlug bei skidl.generate_netlist() fehl:\n"
                    f"-> {last_error}\n\n"
                    f"Bitte korrigiere die Bibliotheksnamen und Pins!"
                )

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

                # Präfix-Säuberung
                cleaned_md = re.sub(r'^(?:RELEVANT[\:\s]*)+', '', generated_md, flags=re.IGNORECASE).strip()

                is_valid, err_msg = QualityControl.validate_skidl_runtime(cleaned_md)
                if is_valid:
                    return "SKIDL_SUBCIRCUIT", cleaned_md

                last_error = err_msg

            except Exception as e:
                last_error = str(e)

        return "SKIDL_SUBCIRCUIT", "SKIP"


OshwCircuitProcessor = OSHWCircuitProcessor
