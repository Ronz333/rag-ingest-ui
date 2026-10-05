import os
import re
import json
from .base_processor import BaseProcessor


class OSHWCircuitProcessor(BaseProcessor):
    """
    Processor für OSHW-Schaltpläne (.sch, .kicad_sch).
    Nützt Clef-27B für schnelles Gating und prüft SKiDL-Code via QualityControl (skidl.generate_netlist()).
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
        clef_model_name: str,
        ollama_client,
        llm_options: dict,
        max_embed_chars: int,
        custom_filters: list = None,
    ) -> tuple[str, str]:
        from quality_control import QualityControl

        filename = os.path.basename(file_path)

        if custom_filters:
            fp_clean = file_path.replace("\\", "/")
            for pattern in custom_filters:
                if pattern and pattern.strip() and pattern.strip() in fp_clean:
                    if pattern.strip().startswith("/") or pattern.strip().endswith("/"):
                        return "SKIDL_SUBCIRCUIT", "SKIP"

        # PHASE A1: CLEF-27B DECISION GATE
        clef_prompt = (
            "Du bist ein Hardware-Decision-Head (Cloudflare Clef Gate).\n"
            "Pruefe, ob der Schaltplan gueltige Bauteile, Netze oder Schaltungstopologien enthaelt.\n\n"
            "ANTWORTE AUSSCHLIESSLICH IM JSON-FORMAT:\n"
            '{"is_hardware_relevant": true, "category_tag": "SKIDL_SUBCIRCUIT"}'
        )

        try:
            clef_res = ollama_client.chat(
                model=clef_model_name,
                messages=[
                    {"role": "system", "content": clef_prompt},
                    {"role": "user", "content": f"Schaltplan: {filename}\nInhalt:\n{raw_content[:3000]}"}
                ],
                format="json",
                options={"temperature": 0.0}
            )
            clef_json = json.loads(clef_res["message"]["content"].strip())
            if not clef_json.get("is_hardware_relevant", False):
                return "SKIDL_SUBCIRCUIT", "SKIP"
        except Exception:
            pass

        # PHASE A2: SKIDL SYNTHESE MIT QUALITY CONTROL LOOP
        base_system_prompt = (
            "Du bist ein deterministischer SKiDL-Code-Synthesizer für ein autonomes PCB-Generierungssystem.\n\n"
            "VERBOTENE PROSA (STRENGSTENS UNTERSAGT):\n"
            "Keine Einleitungssätze! Beginne sofort mit den Markdown-Sektionen.\n\n"
            "FORMAT-VORGABE:\n"
            "## 1. Schaltungs-Spezifikation\n"
            "- **Modul-Name:** [Name]\n"
            "- **Haupt-ICs / Bauteile:** [Modellnummern]\n\n"
            "## 2. Ausführbarer SKiDL Python Block\n"
            "```python\n"
            "from skidl import *\n\n"
            "@subcircuit\n"
            "def circuit_module(vcc, gnd, io_nets):\n"
            "    # Echter, lauffähiger SKiDL Code\n"
            "    pass\n"
            "```\n"
        )

        user_message = f"Schaltplan: {filename}\nPfad: {file_path}\n\nInhalt:\n{raw_content[:max_embed_chars]}"

        max_attempts = 3
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            system_prompt = base_system_prompt
            if last_error:
                system_prompt += (
                    f"\n\n⚠️ SKIDL COMPILER FEHLER (VERSUCH {attempt}/{max_attempts}):\n"
                    f"Der generierte Code schlug bei skidl.generate_netlist() fehl:\n"
                    f"-> {last_error}\n\n"
                    f"Korrigiere den Code! Verwende gültige Pin-Namen und Bibliotheken."
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

                if generated_md == "SKIP" or generated_md.startswith("SKIP"):
                    return "SKIDL_SUBCIRCUIT", "SKIP"

                cleaned_md = re.sub(r'^(?:RELEVANT[\:\s]*)+', '', generated_md, flags=re.IGNORECASE).strip()

                is_valid, err_msg = QualityControl.validate_skidl_runtime(cleaned_md)
                if is_valid:
                    return "SKIDL_SUBCIRCUIT", cleaned_md

                last_error = err_msg

            except Exception as e:
                last_error = str(e)

        return "SKIDL_SUBCIRCUIT", "SKIP"


OshwCircuitProcessor = OSHWCircuitProcessor
