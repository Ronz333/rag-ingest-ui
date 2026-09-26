import os
import re
from .base_processor import BaseProcessor


class OSHWCircuitProcessor(BaseProcessor):

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

    base_system_prompt = (
        "Du bist ein Senior PCB Electronics Architect und SKiDL-Code-Synthesizer.\n"
        "Deine Aufgabe ist es, aus dem Schaltplan ein wiederverwendbares, hochpräzises SKiDL Entwurfsmuster (Subcircuit) zu synthetisieren.\n\n"
        "STRENGE KICAD SYMBOL- & FOOTPRINT REGELN (HALLUZINATIONS-SCHUTZ):\n"
        "1. KICAD SYMBOL-BIBLIOTHEKEN:\n"
        "   Verwende ausschließlich korrekte KiCad-Symbolkategorien!\n"
        "   - Widerstände/Kondensatoren/Dioden: Part('Device', 'R'), Part('Device', 'C'), Part('Device', 'D_TVS')\n"
        "   - Ethernet-Chips: Part('Interface_Ethernet', 'LAN8710A')\n"
        "   - Spannungswandler: Part('Regulator_Linear', 'SY8089AAAC') oder Part('Regulator_Switching', ...)\n"
        "   VERBOTEN: Nutze NIEMALS Footprint-Namen wie 'Resistor_SMD' als Symbolbibliothek!\n\n"
        "2. ZWINGENDE FOOTPRINT-ZUWEISUNG:\n"
        "   Jedes Bauteil MUSS ein explizites 'footprint=' Attribut enthalten (z. B. footprint='Resistor_SMD:R_0603_1608Metric').\n\n"
        "3. NETZLISTEN-VALIDITÄT:\n"
        "   Der Code muss so geschrieben sein, dass skidl.generate_netlist() fehlerfrei ausgeführt werden kann.\n\n"
        "FORMAT-VORGABE:\n"
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

    user_message = (
        f"Schaltplan: {filename}\nPfad:"
        f" {file_path}\n\nInhalt:\n{raw_content[:max_embed_chars]}"
    )

    max_attempts = 3
    last_error = ""

    for attempt in range(1, max_attempts + 1):
      system_prompt = base_system_prompt
      if last_error:
        system_prompt += (
            f"\n\n⚠️ KORREKTUR-AUFFORDERUNG (VERSUCH {attempt}/{max_attempts}):\n"
            "Dein vorheriger SKiDL-Code-Entwurf schlug bei skidl.generate_netlist() fehl:\n"
            f"-> {last_error}\n\n"
            "Bitte korrigiere die Bibliotheksnamen und Pins!"
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

        is_valid, err_msg = QualityControl.validate_skidl_runtime(generated_md)
        if is_valid:
          return "SKIDL_SUBCIRCUIT", generated_md

        last_error = err_msg
      except Exception as e:
        last_error = str(e)

    return "SKIDL_SUBCIRCUIT", "SKIP"


OshwCircuitProcessor = OSHWCircuitProcessor
