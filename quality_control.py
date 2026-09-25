import ast
import io
import re
import sys


class QualityControl:
  """Gatekeeper-Modul: Prüft SKiDL-, KiCad- und Freerouting-Chunks zur Laufzeit."""

  @staticmethod
  def validate(
      category_tag: str, processed_md: str, rel_path: str
  ) -> tuple[bool, str]:
    if not processed_md or not processed_md.strip():
      return False, "Chunk ist leer."

    if len(processed_md) < 50:
      return False, "Inhalt zu kurz (< 50 Zeichen)."

    if category_tag == "SKIDL_SUBCIRCUIT":
      return QualityControl.validate_skidl_runtime(processed_md)

    elif category_tag == "DESIGN_RULE":
      return QualityControl.validate_design_rules(processed_md)

    elif category_tag == "DATASHEET_PINOUT":
      return QualityControl._validate_pinout(processed_md)

    return True, "OK"

  # --- 1. SKIDL LAUFZEIT-CHECK ---
  @staticmethod
  def validate_skidl_runtime(md_text: str) -> tuple[bool, str]:
    code_blocks = re.findall(r"```python(.*?)```", md_text, re.DOTALL)
    if not code_blocks:
      return False, "Kein ```python Codeblock im SKiDL-Markdown gefunden."

    python_code = code_blocks[0].strip()

    try:
      ast.parse(python_code)
    except SyntaxError as e:
      return False, f"Python SyntaxError in Zeile {e.lineno}: {e.msg}"

    has_footprint = (
        "footprint=" in python_code
        or ".footprint =" in python_code
        or ".footprint=" in python_code
    )
    if not has_footprint:
      return (
          False,
          "SKiDL-Code enthält keine expliziten Footprint-Zuweisungen"
          " (footprint='...').",
      )

    exec_scope = {}
    wrapper_code = f"""
from skidl import *
default_circuit.reset()
{python_code}
"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()

    try:
      exec(wrapper_code, exec_scope)
    except Exception as ex:
      return False, f"SKiDL Laufzeit-Fehler ({type(ex).__name__}): {str(ex)}"
    finally:
      sys.stdout, sys.stderr = old_stdout, old_stderr

    return True, "OK"

  # --- 2. FREEROUTING & KICAD DESIGN-RULES CHECK ---
  @staticmethod
  def validate_design_rules(md_text: str) -> tuple[bool, str]:
    # 2a. Sicherstellen, dass keine ungewollte Trace-Geometrie enthalten ist
    forbidden_terms = ["(path ", "(wire ", "(placement "]
    for term in forbidden_terms:
      if term in md_text:
        return (
            False,
            f"Verbotene Trace-Geometrie '{term}' im DRC/DSN-Chunk enthalten.",
        )

    # 2b. Lisp / S-Expression Codeblock extrahieren (.dsn / .rules / .kicad_pcb)
    lisp_blocks = re.findall(r"```(?:lisp|text)(.*?)```", md_text, re.DOTALL)
    if lisp_blocks:
      lisp_code = lisp_blocks[0].strip()

      # S-Expression Klammer-Symmetrie prüfen
      open_brackets = lisp_code.count("(")
      close_brackets = lisp_code.count(")")
      if open_brackets != close_brackets:
        return (
            False,
            f"S-Expression Klammerfehler: {open_brackets} öffnende vs."
            f" {close_brackets} schließende Klammern.",
        )

      # DSN-Spezifische Schlüsselwörter prüfen
      if "(parser" in lisp_code or "(structure" in lisp_code:
        required_keywords = ["rule", "clearance"]
        missing = [kw for kw in required_keywords if kw not in lisp_code.lower()]
        if missing:
          return (
              False,
              f"Specctra DSN-Regeln unvollständig. Fehlende Schlüsselwörter:"
              f" {missing}",
          )

      # KiCad DRC-Spezifische Schlüsselwörter prüfen
      if "(kicad_pcb" in lisp_code or "(setup" in lisp_code:
        if not any(
            kw in lisp_code.lower()
            for kw in ["netclass", "clearance", "trace_min", "via_size"]
        ):
          return (
              False,
              "KiCad DRC-Chunk enthält keine gültigen Netclass- oder"
              " Clearance-Regeln.",
          )

    return True, "OK"

  # --- 3. PINOUT-TABLE CHECK ---
  @staticmethod
  def _validate_pinout(md_text: str) -> tuple[bool, str]:
    if "| Pin Number |" not in md_text and "| Pin |" not in md_text:
      return False, "Keine valide Markdown-Pin-Tabelle gefunden."

    table_lines = [
        line
        for line in md_text.splitlines()
        if line.startswith("|") and "---" not in line
    ]
    if len(table_lines) < 2:
      return False, "Pin-Tabelle enthält keine Datenzeilen."

    return True, "OK"
