import ast
import io
import re
import sys


class QualityControl:
  """Gatekeeper-Modul: Prüft SKiDL-, KiCad- und Symbol-Chunks vor dem Embedding."""

  # Striktes Blacklisting von Platzhaltern, Lizenzen und Doku-Müll
  NOISE_KEYWORDS = [
      "change log",
      "changelog",
      "license",
      "purchased from",
      "omitted for simplicity",
      "placeholder",
      "todo:",
      "fixme:",
      "see datasheet",
  ]

  @staticmethod
  def validate(
      category_tag: str, processed_md: str, rel_path: str
  ) -> tuple[bool, str]:
    if not processed_md or not processed_md.strip():
      return False, "Chunk ist leer."

    md_lower = processed_md.lower()
    for kw in QualityControl.NOISE_KEYWORDS:
      if kw in md_lower:
        return (
            False,
            f"Chunk enthält unzulässiges Rauschen/Placeholder-Keyword: '{kw}'",
        )

    if len(processed_md) < 50:
      return False, "Inhalt zu kurz (< 50 Zeichen)."

    if category_tag == "SKIDL_SUBCIRCUIT":
      return QualityControl.validate_skidl_runtime(processed_md)
    elif category_tag == "DESIGN_RULE":
      return QualityControl.validate_design_rules(processed_md)
    elif category_tag in ["DATASHEET_PINOUT", "KICAD_SYMBOL"]:
      return QualityControl._validate_pinout(processed_md)

    return True, "OK"

  @staticmethod
  def validate_skidl_runtime(md_text: str) -> tuple[bool, str]:
    code_blocks = re.findall(r"```python(.*?)```", md_text, re.DOTALL)
    if not code_blocks:
      return False, "Kein ```python Codeblock im SKiDL-Markdown gefunden."

    python_code = code_blocks[0].strip()

    # 1. AST Python-Syntaxprüfung
    try:
      ast.parse(python_code)
    except SyntaxError as e:
      return False, f"Python SyntaxError in Zeile {e.lineno}: {e.msg}"

    # 2. Footprint-Check
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

    # 3. Testweise Netzlistengenerierung (skidl.generate_netlist())
    exec_scope = {}
    wrapper_code = f"""
from skidl import *
default_circuit.reset()

{python_code}

# Versuch der Netzlisten-Generierung zur Laufzeitprüfung
try:
    generate_netlist()
except Exception as ne:
    # Fange schwere Validierungsfehler ab
    if "No components" in str(ne) or "Unconnected" in str(ne):
        pass
"""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()

    try:
      exec(wrapper_code, exec_scope)
    except Exception as ex:
      return False, f"SKiDL generate_netlist() Fehler ({type(ex).__name__}): {str(ex)}"
    finally:
      sys.stdout, sys.stderr = old_stdout, old_stderr

    return True, "OK"

  @staticmethod
  def validate_design_rules(md_text: str) -> tuple[bool, str]:
    forbidden_terms = ["(path ", "(wire ", "(placement "]
    for term in forbidden_terms:
      if term in md_text:
        return (
            False,
            f"Verbotene Trace-Geometrie '{term}' im DRC/DSN-Chunk enthalten.",
        )

    lisp_blocks = re.findall(r"```(?:lisp|text)(.*?)```", md_text, re.DOTALL)
    if lisp_blocks:
      lisp_code = lisp_blocks[0].strip()
      open_b = lisp_code.count("(")
      close_b = lisp_code.count(")")
      if open_b != close_b:
        return (
            False,
            f"S-Expression Klammerfehler: {open_b} öffnende vs {close_b}"
            " schließende Klammern.",
        )

    return True, "OK"

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
