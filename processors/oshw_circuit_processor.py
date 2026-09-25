import ast
import re
import sys
import io


class QualityControl:
    """
    Gatekeeper-Modul: Prüft synthetisierte Markdown-Chunks und führt 
    SKiDL-Code zur Laufzeit aus, um Ausführungsfehler und Pin-Exceptions zu fangen.
    """

    @staticmethod
    def validate(category_tag: str, processed_md: str, rel_path: str) -> tuple[bool, str]:
        if not processed_md or not processed_md.strip():
            return False, "Empty Chunk (Inhalt ist leer)"

        if len(processed_md) < 50:
            return False, "Inhalt zu kurz (< 50 Zeichen)"

        if category_tag == "SKIDL_SUBCIRCUIT":
            return QualityControl.validate_skidl_runtime(processed_md)

        elif category_tag == "DATASHEET_PINOUT":
            return QualityControl._validate_pinout(processed_md)

        elif category_tag == "DESIGN_RULE":
            return QualityControl._validate_design_rules(processed_md)

        return True, "OK"

    @staticmethod
    def validate_skidl_runtime(md_text: str) -> tuple[bool, str]:
        # 1. Codeblock extrahieren
        code_blocks = re.findall(r"```python(.*?)```", md_text, re.DOTALL)
        if not code_blocks:
            return False, "Kein ```python Codeblock im SKiDL-Markdown gefunden."

        python_code = code_blocks[0].strip()

        # 2. Statische AST-Syntaxprüfung
        try:
            ast.parse(python_code)
        except SyntaxError as e:
            return False, f"Python SyntaxError in Zeile {e.lineno}: {e.msg}"

        # 3. Zwingende Footprint-Zuweisung prüfen (Freerouting / DSN Anforderung)
        has_footprint = "footprint=" in python_code or ".footprint =" in python_code or ".footprint=" in python_code
        if not has_footprint:
            return False, "SKiDL-Code enthält keine expliziten Footprint-Zuweisungen (footprint='...')."

        # 4. Der Königsweg: Echte SKiDL Laufzeit-Ausführung im isolierten Scope
        exec_scope = {}
        wrapper_code = f"""
from skidl import *
default_circuit.reset() # SKiDL Zustand zurücksetzen

{python_code}
"""
        # stdout/stderr abfangen, um Konsole während Ingest sauber zu halten
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            exec(wrapper_code, exec_scope)
        except Exception as ex:
            error_type = type(ex).__name__
            error_msg = str(ex)
            return False, f"SKiDL Laufzeit-Fehler ({error_type}): {error_msg}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return True, "OK"

    @staticmethod
    def _validate_pinout(md_text: str) -> tuple[bool, str]:
        if "| Pin Number |" not in md_text and "| Pin |" not in md_text:
            return False, "Keine valide Markdown-Pin-Tabelle gefunden."

        table_lines = [
            line for line in md_text.splitlines() 
            if line.startswith("|") and "---" not in line
        ]
        if len(table_lines) < 2:
            return False, "Pin-Tabelle enthält keine Datenzeilen."

        return True, "OK"

    @staticmethod
    def _validate_design_rules(md_text: str) -> tuple[bool, str]:
        forbidden_terms = ["(path ", "(wire ", "(placement "]
        for term in forbidden_terms:
            if term in md_text:
                return False, f"Verbotene Trace-Geometrie '{term}' im DRC/DSN-Chunk enthalten."

        return True, "OK"
