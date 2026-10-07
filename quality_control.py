import ast
import re
from typing import Tuple

class QualityControl:
    @staticmethod
    def validate(tag: str, markdown_content: str, rel_path: str = "") -> Tuple[bool, str]:
        if not markdown_content or markdown_content.strip() in ["SKIP", ""]:
            return False, "Inhalt ist leer oder auf SKIP gesetzt"

        # 1. Strikter Dateinamen- / Pfad-Check (Build-Skripte & Doku abfangen)
        forbidden_files = ["cmakelists.txt", "makefile", "kbuild", "package-lock.json"]
        if any(ff in rel_path.lower() for ff in forbidden_files):
            return False, f"Ungültige Quelldatei für Schaltungssynthese ({rel_path})"

        # 2. Prüfe FreeRouting-Regeln
        if tag == "DESIGN_RULE":
            has_dsn_rules = any(kw in markdown_content.lower() for kw in [
                "clearance", "trace_width", "via_dia", "netclass", "(rule", "pair_gap", "circuit"
            ])
            if not has_dsn_rules:
                return False, "FreeRouting-Regeln enthalten keine strukturierten DSN/NetClass-Parameter (nur Fließtext/Prosa)"

        # 3. Extrahiere Python-Codeblöcke für SKiDL-Prüfung
        code_blocks = re.findall(r"```python(.*?)```", markdown_content, re.DOTALL)
        if tag == "SKIDL_SUBCIRCUIT" and not code_blocks:
            return False, "Kein ```python Code-Block im SKiDL-Subcircuit vorhanden"

        for code in code_blocks:
            code_str = code.strip()

            # A. Python-Syntaxprüfung via AST
            try:
                ast.parse(code_str)
            except SyntaxError as e:
                return False, f"Python SyntaxError in SKiDL-Code: {e.msg} (Zeile {e.lineno})"

            # B. Erfundene KiCad-Bibliotheken
            forbidden_libs = [
                r"Part\s*\(\s*['\"]Generic['\"]", 
                r"Part\s*\(\s*['\"]RESISTOR['\"]", 
                r"Part\s*\(\s*['\"]CAN['\"]",
                r"Part\s*\(\s*['\"]POWER['\"]"
            ]
            for lib_pattern in forbidden_libs:
                if re.search(lib_pattern, code_str, re.IGNORECASE):
                    return False, f"Erfundene KiCad-Bibliothek entdeckt ({lib_pattern})"

            # C. Invalide SKiDL-Syntax & Halluzinierte Methoden
            if re.search(r"\bPin\s*\(\s*\d", code_str):
                return False, "Invalide Pin-Syntax mit Zahl/Einheit (z.B. Pin(3.3V))"

            if ".connect(" in code_str:
                return False, "Invalide SKiDL-Syntax: .connect() verwendet (nutze += Operator)"

            if re.search(r"\.(set_param|set_routable|set_default_value)\b", code_str):
                return False, "Halluzinierte SKiDL-Methode entdeckt (.set_param / .set_routable)"

            if re.search(r"\w+\[['\"]\w+['\"]\]\s*=\s*", code_str):
                return False, "Invalide Zuweisung an Pin-Dictionary (nutze += Operator)"

            # D. Schaltungstechnische Plausibilitätsprüfungen (Elektrische Fehler)
            # D1: VCC/GND direkt an Daten- oder Taktleitungen
            if re.search(r"(vcc|gnd)\s*\+=\s*.*(eth_mdc|eth_mdio|sd_clk|sd_cmd|spi_miso|spi_mosi)", code_str, re.IGNORECASE):
                return False, "Elektrischer Kurzschluss: VCC/GND direkt mit Daten/Taktleitung verbunden"

            # D2: Relais direkt an MCU-Pin ohne Transistor
            if "Relay" in code_str and not any(q in code_str for q in ["Q_NPN", "Q_NMOS", "2N7002", "BC817", "ULN2003", "Transistor"]):
                return False, "Fehlender Relais-Treiber: Relaisspule ohne Ansteuertransistor/Treiber definiert"

            # D3: Relais ohne Freilaufdiode
            if "Relay" in code_str and not any(d in code_str for d in ["D_Flyback", "D", "1N4148", "Diode"]):
                return False, "Gefährliche Relais-Schaltung: Induktive Last ohne Freilaufdiode"

        return True, "Validierung erfolgreich"
