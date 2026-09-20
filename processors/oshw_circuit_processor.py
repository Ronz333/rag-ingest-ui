"""
OSHW CIRCUIT & SUBCIRCUIT PROCESSOR
-----------------------------------
Verarbeitet Open-Source Hardware Netzlisten und KiCad-Schaltpläne (.net, .xml, .kicad_sch, .sch).
Extrahiert funktional zusammenhängende Bauteilgruppen (z. B. Spannungsregler, Quarz-Oszillatoren, 
USB-Interfaces, Sensorschnittstellen) und synthetisiert daraus modulare SKiDL-Sub-Circuits.
"""

import os
import re
from typing import Tuple, Optional, List
from .base_processor import BaseProcessor


class OshwCircuitProcessor(BaseProcessor):
    category_key = "⚡ PCB & Hardware Design"
    collection_name = "pcb_knowledge_base"
    supported_extensions = {".net", ".xml", ".kicad_sch", ".sch"}

    GENERAL_IGNORED_PARTS = [
        "/.git/", "/build/", "/dist/", "/fixtures/", "/tests/", "/test/", "/benchmarks/"
    ]

    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        rel_lower = rel_path.lower()
        if any(p in rel_lower for p in self.GENERAL_IGNORED_PARTS):
            return False
        return ext in self.supported_extensions

    def parse(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_model: str, 
        ollama_client, 
        num_ctx, 
        max_code_len: int,
        selected_category: str = "",
        custom_filters: List[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        
        rel_lower = rel_path.lower()
        filename = os.path.basename(rel_path)
        custom_filters = custom_filters or []

        # 1. Allgemeine Ignorier-Pfade prüfen
        if any(p in rel_lower for p in self.GENERAL_IGNORED_PARTS):
            return "IGNORED", "SKIP"

        # 2. Dynamische Webpanel-Filter anwenden
        for rule in custom_filters:
            rule_lower = rule.lower()
            if rule_lower in rel_lower or rule_lower in filename.lower():
                return "IGNORED", "SKIP"

        ext = os.path.splitext(rel_path)[1].lower()
        cleaned_circuit_data = self._clean_netlist_content(raw_text, ext)

        if not cleaned_circuit_data.strip():
            return "IGNORED", "SKIP"

        options_dict = num_ctx if isinstance(num_ctx, dict) else {"num_ctx": num_ctx}

        enrichment_prompt = f"""Du bist ein Experte für Hardware-Synthese und SKiDL (Python for Circuit Design).
Analysiere die folgende Netzliste / den Schaltplan aus einem Open-Source Hardware Projekt:

DATEIPFAD: {rel_path}
SCHALTUNGS-DATEN:
{cleaned_circuit_data[:max_code_len]}

Erstelle daraus eine hochgradig strukturierte Wissenseinheit für ein PCB-Agenten-System auf Deutsch:

1. **Funktionale Sub-Circuits (Bausteine)**:
   - Identifiziere alle isolierbaren Schaltungsmodule (z. B. Power Supply / Step-Down, USB-C ESD Protection, MCU Crystal Oscillator, Sensor Interface, Status-LEDs).
   - Benenne die Kern-Bauteile (RefDes & Werte, z. B. U1: AMS1117-3.3, C1: 10uF, C2: 100nF).

2. **SKiDL Sub-Circuit Code-Synthese**:
   - Generiere für jedes identifizierte Modul einen voll funktionsfähigen, syntaktisch korrekten SKiDL Python-Code mit dem `@subcircuit` Decorator.
   - Definiere klare Eingangs- und Ausgangs-Nets (z. B. `v_in`, `v_out`, `gnd`).

3. **Exaktes Pin- & Signal-Mapping**:
   - Erstelle eine übersichtliche Markdown-Tabelle mit den Pin-Verbindungen für den Agenten.
"""

        try:
            response = ollama_client.chat(
                model=active_model,
                messages=[{'role': 'user', 'content': enrichment_prompt}],
                options=options_dict
            )
            processed_md = response['message']['content']
            return "OSHW_SUBCIRCUIT", processed_md
        except Exception:
            return "OSHW_SUBCIRCUIT", f"# OSHW Schaltungs-Referenz: {filename}\n\n```text\n{cleaned_circuit_data[:max_code_len]}\n```"

    def _clean_netlist_content(self, raw_text: str, ext: str) -> str:
        """
        Filtert grafische/Darstellungs-Informationen aus Netzlisten und Schaltplänen,
        sodass nur noch logische Komponenten, Nets und Verbindungen übrig bleiben.
        """
        if ext in {".kicad_sch", ".sch"}:
            # KiCad Schematic S-Expressionen: Entferne reine Zeichen- und Positionsbefehle (fill, stroke, uuid, at, effects)
            lines = []
            for line in raw_text.splitlines():
                line_str = line.strip()
                if any(kw in line_str for kw in ["(symbol", "(property", "(pin", "(instances", "(net", "(comp", "(value", "(footprint"]):
                    if not any(skip in line_str for skip in ["(at ", "(effects", "(uuid", "(stroke", "(fill"]):
                        lines.append(line)
                if len(lines) >= 400:
                    break
            return "\n".join(lines) if lines else raw_text[:4000]
        else:
            # Netzlisten (.net, .xml): Behalte Bauteile (components) und Verbindungen (nets)
            cleaned = re.sub(r'<tstamp>.*?</tstamp>', '', raw_text)
            cleaned = re.sub(r'\(sheetpath.*?\)', '', cleaned)
            lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
            return "\n".join(lines[:400])
