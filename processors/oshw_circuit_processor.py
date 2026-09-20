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
        custom_filters: Optional[List[str]] = None
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

Erstelle daraus eine hochgradig strukturierte Wissenseinheit für ein PCB-Agenten-System auf Deutsch.

WICHTIGE STRUKTUR- UND REGEL-VORGABEN FÜR DEN AGENTEN-CODE:
1. **KiCad Standard-Bibliotheken nutzen**: Verwende für `Part()` NIEMALS projektspezifische oder proprietäre Bibliotheksnamen (wie 'OLIMEX_RCL' oder '{filename}'). Nutze AUSSCHLIESSLICH offizielle KiCad-Standardbibliotheken wie 'Device', 'Regulator_Linear', 'Regulator_Switching', 'Diode', 'Switch', 'Interface_Ethernet', 'Connector', 'Power_Protection'.
2. **Standard SMD-Footprints deklarieren**: Jedes Bauteil MUSS nach Möglichkeit ein explizites `footprint='...'` Attribut enthalten (z. B. `footprint='Resistor_SMD:R_0603_1608Metric'`, `footprint='Capacitor_SMD:C_0603_1608Metric'`, `footprint='Package_TO_SOT_SMD:SOT-23-5'`).
3. **Modulare `@subcircuit`-Funktionen**: Erstelle für jeden isolierbaren Baustein eine eigene, saubere Python-Funktion mit `@subcircuit`.
4. **Header & Trennlinien**: Trenne Abschnitte strikt mit horizontalen Linien (`---`), damit das RAG-System die Chunks verlustfrei schneiden kann.

STRUKTUR DER ANTWORT:

## 1. Funktionale Sub-Circuits (Bausteine)
- Liste der Sub-Circuits mit Zweck und Hauptbauteilen.

---

## 2. SKiDL Sub-Circuit Code-Synthese
```python
from skidl import *

@subcircuit
def power_esd_protection(v_in, v_out, gnd):
    # Beispiel mit KiCad Standard-Libs und Footprints
    d1 = Part('Diode', 'TVS', footprint='Diode_SMD:D_SMA')
    ...
