import json
import re
from typing import Tuple, Dict, Any, Optional

class OSHWCircuitProcessor:
    def __init__(self):
        self.category_name = "⚡ PCB & Hardware Design"

    def evaluate_gate(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_clef_model: str, 
        ollama_client: Any, 
        llm_options: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Phase A1: Schnellprüfung via Clef-27B (Batch-kompatibel)
        """
        clef_opts = llm_options.get("clef_options", {
            "num_ctx": 4096,
            "temperature": 0.0
        })

        gate_prompt = (
            f"You are a strict hardware engineering decision gate.\n"
            f"Analyze the following file to determine if it contains hardware schematics, pinouts, IC datasheets, C/C++ board drivers, or PCB routing rules.\n\n"
            f"File Path: {rel_path}\n"
            f"Content Preview:\n"
            f"===\n{raw_text[:3000]}\n===\n\n"
            f"Respond with a JSON object ONLY:\n"
            f"{{\n"
            f'  "is_relevant": true,\n'
            f'  "category_tag": "SKIDL_SUBCIRCUIT",\n'
            f'  "reason": "Brief explanation"\n'
            f"}}\n"
            f'Set is_relevant to false if the file is build script, binary, or unrelated boilerplate. Valid category_tags: "SKIDL_SUBCIRCUIT", "KICAD_SYMBOL", "DESIGN_RULE", "GENERAL".'
        )

        try:
            gate_res = ollama_client.chat(
                model=active_clef_model,
                messages=[{"role": "user", "content": gate_prompt}],
                options=clef_opts
            )
            gate_content = gate_res['message']['content'].strip()
            
            json_match = re.search(r'\{.*\}', gate_content, re.DOTALL)
            if json_match:
                gate_data = json.loads(json_match.group(0))
                is_rel = gate_data.get("is_relevant", True)
                cat_tag = gate_data.get("category_tag", "SKIDL_SUBCIRCUIT")
                return is_rel, cat_tag
            return True, "SKIDL_SUBCIRCUIT"

        except Exception as gate_err:
            print(f"⚠️ Decision Gate Fehler bei {rel_path}: {gate_err}")
            return True, "SKIDL_SUBCIRCUIT"

    def synthesize_code(
        self, 
        rel_path: str, 
        raw_text: str, 
        initial_tag: str, 
        active_model: str, 
        ollama_client: Any, 
        llm_options: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Phase A2: SKiDL Code-Synthese via Qwen3-Coder-30B (Batch-kompatibel)
        """
        synthesis_opts = llm_options.get("synthesis_options", {
            "num_ctx": llm_options.get("num_ctx", 32768),
            "temperature": 0.0
        })

        synthesis_prompt = (
            f"You are an expert Hardware & Electronics Design Automation Assistant.\n"
            f"Synthesize the provided source file into structured Markdown documentation including an executable SKiDL (Python) circuit representation or PCB constraints.\n\n"
            f"Requirements:\n"
            f"1. Pinout & Hardware Mapping Table (Signals, Pins, Buses, Voltage Levels)\n"
            f"2. Valid SKiDL Python Code Block (`from skidl import * ...`) representing the component/circuit connections\n"
            f"3. Electrical Parameters & Constraints\n\n"
            f"File Path: {rel_path}\n"
            f"Source Code / Content:\n"
            f"===\n{raw_text[:25000]}\n===\n\n"
            f"Generate clean Markdown documentation:"
        )

        try:
            synth_res = ollama_client.chat(
                model=active_model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                options=synthesis_opts
            )
            markdown_out = synth_res['message']['content'].strip()
            return initial_tag, markdown_out

        except Exception as synth_err:
            print(f"❌ Synthese-Fehler bei {rel_path}: {synth_err}")
            return initial_tag, "SKIP"

    def parse(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_model: str, 
        active_clef_model: str, 
        ollama_client: Any, 
        llm_options: Dict[str, Any], 
        max_embed_chars: int = 50000,
        custom_filters: Optional[list] = None
    ) -> Tuple[str, str]:
        """Rückwärtskompatible Parse-Methode für Einzelaufrufe"""
        is_rel, tag = self.evaluate_gate(rel_path, raw_text, active_clef_model, ollama_client, llm_options)
        if not is_rel:
            return "GENERAL", "SKIP"
        return self.synthesize_code(rel_path, raw_text, tag, active_model, ollama_client, llm_options)
