import json
import re
import time
from typing import Tuple, Dict, Any, Optional

class OSHWCircuitProcessor:
    supported_extensions = [
        ".py", ".kicad_sym", ".kicad_sch", ".kicad_pcb", 
        ".dsn", ".rules", ".json", ".yaml", ".yml", 
        ".txt", ".md", ".cpp", ".h", ".c", ".ino"
    ]

    def __init__(self):
        self.category_name = "⚡ PCB & Hardware Design"
        self.supported_extensions = OSHWCircuitProcessor.supported_extensions

    def can_handle(self, rel_path: str = "", content: Optional[str] = None, file_path: Optional[str] = None, **kwargs) -> bool:
        target_path = file_path or rel_path or ""
        ext = "." + target_path.rsplit(".", 1)[-1].lower() if "." in target_path else ""
        return ext in self.supported_extensions

    def _log(self, log_list: Optional[Any], text: str):
        if log_list is not None:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            log_list.append(f"[{timestamp}] {text}")
        else:
            print(text)

    def evaluate_gate(
        self, 
        rel_path: str = "", 
        raw_text: str = "", 
        active_clef_model: str = "", 
        ollama_client: Any = None, 
        llm_options: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        log_list: Optional[Any] = None,
        **kwargs
    ) -> Tuple[bool, str]:
        target_path = file_path or rel_path or ""
        text_data = content or raw_text or ""
        opts = llm_options or {}

        clef_opts = opts.get("clef_options", {
            "num_ctx": 4096,
            "temperature": 0.0
        })

        gate_prompt = (
            f"You are a strict hardware engineering decision gate.\n"
            f"Analyze the following file to determine if it contains hardware schematics, pinouts, IC datasheets, C/C++ board drivers, or PCB routing rules.\n\n"
            f"File Path: {target_path}\n"
            f"Content Preview:\n"
            f"===\n{text_data[:3000]}\n===\n\n"
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
            self._log(log_list, f"   ⚠️ Decision Gate Exception bei {target_path}: {gate_err}")
            return True, "SKIDL_SUBCIRCUIT"

    def synthesize_code(
        self, 
        rel_path: str = "", 
        raw_text: str = "", 
        initial_tag: str = "SKIDL_SUBCIRCUIT", 
        active_model: str = "", 
        ollama_client: Any = None, 
        llm_options: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        log_list: Optional[Any] = None,
        **kwargs
    ) -> Tuple[str, str]:
        target_path = file_path or rel_path or ""
        text_data = content or raw_text or ""
        opts = llm_options or {}

        synthesis_opts = opts.get("synthesis_options", {
            "num_ctx": opts.get("num_ctx", 32768),
            "temperature": 0.0
        })

        synthesis_prompt = (
            f"You are an expert Hardware & Electronics Design Automation Assistant.\n"
            f"Synthesize the provided source file into structured Markdown documentation including an executable SKiDL (Python) circuit representation or PCB constraints.\n\n"
            f"Requirements:\n"
            f"1. Pinout & Hardware Mapping Table (Signals, Pins, Buses, Voltage Levels)\n"
            f"2. Valid SKiDL Python Code Block (`from skidl import * ...`) representing the component/circuit connections\n"
            f"3. Electrical Parameters & Constraints\n\n"
            f"File Path: {target_path}\n"
            f"Source Code / Content:\n"
            f"===\n{text_data[:25000]}\n===\n\n"
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
            self._log(log_list, f"   ❌ Synthese-Fehler (Ollama Exception) bei {target_path}: {synth_err}")
            return initial_tag, "SKIP"

    def parse(
        self, 
        rel_path: str = "", 
        raw_text: str = "", 
        active_model: str = "", 
        active_clef_model: str = "", 
        ollama_client: Any = None, 
        llm_options: Optional[Dict[str, Any]] = None, 
        max_embed_chars: int = 50000,
        custom_filters: Optional[list] = None,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        log_list: Optional[Any] = None,
        **kwargs
    ) -> Tuple[str, str]:
        target_path = file_path or rel_path or ""
        text_data = content or raw_text or ""
        opts = llm_options or {}

        is_rel, tag = self.evaluate_gate(
            rel_path=target_path, 
            raw_text=text_data, 
            active_clef_model=active_clef_model, 
            ollama_client=ollama_client, 
            llm_options=opts,
            log_list=log_list,
            **kwargs
        )
        if not is_rel:
            return "GENERAL", "SKIP"
        return self.synthesize_code(
            rel_path=target_path, 
            raw_text=text_data, 
            initial_tag=tag, 
            active_model=active_model, 
            ollama_client=ollama_client, 
            llm_options=opts,
            log_list=log_list,
            **kwargs
        )

OshwCircuitProcessor = OSHWCircuitProcessor
