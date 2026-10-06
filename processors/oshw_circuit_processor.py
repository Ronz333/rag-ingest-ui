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

        gate_prompt = f"""You are a strict hardware engineering decision gate.
Analyze the following file to determine if it contains hardware schematics, pinouts, IC datasheets, C/C++ board drivers, or PCB routing rules.

File Path: {rel_path}
Content Preview:
