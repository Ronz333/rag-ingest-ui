import json
import re
import time
from typing import Tuple, Dict, Any, Optional

class OSHWCircuitProcessor:
    supported_extensions = [
        ".py", ".kicad_sym", ".kicad_sch", ".kicad_pcb", 
        ".dsn", ".rules", ".cpp", ".hpp", ".h", ".c", ".ino"
    ]

    def __init__(self):
        self.category_name = "⚡ PCB & Hardware Design"
        self.supported_extensions = OSHWCircuitProcessor.supported_extensions

    def can_handle(self, rel_path: str = "", content: Optional[str] = None, file_path: Optional[str] = None, **kwargs) -> bool:
        target_path = (file_path or rel_path or "").lower()
        ext = "." + target_path.rsplit(".", 1)[-1] if "." in target_path else ""
        return ext in self.supported_extensions

    def _log(self, log_list: Optional[Any], text: str):
        if log_list is not None:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            log_list.append(f"[{timestamp}] {text}")
        else:
            print(text)

    def _llm_call(
        self, 
        ollama_client: Any, 
        primary_model: str, 
        fallback_model: str, 
        prompt: str, 
        options: Dict[str, Any], 
        log_list: Optional[Any] = None
    ) -> str:
        """
        Führt LLM-Aufruf mit dreistufigem Failover aus:
        1. primary_model via chat()
        2. primary_model via generate()
        3. fallback_model via chat() (falls primary_model in Ollama defekt/unsupported ist)
        """
        try:
            res = ollama_client.chat(
                model=primary_model,
                messages=[{"role": "user", "content": prompt}],
                options=options
            )
            return res['message']['content'].strip()
        except Exception as chat_err:
            err_str = str(chat_err)
            if "does not support chat" in err_str or "400" in err_str:
                try:
                    self._log(log_list, f"   ⚠️ Modell '{primary_model}' unterstützt kein 'chat()'. Wechsle zu 'generate()'...")
                    res = ollama_client.generate(
                        model=primary_model,
                        prompt=prompt,
                        options=options
                    )
                    return res['response'].strip()
                except Exception as gen_err:
                    gen_str = str(gen_err)
                    if ("does not support generate" in gen_str or "400" in gen_str) and fallback_model and fallback_model != primary_model:
                        self._log(log_list, f"   ❌ Modell '{primary_model}' in Ollama defekt (Status 400 auf chat & generate).")
                        self._log(log_list, f"   🔄 Wechsle für Decision Gate automatisch zum Fallback-Modell '{fallback_model}'...")
                        res = ollama_client.chat(
                            model=fallback_model,
                            messages=[{"role": "user", "content": prompt}],
                            options=options
                        )
                        return res['message']['content'].strip()
                    raise gen_err
            raise chat_err

    def evaluate_gate(
        self, 
        rel_path: str = "", 
        raw_text: str = "", 
        active_clef_model: str = "", 
        active_model: str = "",
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
        clef_opts = opts.get("clef_options", {"num_ctx": 4096, "temperature": 0.0})

        self._log(log_list, f"   🔍 [GATE-START] Sende Prompt an Model '{active_clef_model}' ({len(text_data)} Zeichen)...")

        gate_prompt = (
            f"You are a strict hardware engineering gatekeeper.\n"
            f"Determine if the file contains executable C/C++ board drivers, pin mappings, KiCad definitions, or routing rules.\n\n"
            f"File Path: {target_path}\n"
            f"Content Preview:\n"
            f"===\n{text_data[:3000]}\n===\n\n"
            f"Respond with a JSON object ONLY:\n"
            f"{{\n"
            f'  "is_relevant": true,\n'
            f'  "category_tag": "SKIDL_SUBCIRCUIT",\n'
            f'  "reason": "Brief explanation"\n'
            f"}}\n"
            f'Rules:\n'
            f'- Set is_relevant to false for build scripts, Makefiles, generic documentation, or non-hardware code.\n'
            f'- Valid category_tags: "SKIDL_SUBCIRCUIT", "KICAD_SYMBOL", "DESIGN_RULE". Do NOT use "GENERAL".'
        )

        try:
            start_t = time.time()
            gate_content = self._llm_call(
                ollama_client=ollama_client,
                primary_model=active_clef_model,
                fallback_model=active_model,
                prompt=gate_prompt,
                options=clef_opts,
                log_list=log_list
            )
            elapsed = time.time() - start_t
            self._log(log_list, f"   ⏱️ [GATE-OK] Antwort in {elapsed:.2f}s: {gate_content[:120]}...")
            
            json_match = re.search(r'\{.*\}', gate_content, re.DOTALL)
            if json_match:
                gate_data = json.loads(json_match.group(0))
                is_rel = gate_data.get("is_relevant", False)
                cat_tag = gate_data.get("category_tag", "SKIDL_SUBCIRCUIT")
                if cat_tag not in ["SKIDL_SUBCIRCUIT", "KICAD_SYMBOL", "DESIGN_RULE"]:
                    cat_tag = "SKIDL_SUBCIRCUIT"
                return is_rel, cat_tag
            return True, "SKIDL_SUBCIRCUIT"

        except Exception as gate_err:
            err_msg = f"{type(gate_err).__name__}: {str(gate_err)}"
            self._log(log_list, f"   ❌ [GATE-FEHLER] Ollama Aufruf fehlgeschlagen für {target_path}: {err_msg}")
            # Bei Modellabsturz Datei zur Sicherheit für Synthese zulassen
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
        synthesis_opts = opts.get("synthesis_options", {"num_ctx": opts.get("num_ctx", 32768), "temperature": 0.0})

        self._log(log_list, f"   🤖 [SYNTHESE-START] Sende Prompt an Model '{active_model}' ({len(text_data)} Zeichen)...")

        synthesis_prompt = (
            f"You are an expert Hardware & Electronics Design Automation Assistant.\n"
            f"Synthesize the provided source file into structured Markdown documentation with EXECUTABLE SKiDL (Python) circuit code.\n\n"
            f"CRITICAL HARDWARE & SKIDL STRICT RULES:\n"
            f"1. PYTHON SYNTAX: Variable names MUST be valid Python identifiers (e.g. use `vcc_3v3`, NEVER `3.3V`).\n"
            f"2. KICAD LIBRARIES: Use ONLY standard KiCad v6+ libraries (`Device`, `MCU_Espressif`, `Interface_CAN`, `Regulator_Linear`, `Relay`). FORBIDDEN: `Generic`, `RESISTOR`, `CAN`, `POWER`.\n"
            f"3. SKIDL CONNECTIONS: Connect pins to nets ONLY via the `+=` operator (e.g. `vcc += part['VCC']`). NEVER use `.connect()`, assignment `=`, or dictionary overrides `part['PIN'] = val`.\n"
            f"4. RELAYS: Relays MUST include an NPN/NMOS transistor (e.g. `Part('Device', 'Q_NPN')`) and a flyback diode (e.g. `Part('Device', 'D_Flyback')`) across the coil.\n"
            f"5. BUSES: Pull-up resistors MUST be instantiated as individual Part() instances per bus line. NEVER connect one resistor to multiple bus signals.\n"
            f"6. NO SHORT CIRCUITS: Never connect VCC or GND directly to signal, clock, or data nets.\n"
            f"7. FREEROUTING: If synthesizing routing rules, generate KiCad `(NetClass ...)` or Specctra DSN syntax `(clearance ...)`. Do NOT write prose text.\n\n"
            f"File Path: {target_path}\n"
            f"Source Content:\n"
            f"===\n{text_data[:25000]}\n===\n\n"
            f"Generate valid Markdown with Python code block (` ```python ... ``` `):"
        )

        try:
            start_t = time.time()
            markdown_out = self._llm_call(
                ollama_client=ollama_client,
                primary_model=active_model,
                fallback_model="",
                prompt=synthesis_prompt,
                options=synthesis_opts,
                log_list=log_list
            )
            elapsed = time.time() - start_t
            self._log(log_list, f"   ✅ [SYNTHESE-OK] Synthese abgeschlossen in {elapsed:.2f}s ({len(markdown_out)} Zeichen).")
            return initial_tag, markdown_out

        except Exception as synth_err:
            err_msg = f"{type(synth_err).__name__}: {str(synth_err)}"
            self._log(log_list, f"   ❌ [SYNTHESE-FEHLER] Ollama Aufruf ({active_model}) abgebrochen: {err_msg}")
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
            active_model=active_model,
            ollama_client=ollama_client, 
            llm_options=opts,
            log_list=log_list,
            **kwargs
        )
        if not is_rel or tag == "SKIP":
            return "SKIP", "SKIP"
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
