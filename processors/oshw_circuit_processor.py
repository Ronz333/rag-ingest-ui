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
        clef_opts = opts.get("clef_options", {"num_ctx": 4096, "temperature": 0.0})

        gate_prompt = (
            f"You are a strict hardware engineering decision gate.\n"
            f"Analyze file: {target_path}\n"
            f"Content:\n{text_data[:3000]}\n\n"
            f'Respond JSON ONLY: {{"is_relevant": true, "category_tag": "SKIDL_SUBCIRCUIT"}}'
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
                return gate_data.get("is_relevant", True), gate_data.get("category_tag", "SKIDL_SUBCIRCUIT")
            return True, "SKIDL_SUBCIRCUIT"

        except Exception as gate_err:
            if log_list is not None:
                log_list.append(f"   ⚠️ Gate-Ollama Error ({active_clef_model}): {gate_err}")
            return True, "GENERAL"

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

        synthesis_prompt = (
            f"Synthesize hardware source file into SKiDL Python & Markdown documentation.\n"
            f"File: {target_path}\nContent:\n{text_data[:25000]}"
        )

        try:
            synth_res = ollama_client.chat(
                model=active_model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                options=synthesis_opts
            )
            return initial_tag, synth_res['message']['content'].strip()

        except Exception as synth_err:
            if log_list is not None:
                log_list.append(f"   ❌ Synthese-Ollama Error ({active_model}): {synth_err}")
            return initial_tag, "SKIP"
