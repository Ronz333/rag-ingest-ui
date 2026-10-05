import os
from .general_processor import GeneralProcessor
from .oshw_circuit_processor import OSHWCircuitProcessor


class ProcessorRegistry:
    """
    Zentrale Registry für dynamisches Routing und Verwaltung aller Datei-Processoren.
    """

    def __init__(self):
        self.processors = [
            OSHWCircuitProcessor(),
            GeneralProcessor()  # Fallback-Processor
        ]

    def get_categories_dict(self) -> dict:
        return {
            "⚡ PCB & Hardware Design": {"collection": "skidl_patterns_kb", "tag": "SKIDL_SUBCIRCUIT"},
            "🌐 General Knowledge Base": {"collection": "general_knowledge_base", "tag": "GENERAL"},
            "📐 KiCad Symbol Library": {"collection": "kicad_sym_kb", "tag": "KICAD_SYMBOL"},
            "🚦 Freerouting Rules": {"collection": "freerouting_rules_kb", "tag": "DESIGN_RULE"}
        }

    def get_all_supported_extensions(self) -> set:
        exts = set()
        for p in self.processors:
            exts.update(p.supported_extensions)
        return exts

    def dispatch_parse(
        self,
        file_path: str,
        raw_content: str,
        model_name: str,
        clef_model_name: str,
        ollama_client,
        llm_options: dict,
        max_embed_chars: int,
        selected_category: str = None,
        custom_filters: list = None
    ) -> tuple[str, str]:
        for processor in self.processors:
            if processor.can_handle(file_path):
                return processor.parse(
                    file_path=file_path,
                    raw_content=raw_content,
                    model_name=model_name,
                    clef_model_name=clef_model_name,
                    ollama_client=ollama_client,
                    llm_options=llm_options,
                    max_embed_chars=max_embed_chars,
                    custom_filters=custom_filters
                )

        fallback = GeneralProcessor()
        return fallback.parse(
            file_path=file_path,
            raw_content=raw_content,
            model_name=model_name,
            clef_model_name=clef_model_name,
            ollama_client=ollama_client,
            llm_options=llm_options,
            max_embed_chars=max_embed_chars,
            custom_filters=custom_filters
        )


registry = ProcessorRegistry()
