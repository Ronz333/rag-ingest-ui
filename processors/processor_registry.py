from .general_processor import GeneralDocumentProcessor
from .pcb_eda_processor import PcbEdaProcessor
from .python_processor import PythonProcessor
from .java_processor import JavaProcessor
from .javascript_processor import JavascriptProcessor
from .oshw_circuit_processor import OshwCircuitProcessor

class ProcessorRegistry:
    def __init__(self):
        # OshwCircuitProcessor steht vor PcbEdaProcessor
        self.processors = [
            OshwCircuitProcessor(),
            PcbEdaProcessor(),
            PythonProcessor(),
            JavaProcessor(),
            JavascriptProcessor(),
            GeneralDocumentProcessor()
        ]

    def get_categories_dict(self):
        categories = {}
        for p in self.processors:
            if p.category_key and p.category_key not in categories:
                categories[p.category_key] = {
                    "collection": p.collection_name
                }
        return categories

    def get_all_supported_extensions(self):
        exts = set()
        for p in self.processors:
            exts.update(p.supported_extensions)
        return exts

    def get_processor_for_file(self, rel_path: str, selected_category: str = ""):
        ext = "." + rel_path.split(".")[-1].lower() if "." in rel_path else ""
        for p in self.processors:
            if p.can_handle(rel_path, ext, selected_category):
                return p
        return None

    def dispatch_parse(self, rel_path, raw_text, active_model, ollama_client, num_ctx, max_code_len, selected_category="", custom_filters=None):
        processor = self.get_processor_for_file(rel_path, selected_category)
        if not processor:
            return "GENERAL", raw_text
        return processor.parse(
            rel_path, raw_text, active_model, ollama_client, num_ctx, max_code_len,
            selected_category=selected_category, custom_filters=custom_filters or []
        )

registry = ProcessorRegistry()
