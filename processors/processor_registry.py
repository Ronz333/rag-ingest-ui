from .general_processor import GeneralProcessor
from .java_processor import JavaProcessor
from .javascript_processor import JavascriptProcessor
from .kicad_sym_processor import KiCadSymProcessor
from .oshw_circuit_processor import OSHWCircuitProcessor
from .pcb_eda_processor import PcbEdaProcessor
from .python_processor import PythonProcessor


class ProcessorRegistry:

  def __init__(self):
    self.processors = [
        KiCadSymProcessor(),
        OSHWCircuitProcessor(),
        PcbEdaProcessor(),
        PythonProcessor(),
        JavaProcessor(),
        JavascriptProcessor(),
        GeneralProcessor(),
    ]

  def get_categories_dict(self) -> dict:
    categories = {}
    for p in self.processors:
      if p.category_key not in categories:
        coll_name = (
            "pcb_knowledge_base"
            if "PCB" in p.category_key
            else "general_knowledge_base"
        )
        categories[p.category_key] = {
            "collection": coll_name,
            "description": f"Collection for {p.category_key}",
        }
    return categories

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
      ollama_client,
      llm_options: dict,
      max_embed_chars: int,
      selected_category: str = None,
      custom_filters: list = None,
  ) -> tuple[str, str]:
    for processor in self.processors:
      if processor.can_handle(file_path):
        return processor.parse(
            file_path,
            raw_content,
            model_name,
            ollama_client,
            llm_options,
            max_embed_chars,
            custom_filters,
        )
    return "GENERAL", raw_content


registry = ProcessorRegistry()
