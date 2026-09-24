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
        KiCadSymProcessor(),  # -> kicad_sym_kb
        OSHWCircuitProcessor(),  # -> skidl_patterns_kb
        PcbEdaProcessor(),  # -> freerouting_rules_kb
        PythonProcessor(),
        JavaProcessor(),
        JavascriptProcessor(),
        GeneralProcessor(),
    ]

  def get_categories_dict(self) -> dict:
    categories = {}
    for p in self.processors:
      # Ziel-Collection direkt aus dem Processor auslesen oder dynamisch zuweisen
      if isinstance(p, KiCadSymProcessor):
        coll_name = "kicad_sym_kb"
      elif isinstance(p, OSHWCircuitProcessor):
        coll_name = "skidl_patterns_kb"
      elif isinstance(p, PcbEdaProcessor):
        coll_name = "freerouting_rules_kb"
      else:
        coll_name = "general_knowledge_base"

      if p.category_key not in categories:
        categories[p.category_key] = {
            "collection": coll_name,
            "description": f"Dedicated collection {coll_name} for"
            f" {p.category_key}",
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
