import os
from .base_processor import BaseProcessor


class PcbEdaProcessor(BaseProcessor):

  def __init__(self):
    super().__init__()
    self.category_key = "⚡ PCB & Hardware Design"
    self.supported_extensions = [".kicad_pcb", ".pro", ".kicad_pro"]

  def can_handle(self, file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in self.supported_extensions

  def parse(
      self,
      file_path: str,
      raw_content: str,
      model_name: str,
      ollama_client,
      llm_options: dict,
      max_embed_chars: int,
      custom_filters: list = None,
  ) -> tuple[str, str]:
    # Ignoriere reine Layout- und Projektdateien vollständig
    return "PCB_EDA", "SKIP"
