import os
from .base_processor import BaseProcessor


class GeneralProcessor(BaseProcessor):

  def __init__(self):
    super().__init__()
    self.category_key = "General Knowledge"
    self.supported_extensions = [
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".pdf",
    ]

  def can_handle(self, file_path: str) -> bool:
    return True

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
    return "GENERAL", raw_content
