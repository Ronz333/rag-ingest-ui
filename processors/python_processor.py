import os
from .base_processor import BaseProcessor


class PythonProcessor(BaseProcessor):

  def __init__(self):
    super().__init__()
    self.category_key = "Code & Software Architecture"
    self.supported_extensions = [".py"]

  def can_handle(self, file_path: str) -> bool:
    return file_path.endswith(".py")

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
    return "PYTHON_CODE", raw_content
