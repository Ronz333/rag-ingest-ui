from abc import ABC, abstractmethod


class BaseProcessor(ABC):

  def __init__(self):
    self.category_key = "General"
    self.supported_extensions = []

  @abstractmethod
  def can_handle(self, file_path: str) -> bool:
    pass

  @abstractmethod
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
    pass
