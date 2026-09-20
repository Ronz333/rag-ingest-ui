from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

class BaseProcessor(ABC):
    category_key: str = ""
    collection_name: str = ""
    supported_extensions: set = set()

    @abstractmethod
    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        pass

    @abstractmethod
    def parse(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_model: str, 
        ollama_client, 
        num_ctx, 
        max_code_len: int,
        selected_category: str = "",
        custom_filters: Optional[List[str]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        pass
