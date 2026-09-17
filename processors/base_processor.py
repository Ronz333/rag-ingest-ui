"""
BASE INGESTION PROCESSOR PLUGIN TEMPLATE
----------------------------------------
Abstrakte Basisklasse für alle RAG-Ingestion-Processoren.
Jedes Modul steuert seine eigenen Prompts, AST/Regex-Analysen und Dateiendungen.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Set, Optional


class BaseProcessor(ABC):
    # Eindeutiger Kategorie-Name für das Webpanel Dropdown
    category_key: str = "Allgemeines Wissen"
    
    # Qdrant Collection Zielname
    collection_name: str = "general_knowledge_base"
    
    # Der spezifische Ingestion-Systemprompt für dieses Modul
    system_prompt: str = ""
    
    # Set aller unterstützten Dateiendungen (z. B. {".py"})
    supported_extensions: Set[str] = set()

    @abstractmethod
    def can_handle(self, rel_path: str, ext: str) -> bool:
        """Prüft, ob dieser Processor für die angegebene Datei zuständig ist."""
        pass

    @abstractmethod
    def parse(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_model: str, 
        ollama_client, 
        num_ctx: int, 
        max_code_len: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Analysiert die Datei und gibt ein Tupel zurück:
        (category_tag, markdown_content) oder (None, None) falls übersprungen.
        """
        pass
