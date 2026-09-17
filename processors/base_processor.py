"""
BASE INGESTION PROCESSOR PLUGIN TEMPLATE
----------------------------------------
Abstrakte Basisklasse mit erweiterter Relevanz-Prüfung für GUI-Kategorien.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Set, Optional


class BaseProcessor(ABC):
    category_key: str = "Allgemeines Wissen"
    collection_name = "general_knowledge_base"
    system_prompt: str = ""
    supported_extensions: Set[str] = set()

    @abstractmethod
    def can_handle(self, rel_path: str, ext: str, selected_category: str = "") -> bool:
        """Prüft, ob dieser Processor für Dateiendung und gewählte Kategorie zuständig ist."""
        pass

    @abstractmethod
    def parse(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_model: str, 
        ollama_client, 
        num_ctx: int, 
        max_code_len: int,
        selected_category: str = ""
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Analysiert die Datei.
        Rückgabe-Optionen:
        - (category_tag, markdown_content) bei Erfolg.
        - ("IGNORED", "SKIP") wenn die Datei für die gewählte Kategorie irrelevant ist.
        - (None, None) wenn die Datei nicht verarbeitet werden kann.
        """
        pass
