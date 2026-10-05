from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """
    Abstrakte Basisklasse für alle Spezial-Processoren im RAG Ingestion System.
    """

    def __init__(self):
        self.category_key = "🌐 General Knowledge Base"
        self.supported_extensions = []

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Prüft anhand der Dateiendung, ob dieser Processor zuständig ist."""
        pass

    @abstractmethod
    def parse(
        self,
        file_path: str,
        raw_content: str,
        model_name: str,
        clef_model_name: str,
        ollama_client,
        llm_options: dict,
        max_embed_chars: int,
        custom_filters: list = None,
    ) -> tuple[str, str]:
        """
        Analysiert und synthetisiert den Dateiinhalt.
        Rückgabe: (category_tag, processed_markdown)
        """
        pass
