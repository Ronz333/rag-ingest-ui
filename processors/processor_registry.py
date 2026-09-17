"""
AUTOMATIC PROCESSOR REGISTRY & PLUGIN DISCOVERER
-----------------------------------------------
Lädt dynamisch alle Processor-Klassen aus dem Ordner `processors/`.
Neue Plugin-Dateien werden beim Start automatisch erkannt, instanziiert
und im GUI-Dropdown bereitgestellt.
"""

import os
import sys
import importlib.util
import inspect
from typing import List, Set, Tuple, Optional, Dict
from .base_processor import BaseProcessor


class ProcessorRegistry:
    def __init__(self, processors_dir: Optional[str] = None):
        self.processors: List[BaseProcessor] = []
        if processors_dir is None:
            processors_dir = os.path.dirname(os.path.abspath(__file__))
        self.processors_dir = processors_dir
        self.reload_processors()

    def reload_processors(self):
        """Durchsucht das Verzeichnis nach allen gültigen Processor-Plugins."""
        self.processors.clear()
        
        if self.processors_dir not in sys.path:
            sys.path.insert(0, self.processors_dir)

        for entry in os.listdir(self.processors_dir):
            if entry.endswith('.py') and not entry.startswith('_') and entry not in ('base_processor.py', 'processor_registry.py'):
                file_path = os.path.join(self.processors_dir, entry)
                module_name = f"processors.{entry[:-3]}"

                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        for _, obj in inspect.getmembers(module, inspect.isclass):
                            if issubclass(obj, BaseProcessor) and obj is not BaseProcessor:
                                instance = obj()
                                self.processors.append(instance)
                except Exception as e:
                    print(f"⚠️ Fehler beim Laden des Processors '{entry}': {e}")

    def get_categories_dict(self) -> Dict[str, dict]:
        """Generiert dynamisch das `CATEGORIES`-Dictionary für die Web-GUI."""
        categories = {}
        for p in self.processors:
            categories[p.category_key] = {
                "collection": p.collection_name,
                "system_prompt": p.system_prompt,
                "processor": p
            }
        # Fallback für allgemeine Dokumente ergänzen
        if "📚 Allgemeines Wissen & Dokumente" not in categories:
            categories["📚 Allgemeines Wissen & Dokumente"] = {
                "collection": "general_knowledge_base",
                "system_prompt": "Du bist ein allgemeiner Dokumenten-Ingestion-Agent.",
                "processor": None
            }
        return categories

    def get_all_supported_extensions(self) -> Set[str]:
        """Sammelt alle unterstützten Dateiendungen aus allen registrierten Processoren."""
        exts = {".md", ".txt", ".json", ".yaml", ".yml", ".pdf", ".csv", ".xml", ".ini", ".conf", ".sh"}
        for p in self.processors:
            exts.update(p.supported_extensions)
        return exts

    def dispatch_parse(
        self, 
        rel_path: str, 
        raw_text: str, 
        active_model: str, 
        ollama_client, 
        num_ctx: int, 
        max_code_len: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """Findet den passenden Processor basierend auf Pfad und Endung."""
        ext = str(rel_path[rel_path.rfind('.'):]).lower() if '.' in rel_path else ""

        for processor in self.processors:
            if processor.can_handle(rel_path, ext):
                return processor.parse(
                    rel_path, raw_text, active_model, ollama_client, num_ctx, max_code_len
                )

        return None, None


registry = ProcessorRegistry()
