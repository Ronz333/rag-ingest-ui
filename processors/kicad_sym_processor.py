import os
import re
from .base_processor import BaseProcessor


class KiCadSymProcessor(BaseProcessor):

  def __init__(self):
    super().__init__()
    self.category_key = "⚡ PCB & Hardware Design"
    self.supported_extensions = [".kicad_sym", ".lib", ".dcm"]

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
    ext = os.path.splitext(file_path)[1].lower()
    lib_name = os.path.basename(file_path).replace(ext, "")

    if ext == ".kicad_sym":
      symbols = self._parse_kicad_sym(raw_content, lib_name)
    elif ext == ".lib":
      symbols = self._parse_legacy_lib(raw_content, lib_name)
    else:
      return "KICAD_SYMBOL", "SKIP"

    if not symbols:
      return "KICAD_SYMBOL", "SKIP"

    output_md = [f"# KiCad Symbol Library Ground Truth: {lib_name}\n"]
    for sym in symbols:
      output_md.append(f"## Symbol: {sym['library']}:{sym['name']}")
      if sym.get("footprint"):
        output_md.append(f"- **Default Footprint:** `{sym['footprint']}`")
      if sym.get("description"):
        output_md.append(f"- **Description:** {sym['description']}")

      output_md.append("\n### Pin Mapping Table")
      output_md.append("| Pin Number | Pin Name | Pin Type |")
      output_md.append("|---|---|---|")
      for pin in sym["pins"]:
        output_md.append(f"| {pin['number']} | {pin['name']} | {pin['type']} |")
      output_md.append("\n---\n")

    return "KICAD_SYMBOL", "\n".join(output_md)

  def _parse_kicad_sym(self, content: str, lib_name: str) -> list:
    symbols = []
    sym_blocks = re.findall(
        r'\(symbol\s+"([^"]+)"\s*(.*?)\n\s*\)', content, re.DOTALL
    )

    for sym_name, sym_body in sym_blocks:
      if sym_name.count("_") > 1 and re.search(r'_\d+_\d+$', sym_name):
        continue

      footprint_match = re.search(
          r'\(property\s+"Footprint"\s+"([^"]*)"', sym_body
      )
      desc_match = re.search(
          r'\(property\s+"Description"\s+"([^"]*)"', sym_body
      )

      footprint = footprint_match.group(1) if footprint_match else ""
      description = desc_match.group(1) if desc_match else ""

      pins = []
      pin_matches = re.findall(
          r'\(pin\s+(\w+)\s+\w+.*?\(name\s+"([^"]*)"\).*?\(number\s+"([^"]*)"\)',
          sym_body,
          re.DOTALL,
      )
      for p_type, p_name, p_num in pin_matches:
        pins.append({"number": p_num, "name": p_name, "type": p_type})

      if pins:
        symbols.append({
            "library": lib_name,
            "name": sym_name,
            "footprint": footprint,
            "description": description,
            "pins": pins,
        })
    return symbols

  def _parse_legacy_lib(self, content: str, lib_name: str) -> list:
    symbols = []
    comp_blocks = re.findall(r"DEF\s+(\S+).*?ENDDEF", content, re.DOTALL)

    for block in comp_blocks:
      lines = block.splitlines()
      if not lines:
        continue
      sym_name = (
          lines[0].split()[1] if len(lines[0].split()) > 1 else "UNKNOWN"
      )
      pins = []
      footprint = ""

      for line in lines:
        if line.startswith("F2 "):
          parts = line.split('"')
          if len(parts) > 1:
            footprint = parts[1]
        elif line.startswith("X "):
          parts = line.split()
          if len(parts) >= 8:
            pins.append({"number": parts[2], "name": parts[1], "type": parts[8]})

      if pins:
        symbols.append({
            "library": lib_name,
            "name": sym_name,
            "footprint": footprint,
            "pins": pins,
        })
    return symbols
