from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple
import re

class DataReader:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.blocks_file = self.data_dir / "unicode" /"Blocks.txt"
        self.scripts_file = self.data_dir / "unicode" / "Scripts.txt"
        self.extensions_file = self.data_dir / "unicode" / "ScriptExtensions.txt"
        self.aliases_file = self.data_dir / "unicode" / "PropertyValueAliases.txt"
        self.block_to_scripts_file = self.data_dir / "unicode_derived" / "block_to_scripts.json"
        self.script_to_blocks_file = self.data_dir / "unicode_derived" / "script_to_blocks.json"
        self.script_name_to_alias_file = self.data_dir / "unicode_derived" / "script_name_to_alias.json"
        self.alias_to_script_name_file = self.data_dir / "unicode_derived" / "alias_to_script_name.json"


    def read_json(self, file_name: str) -> dict:
        with open(self.data_dir / file_name, 'r') as f:
            return json.load(f)
    

    def read_pickle(self, file_name: str) -> dict:
        with open(self.data_dir / file_name, 'rb') as f:
            return pickle.load(f)
    

    def parse_code_point_range(self, range_str: str) -> Tuple[int, int]:
        """Parse a code point range like '0600..06FF' or single point like '0020'."""
        if '..' in range_str:
            start, end = range_str.split('..')
            return int(start, 16), int(end, 16)
        else:
            point = int(range_str, 16)
            return point, point


    def read_blocks(self) -> dict:
        """
        Parse Blocks.txt to get block definitions.
        Returns: dict mapping block name to (start, end) code point tuple
        """
        blocks = {}
        with open(self.blocks_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Format: Start Code..End Code; Block Name
                match = re.match(r'^([0-9A-F]+)\.\.([0-9A-F]+)\s*;\s*(.+)$', line)
                if match:
                    start = int(match.group(1), 16)
                    end = int(match.group(2), 16)
                    block_name = match.group(3).strip().lower()
                    blocks[block_name] = (start, end)
        
        return blocks


    def read_scripts(self) -> Dict[int, str]:
        """
        Parse Scripts.txt to get script assignments.
        Returns: dict mapping code point to script name
        """
        script_map = {}
        
        with open(self.scripts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Format: code_point(s) ; script_name # comment
                match = re.match(r'^([0-9A-F.]+)\s*;\s*(\w+)', line)
                if match:
                    range_str = match.group(1)
                    script_name = match.group(2).lower()
                    
                    start, end = self.parse_code_point_range(range_str)
                    for cp in range(start, end + 1):
                        script_map[cp] = script_name
        
        return script_map


    def read_script_extensions(self) -> Dict[int, List[str]]:
        """
        Parse ScriptExtensions.txt to get extended script assignments.
        Returns: dict mapping code point to list of script aliases
        """
        extensions_map = {}
        
        with open(self.extensions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Format: code_point(s) ; script1 script2 script3 # comment
                match = re.match(r'^([0-9A-F.]+)\s*;\s*([^#]+)', line)
                if match:
                    range_str = match.group(1)
                    scripts_str = match.group(2).strip()
                    
                    start, end = self.parse_code_point_range(range_str)
                    for cp in range(start, end + 1):
                        extensions_map[cp] = [script.lower() for script in scripts_str.split()]
        
        return extensions_map
    

    def read_script_aliases(self) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """
        Parse PropertyValueAliases.txt to get script aliases.
        Returns: tuple of (unified_aliases, name_to_alias, alias_to_name)
            - general_aliases: maps both full names and aliases to aliases (for convenience)
            - name_to_alias: maps full script names to short aliases
            - alias_to_name: maps short aliases to full script names
        """
        unified_aliases = {}
        name_to_alias = {}
        alias_to_name = {}
        in_script_section = False
        
        with open(self.aliases_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Check if we're entering the Script section
                if line == '# Script (sc)':
                    in_script_section = True
                    continue
                
                # Check if we're leaving the Script section
                if in_script_section and line.startswith('# ') and line != '# Script (sc)':
                    break
                
                # Skip comments and empty lines
                if not in_script_section or not line or line.startswith('#'):
                    continue
                
                # Format: sc ; Alias ; Full_Name [ ; other_alias ]
                parts = [p.strip() for p in line.split(';')]
                if len(parts) >= 3 and parts[0] == 'sc':
                    alias = parts[1].lower()
                    full_name = parts[2].lower()
                    
                    # General aliases map (for backwards compatibility)
                    unified_aliases[full_name] = full_name
                    unified_aliases[alias] = full_name
                    
                    # Bidirectional specific maps
                    name_to_alias[full_name] = alias
                    alias_to_name[alias] = full_name
        
        return unified_aliases, name_to_alias, alias_to_name
    

