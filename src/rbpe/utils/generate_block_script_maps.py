"""
Generate bidirectional mappings between Unicode blocks and scripts.

This script creates four maps:
1. block_to_scripts: Maps each Unicode block to the set of scripts that appear in it
2. script_to_blocks: Maps each script to the set of blocks it appears in
3. script_name_to_alias: Maps full script names to their short aliases (e.g., "Arabic" -> "Arab")
4. alias_to_script_name: Maps short aliases to their full script names (e.g., "Arab" -> "Arabic")

For code points with "Common" or "Inherited" scripts, the script looks up the
ScriptExtensions to find all scripts the code point can be used with.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, List
from .data_reader import DataReader


class UnicodeDataGenerator:
    def __init__(self):
        self.data_reader = DataReader()
        self.blocks = self.data_reader.read_blocks()
        self.unified_aliases, self.script_name_to_alias, self.alias_to_script_name = self.data_reader.read_script_aliases()
        self.script_map = self.data_reader.read_scripts()
        self.extensions_map = self.data_reader.read_script_extensions()


    def get_block_for_codepoint(self, cp: int) -> str:
        """Find which block a code point belongs to."""
        for block_name, (start, end) in self.blocks.items():
            if start <= cp <= end:
                return block_name
        return None

    def generate_maps(self) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Generate block-to-scripts and script-to-blocks mappings.
        
        Returns:
            Tuple of (block_to_scripts, script_to_blocks)
        """
        block_to_scripts = defaultdict(set)
        script_to_blocks = defaultdict(set)
        
        # Iterate through all code points that have script assignments
        for cp, script_name in self.script_map.items():
            script_alias = self.unified_aliases[script_name]
            block = self.get_block_for_codepoint(cp)
            if not block:
                continue
            
            # Check if this is Common or Inherited
            if script_alias in ['zyyy', 'zinh']:
                # Look up in script extensions
                if cp in self.extensions_map:
                    scripts = self.extensions_map[cp]
                else:
                    # If not in extensions, just use the script itself
                    scripts = [script_alias]
            else:
                scripts = [script_alias]
            
            # Add to both maps
            for script in scripts:
                block_to_scripts[block].add(script)
                script_to_blocks[script].add(block)
        
        # Convert sets to sorted lists for JSON serialization
        block_to_scripts_json = {
            block: sorted(list(scripts)) 
            for block, scripts in sorted(block_to_scripts.items())
        }
        script_to_blocks_json = {
            script: sorted(list(blocks)) 
            for script, blocks in sorted(script_to_blocks.items())
        }
        
        return block_to_scripts_json, script_to_blocks_json


def main():
    parser = argparse.ArgumentParser(
        description='Generate bidirectional mappings between Unicode blocks and scripts.'
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / "data" / "unicode_derived",
        help='Directory to write output JSON files (default: data/)'
    )
    parser.add_argument(
        '-p',
        '--output-prefix',
        type=str,
        default='',
        help='Prefix for output filenames (default: none)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    output_dir = args.output_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unicode_data_generator = UnicodeDataGenerator()
    
    print("Parsing Unicode data files...")
    print("\nGenerating mappings...")
    block_to_scripts, script_to_blocks = unicode_data_generator.generate_maps()
    
    # Output files
    prefix = args.output_prefix + '_' if args.output_prefix else ''
    block_to_scripts_file = output_dir / f'{prefix}block_to_scripts.json'
    script_to_blocks_file = output_dir / f'{prefix}script_to_blocks.json'
    unified_aliases_file = output_dir / f'{prefix}unified_aliases.json'
    name_to_alias_file = output_dir / f'{prefix}script_name_to_alias.json'
    alias_to_name_file = output_dir / f'{prefix}alias_to_script_name.json'
    
    print(f"\nWriting output files...")
    print(f"  - {block_to_scripts_file}")
    with open(block_to_scripts_file, 'w', encoding='utf-8') as f:
        json.dump(block_to_scripts, f, indent=2, ensure_ascii=False)
    print(f"    {len(block_to_scripts)} blocks mapped to scripts")
    
    print(f"  - {script_to_blocks_file}")
    with open(script_to_blocks_file, 'w', encoding='utf-8') as f:
        json.dump(script_to_blocks, f, indent=2, ensure_ascii=False)
    print(f"    {len(script_to_blocks)} scripts mapped to blocks")

    print(f"  - {unified_aliases_file}")
    with open(unified_aliases_file, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(unicode_data_generator.unified_aliases.items())), f, indent=2, ensure_ascii=False)
    print(f"    {len(unicode_data_generator.unified_aliases)} unified aliases mapped to aliases")
    
    print(f"  - {name_to_alias_file}")
    with open(name_to_alias_file, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(unicode_data_generator.script_name_to_alias.items())), f, indent=2, ensure_ascii=False)
    print(f"    {len(unicode_data_generator.script_name_to_alias)} script names mapped to aliases")
    
    print(f"  - {alias_to_name_file}")
    with open(alias_to_name_file, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(unicode_data_generator.alias_to_script_name.items())), f, indent=2, ensure_ascii=False)
    print(f"    {len(unicode_data_generator.alias_to_script_name)} aliases mapped to script names")
    
    print("\n✓ Done!")
    return 0


if __name__ == '__main__':
    exit(main())

