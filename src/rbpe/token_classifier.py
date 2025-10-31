import yaml
import re
from collections import defaultdict
from pathlib import Path
import os
from transformers import AutoTokenizer
from huggingface_hub import login
import logging

from .utils.data_reader import DataReader

logger = logging.getLogger('BPE')


class TokenClassifier:
    def __init__(
        self,
        min_reusable_ids: int,
        old_tokenizer_model_id: str,
        hf_api_key: str = None,
        target_language_scripts: list = ["arabic"],
        preserved_languages_scripts: list = ["latin", "greek"],
    ):
        """
        Initialize TokenClassifier.

        Args:
            min_reusable_ids: Minimum number of reusable IDs needed
            old_tokenizer_model_id: HuggingFace model ID for base tokenizer
            hf_api_key: Optional HuggingFace API key
            target_language_scripts: Target language scripts of the new tokenizer and that will be preserved
            preserved_languages_scripts: Language scripts to preserve and exclude from reuse
        """
        self.data_reader = DataReader()
        self.unified_aliases, self.script_name_to_alias, self.alias_to_script_name = self.data_reader.read_script_aliases()
        self.blocks = self.data_reader.read_blocks()
        self.script_to_blocks = self.data_reader.read_json(self.data_reader.script_to_blocks_file)
        
        self._validate_params(
            min_reusable_ids,
            target_language_scripts,
            preserved_languages_scripts,
            old_tokenizer_model_id,
            hf_api_key,
        )
        
        self.min_reusable_ids = min_reusable_ids
        self.target_language_scripts = self._map_languages_to_aliases(target_language_scripts)
        self.preserved_languages_scripts = self._map_languages_to_aliases(preserved_languages_scripts)
        # Preserve both target and preserved languages
        self.preserved_languages_set = set(self.preserved_languages_scripts) | set(self.target_language_scripts)
        self.preserved_languages_scripts = list(self.preserved_languages_set)
        self.old_tokenizer_model_id = old_tokenizer_model_id
        self.hf_api_key = hf_api_key
        self.reusable_languages = []
        self.classified_ids_with_ranges = None
        self.classified_tokens_with_ranges = None
        self.all_languages_data = None
        self.preserved_blocks_ranges = None
    

    def _validate_params(
        self,
        min_reusable_ids: int,
        target_language_scripts: list,
        preserved_languages_scripts: list,
        old_tokenizer_model_id: str,
        hf_api_key: str = None,
    ):
        """
        Validate all initialization parameters.
        """
        if not isinstance(min_reusable_ids, int):
            raise ValueError(f"min_reusable_ids must be an integer: {min_reusable_ids}")
        if min_reusable_ids <= 0:
            raise ValueError(f"min_reusable_ids must be greater than 0: {min_reusable_ids}")
        
        if not isinstance(target_language_scripts, list):
            raise ValueError(f"target_language_scripts must be a list: {target_language_scripts}")
        if not isinstance(preserved_languages_scripts, list):
            raise ValueError(f"preserved_languages_scripts must be a list: {preserved_languages_scripts}")
        
        if not self._validate_languages(target_language_scripts):
            raise ValueError(f"target_language_scripts must be valid script names or aliases: {target_language_scripts}")
        if not self._validate_languages(preserved_languages_scripts):
            raise ValueError(f"preserved_languages_scripts must be valid script names or aliases: {preserved_languages_scripts}")


    def _validate_languages(self, languages: list) -> bool:
        """
        Validate a language is a valid script name or alias.
        """
        for language in languages:
            if language.lower() in self.unified_aliases:
                return True
            if language.lower() in self.script_name_to_alias:
                return True
        return False
    

    def _map_languages_to_aliases(self, languages: list) -> list:
        """
        Map languages to aliases.
        """
        return [self.unified_aliases[language.lower()] for language in languages]

    def _get_preserved_blocks_ranges(self) -> list:
        """
        Get the Unicode ranges (lower_bound, upper_bound) for blocks associated with
        target_language_scripts and preserved_languages_scripts.
        
        Returns:
            list of tuples: [(lower_bound, upper_bound), ...]
        """
        if self.preserved_blocks_ranges is not None:
            return self.preserved_blocks_ranges
        
        # Get all blocks associated with excluded scripts
        preserved_block_names = set()    
        for script in self.preserved_languages_scripts:
            if script in self.script_to_blocks:
                preserved_block_names.update(self.script_to_blocks[script])
        
        # Convert block names to ranges
        preserved_ranges = []
        for block_name in preserved_block_names:
            if block_name in self.blocks:
                logger.debug(f"Preserved block: {block_name}, ranges: {self.blocks[block_name]}")
                preserved_ranges.append(self.blocks[block_name])
        
        self.preserved_blocks_ranges = preserved_ranges
        logger.info(f"Preserving {len(preserved_ranges)} blocks associated with scripts: {self.preserved_languages_scripts}")
        
        return self.preserved_blocks_ranges

    def _ranges_overlap_with_preserved_blocks(self, ranges: list) -> bool:
        """
        Check if any of the given ranges overlap with preserved blocks.
        
        Args:
            ranges: list of (lower_bound, upper_bound) tuples
            
        Returns:
            bool: True if any range overlaps with preserved blocks
        """
        preserved_ranges = self._get_preserved_blocks_ranges()
        
        if not preserved_ranges:
            return False
        
        for range_tuple in ranges:
            range_start, range_end = range_tuple
            for preserved_start, preserved_end in preserved_ranges:
                # Check if ranges overlap
                if not (range_end < preserved_start or range_start > preserved_end):
                    return True
        
        return False

    def _hex_to_int(self, hex_str: str) -> int:
        """Convert a hexadecimal string to an integer."""
        return int(hex_str, 16)

    def _find_unicode_range(self, code_point: int, unicode_ranges: list) -> tuple:
        """
        Perform a binary search to find the corresponding language/script
        for a given Unicode code point within the specified unicode_ranges.
        """
        left, right = 0, len(unicode_ranges) - 1
        
        while left <= right:
            mid = (left + right) // 2
            lower_bound, upper_bound, script_language = unicode_ranges[mid]
            
            if lower_bound <= code_point <= upper_bound:
                return script_language, (lower_bound, upper_bound)
            elif code_point < lower_bound:
                right = mid - 1
            else:
                left = mid + 1
        
        return "Unknown", None

    def _load_unicode_ranges(self, file_path: str) -> list:
        """Load Unicode ranges and associated languages/scripts from a file."""
        unicode_ranges = []
        with open(file_path) as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                range_str, language = line.strip().split(";")
                language = language.strip()
                range_parts = range_str.strip().split("..")
                lower_bound = self._hex_to_int(range_parts[0])
                upper_bound = self._hex_to_int(range_parts[1])
                
                language_keywords = ["Arabic", "CJK", "Greek", "Latin"]
                for keyword in language_keywords:
                    if keyword in language:
                        language = f"{keyword}_merged"
                
                unicode_ranges.append([lower_bound, upper_bound, language])
        
        return sorted(unicode_ranges, key=lambda x: x[0])

    def _classify_token(self, token: str, unicode_ranges: list) -> tuple:
        """Classify a single token based on its characters."""
        found_languages = set()
        used_ranges = set()
        
        for char in token:
            script_language, char_range = self._find_unicode_range(ord(char), unicode_ranges)
            if script_language != 'General Punctuation' and char != "▁":
                found_languages.add(script_language)
                if char_range:
                    used_ranges.add(char_range)
        
        if len(found_languages) > 1:
            if all(lang in {'Katakana', 'Hiragana', 'CJK_merged'} for lang in found_languages):
                found_languages = {lang for lang in found_languages if "CJK" not in lang}
                if found_languages == {'Katakana', 'Hiragana'}:
                    found_languages = {'Katakana_Hiragana'}
            elif found_languages == {'Greek and Coptic', 'Greek Extended'}:
                found_languages = {'Greek and Coptic'}
            else:
                found_languages = {"_".join(sorted(found_languages))}
    
        classified_language = list(found_languages)[0] if found_languages else None
        return classified_language, list(used_ranges)

    def _classify_tokens_by_language(self, tokenizer: dict, unicode_ranges: list) -> tuple:
        """
        Classify tokens in the tokenizer's vocabulary according to their primary
        Unicode script or language, based on predefined Unicode ranges.
        """
        classified_tokens = defaultdict(list)
        classified_tokens_ids = defaultdict(list)
        classified_ranges = defaultdict(set)
        
        for token in tokenizer['model']['vocab']:
            token_id = tokenizer['model']['visible'][token]['id']
            visible_token = tokenizer['model']['visible'][token]['visible']
            language, ranges = self._classify_token(visible_token, unicode_ranges)
            
            if language:
                classified_tokens[language.lower()].append(visible_token)
                classified_tokens_ids[language.lower()].append(token_id)
                classified_ranges[language.lower()].update(ranges)

        return classified_tokens, classified_tokens_ids, classified_ranges

    def _load_tokenizer(self, model_id: str) -> dict:
        """Load and prepare the tokenizer data."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        tokenizer_data = {
            'model': {
                'vocab': tokenizer.get_vocab(),
                'visible': {}
            }
        }
        
        for vocab in tokenizer_data['model']['vocab']:
            token_ids = tokenizer.convert_tokens_to_ids([vocab])
            decoded_text = tokenizer.decode(token_ids)
            id = tokenizer_data['model']['vocab'][vocab]
            tokenizer_data['model']['visible'][vocab] = {
                'original': vocab, 
                'visible': decoded_text.strip(), 
                'id': id
            }
        
        return tokenizer_data

    def _classify_tokens(self) -> None:
        """
        Classify vocabulary tokens by language.
        """
        # Load and process the data
        tokenizer_data = self._load_tokenizer(self.old_tokenizer_model_id)
        unicode_ranges = self._load_unicode_ranges(self.data_reader.blocks_file)
        
        # Classify the tokens
        classified_tokens, classified_tokens_ids, classified_ranges = self._classify_tokens_by_language(
            tokenizer_data, 
            unicode_ranges
        )
        
        # Prepare classified token IDs with ranges
        self.classified_ids_with_ranges = {}
        for language in classified_tokens_ids:
            self.classified_ids_with_ranges[language] = {
                "ranges": list(classified_ranges[language]),
                "tokens": classified_tokens_ids[language]
            }
        
        # Prepare classified tokens with ranges
        self.classified_tokens_with_ranges = {}
        for language, tokens in classified_tokens.items():
            self.classified_tokens_with_ranges[language] = {
                "ranges": list(classified_ranges[language]),
                "tokens": tokens
            }
        
        # Log statistics
        total_tokens = len(tokenizer_data['model']['vocab'])
        total_classified_tokens = sum(len(tokens) for tokens in classified_tokens_ids.values())
        
        logger.debug(f"Total Tokens in Vocab: {total_tokens}")
        logger.debug(f"Total Classified Tokens: {total_classified_tokens}")
        

    def _analyze_tokenizer_languages(self) -> tuple:
        """
        Analyze classified tokens by language and their corresponding reusable IDs.
        Returns total reusable IDs, selected languages, total IDs available, and all languages.
        """
        self._classify_tokens()
        # all scripts are reusable besides common, inherited, and braille scripts
        include_languages = [script.lower() for script in self.script_name_to_alias.keys() if script not in ['common', 'inherited', 'braille']]
        include_languages.append("cjk")
        include_languages.append("dingbats") # to match legacy token classifcation
        logger.debug(f"Include languages: {include_languages}")
        
        language_data = self.classified_ids_with_ranges
        
        # CJK case: if any CJK scripts are preserved, add remaining CJK scripts to the preserved languages
        # CJK scripts overlap a lot in unicode blocks. It is safer to preserve all CJK scripts when one is needed to be preserved.
        cjk_case = False
        cjk_scripts = ["bopomofo", "hangul", "han", "hiragana", "katakana"]
        if any(script in self.preserved_languages_scripts for script in cjk_scripts):
            cjk_set = set(cjk_scripts)
            preserved_set = set(self.preserved_languages_scripts)
            preserved_set.update(cjk_set)
            self.preserved_languages_scripts = list(preserved_set)
            logger.info(f"CJK scripts detected. Updated preserved languages: {self.preserved_languages_scripts}")
            cjk_case = True
        
        filtered_languages = []
        for language, data in language_data.items():
            language = language.lower()
            logger.debug(f"Language: {language} Ranges: {data.get('ranges', [])}")

            # Exclude common, inherited, and braille scripts
            language_parts = language.replace('_', ' ').split()
            if not any(included.lower() in language_parts for included in include_languages):
                logger.debug(f"Excluding language '{language}'")
                continue
            
            # Check if language's ranges overlap with preserved blocks
            ranges = data.get("ranges", [])
            if ranges and self._ranges_overlap_with_preserved_blocks(ranges):
                logger.debug(f"Ranges overlap with preserved blocks: {ranges}")
                if any(preserved_language in language for preserved_language in self.preserved_languages_scripts) or (cjk_case and "cjk" in language):
                    logger.debug(f"Excluding language '{language}' due to overlap with preserved blocks")
                    continue
            
            filtered_languages.append((language, len(data["tokens"])))

        self.reusable_languages = [language for language, _ in filtered_languages]

        sorted_languages = sorted(filtered_languages, key=lambda x: x[1], reverse=False)
        
        total_reusable_ids = 0
        selected_languages = []
        for language, id_count in sorted_languages:
            total_reusable_ids += id_count
            selected_languages.append(language)
            if total_reusable_ids >= self.min_reusable_ids:
                break
        
        total_ids_available = sum(len(data["tokens"]) for data in language_data.values())
        all_languages = [(lang, len(data["tokens"])) for lang, data in language_data.items()]
        
        return total_reusable_ids, selected_languages, total_ids_available, all_languages

    def write_sorted_languages_to_file(self, all_languages: list, output_file: str) -> None:
        """Write sorted languages and their ID counts to a file or store in memory."""
        sorted_all_languages = sorted(all_languages, key=lambda x: x[1], reverse=False)
        with open(output_file, 'w') as file:
            for language, id_count in sorted_all_languages:
                file.write(f"{language}\t{id_count}\n")

    def _get_reusable_languages(self) -> None:
        """
        Analyze tokenizer languages and logs statistics about reusable IDs.
        """
        total_ids, selected_languages, total_ids_available, all_languages = self._analyze_tokenizer_languages()
        self.all_languages_data = all_languages
        
        logger.info(f"Total number of IDs available: {total_ids_available}")
        logger.info(f"Total number of reusable IDs (excluding languages containing {self.preserved_languages_scripts}): {total_ids}")
        logger.info(f"Languages selected to reach or exceed {self.min_reusable_ids} reusable IDs:")
        for lang in selected_languages:
            logger.info(f"  - {lang}")
        logger.info(f"Total number of languages selected for reuse: {len(selected_languages)}")

    def _read_text_file(self, text_file_path: str) -> dict:
        """Reads the text file and returns a dictionary of language counts."""
        language_counts = {}
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            for line in text_file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    language, count = parts[0], int(parts[1])
                    language_counts[language] = count
        return language_counts

    def _count_total_reusable_languages(self, language_counts: dict) -> int:
        """
        Calculates the total count of reusable languages.
        
        Args:
            language_counts: Dictionary containing language counts
            
        Returns:
            Total count of reusable languages
        """
        return sum(language_counts.get(lang, 0) for lang in self.reusable_languages)

    def get_reusable_languages_and_count(self) -> tuple:
        """
        Get a dictionary containing reusable languages and their corresponding Unicode ranges,
        along with the total count of reusable token IDs.
        
        Returns:
            tuple: (
                reusable_languages_with_ranges_dict: dict of {language: list of [lower_bound: int, upper_bound: int]},
                total_reusable_language_count: int
            )
        """
        self._get_reusable_languages()
        
        classified_tokens_with_ranges = self.classified_tokens_with_ranges

        reusable_languages_with_ranges_dict = {}
        total_reusable_language_count = 0

        for language in self.reusable_languages:
            language_data = classified_tokens_with_ranges.get(language)
            if language_data and "ranges" in language_data:
                # Each range is a list [lower_bound, upper_bound]
                ranges = language_data["ranges"]
                reusable_languages_with_ranges_dict[language] = ranges  # Directly assign ranges list
                total_reusable_language_count += len(language_data.get("tokens", []))
            else:
                # If no ranges found, assign an empty list
                reusable_languages_with_ranges_dict[language] = []
                # Optionally, count tokens even if ranges are missing
                tokens = language_data.get("tokens", []) if language_data else []
                total_reusable_language_count += len(tokens)
        
        return reusable_languages_with_ranges_dict, total_reusable_language_count
    
    def get_target_language_scripts_ranges(self) -> list:
        """
        Get the Unicode ranges (lower_bound, upper_bound) for blocks associated with
        target_language_scripts.
        
        Returns:
            list of tuples: [(lower_bound, upper_bound), ...]
        """
        target_ranges = []
        for script in self.target_language_scripts:
            if script in self.script_to_blocks:
                blocks = self.script_to_blocks[script]
                logger.debug(f"Target language scripts blocks for script {script}: {blocks}")
                for block in blocks:
                    if block in self.blocks:
                        target_ranges.append(self.blocks[block])
        return target_ranges