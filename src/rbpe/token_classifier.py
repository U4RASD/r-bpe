import yaml
import json
from collections import defaultdict
from transformers import AutoTokenizer
from huggingface_hub import login
import logging

logger = logging.getLogger('BPE')


class TokenClassifier:
    def __init__(
        self,
        ranges_path: str,
        token_id_language_map_path: str,
        token_text_language_map_path: str,
        min_reusable_ids: int,
        vocabulary_languages_path: str,
        preserved_languages: list,
        old_tokenizer_model_id: str,
        hf_api_key: str = None,
        save_classified_tokens: bool = True
    ):
        """
        Initialize TokenClassifier.

        Args:
            ranges_path: Path to file containing Unicode character ranges
            token_id_language_map_path: Path to save token IDs classified by language
            token_text_language_map_path: Path to save token text classified by language
            min_reusable_ids: Minimum number of reusable IDs needed
            vocabulary_languages_path: Path to save classified vocabulary languages list
            preserved_languages: Languages to preserve and exclude from reuse
            old_tokenizer_model_id: HuggingFace model ID for base tokenizer
            hf_api_key: Optional HuggingFace API key
            save_classified_tokens: Whether to save classification results to files
        """
        self.ranges_path = ranges_path
        self.token_id_language_map_path = token_id_language_map_path
        self.token_text_language_map_path = token_text_language_map_path
        self.min_reusable_ids = min_reusable_ids
        self.vocabulary_languages_path = vocabulary_languages_path
        self.preserved_languages = preserved_languages
        self.old_tokenizer_model_id = old_tokenizer_model_id
        self.hf_api_key = hf_api_key
        self.reusable_languages = []
        self.save_classified_tokens = save_classified_tokens
        self.classified_ids_with_ranges = None
        self.classified_tokens_with_ranges = None
        self.all_languages_data = None

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
                range_str, language = line.strip().split("\t")
                lower_bound = self._hex_to_int(range_str.split()[0])
                upper_bound = self._hex_to_int(range_str.split()[2])
                
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
                    found_languages = {'Katakana + Hiragana'}
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
                classified_tokens[language].append(visible_token)
                classified_tokens_ids[language].append(token_id)
                classified_ranges[language].update(ranges)

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

    def _classify_and_save_tokens(self) -> None:
        """
        Classify tokens by language and optionally save the results to JSON files.
        """
        # Load and process the data
        tokenizer_data = self._load_tokenizer(self.old_tokenizer_model_id)
        unicode_ranges = self._load_unicode_ranges(self.ranges_path)
        
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
        
        if self.save_classified_tokens:
            # Save classified token IDs with ranges
            with open(self.token_id_language_map_path, "w") as json_file:
                json.dump(self.classified_ids_with_ranges, json_file, indent=4)
            logger.info(f"Saved classified token IDs to {self.token_id_language_map_path}")
            
            # Save classified tokens with ranges
            with open(self.token_text_language_map_path, "w") as json_file:
                json.dump(self.classified_tokens_with_ranges, json_file, indent=4)
            logger.info(f"Saved classified tokens with ranges to {self.token_text_language_map_path}")
        
        # Log statistics
        total_tokens = len(tokenizer_data['model']['vocab'])
        total_classified_tokens = sum(len(tokens) for tokens in classified_tokens_ids.values())
        
        logger.debug(f"Total Tokens in Vocab: {total_tokens}")
        logger.debug(f"Total Classified Tokens: {total_classified_tokens}")
        

    def _analyze_tokenizer_languages(self) -> tuple:
        """
        Analyze JSON file with tokenizer languages and their corresponding reusable IDs.
        Returns total reusable IDs, selected languages, total IDs available, and all languages.
        """
        self._classify_and_save_tokens()
        include_languages = [
            "cyrillic", "chinese", "korean", "japanese", "hebrew", "hindi", 
            "hangul", "hiragana", "thai", "tamil", "bengali", "armenian", 
            "burmese", "georgian", "tibetan", "khmer", "malayalam", "sinhala", 
            "kannada", "telugu", "katakana", "myanmar", "cjk", "devanagari", 
            "dingbats", "greek"
        ]
        with open(self.token_id_language_map_path, 'r') as file:
            language_data = json.load(file)
        
        filtered_languages = [
            (language, len(data["tokens"]))
            for language, data in language_data.items()
            if not any(preserved.lower() in language.lower() for preserved in self.preserved_languages) 
            and any(included.lower() in language.lower() for included in include_languages)
        ]

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

    def _write_sorted_languages_to_file(self, all_languages: list, output_file: str) -> None:
        """Write sorted languages and their ID counts to a file or store in memory."""
        sorted_all_languages = sorted(all_languages, key=lambda x: x[1], reverse=False)
        if self.save_classified_tokens:
            with open(output_file, 'w') as file:
                for language, id_count in sorted_all_languages:
                    file.write(f"{language}\t{id_count}\n")
        self.all_languages_data = sorted_all_languages

    def _get_reusable_languages(self) -> None:
        """
        Analyze tokenizer languages and logs statistics about reusable IDs.
        Also writes sorted languages to output file if save_classified_tokens is True.
        """
        total_ids, selected_languages, total_ids_available, all_languages = self._analyze_tokenizer_languages()
        
        logger.info(f"Total number of IDs available: {total_ids_available}")
        logger.info(f"Total number of reusable IDs (excluding languages containing {self.preserved_languages}): {total_ids}")
        logger.info(f"Languages selected to reach or exceed {self.min_reusable_ids} reusable IDs:")
        for lang in selected_languages:
            logger.info(f"  - {lang}")
        logger.info(f"Total number of languages selected for reuse: {len(selected_languages)}")

        if self.vocabulary_languages_path:
            self._write_sorted_languages_to_file(all_languages, self.vocabulary_languages_path)
            if self.save_classified_tokens:
                logger.info(f"Sorted languages and their ID counts have been written to {self.vocabulary_languages_path}")

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
            
        Raises:
            ValueError: If the classified tokens JSON file is not found
        """
        self._get_reusable_languages()
        
        # Use in-memory data if not saving to files
        if not self.save_classified_tokens:
            classified_tokens_with_ranges = self.classified_tokens_with_ranges
        else:
            try:
                with open(self.token_text_language_map_path, 'r', encoding='utf-8') as f:
                    classified_tokens_with_ranges = json.load(f)
            except FileNotFoundError:
                raise ValueError(f"Classified tokens file not found at {self.token_text_language_map_path}. Ensure tokens are classified and the file exists.")

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
