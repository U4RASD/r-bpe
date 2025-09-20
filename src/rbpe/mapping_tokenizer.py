import yaml
import string

from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import json

from .token_classifier import TokenClassifier
from .utils.unicode_normalizer import UnicodeNormalizer

from huggingface_hub import login

import logging

from .logger_config import setup_logger

logger = logging.getLogger('BPE')
if not logger.handlers:
    logger = setup_logger('BPE')

class MappingTokenizer:
    def __init__(self, new_tokenizer_path, old_tokenizer_model_id, token_id_language_map_path, reusable_languages,
                 cache_dir="./huggingface", new_to_old_map_path=None, old_to_new_map_path=None, 
                 replacement_character_map_path=None, save_maps=False, debug_mode=False):
        """
        Initialize the MappingTokenizer.

        Args:
            new_tokenizer_path (str): Path to the new tokenizer folder
            old_tokenizer_model_id (str): Model ID for the old tokenizer
            token_id_language_map_path (str): Path to JSON file with languages and their corresponding token IDs from the old tokenizer
            reusable_languages (List[str]): List of languages whose token ids will be reused
            cache_dir (str): Directory to use to cache tokenizers
            new_to_old_map_path (str): Path to save or load the new_to_old_map (new ids to old ids)
            old_to_new_map_path (str): Path to save or load the old_to_new_map (old ids to new ids)
            replacement_character_map_path (str): Path to save or load the replacement_character_map (old ids to replacement characters)
            save_maps (bool): Save the maps to the specified paths
        """
        self.new_tokenizer_path = new_tokenizer_path
        self.old_tokenizer_model_id = old_tokenizer_model_id
        self.new_tokenizer = AutoTokenizer.from_pretrained(self.new_tokenizer_path)
        self.old_tokenizer = AutoTokenizer.from_pretrained(self.old_tokenizer_model_id, cache_dir=cache_dir)
        self.old_vocab = self.old_tokenizer.get_vocab()
        self.new_vocab = self.new_tokenizer.get_vocab()
        self.token_id_language_map_path = token_id_language_map_path
        self.reusable_languages = reusable_languages
        self.reusable_token_ids, self.reusable_langs_unicode_ranges = self.get_reusable_token_ids()
        self.old_tokenizer_last_special_token_id = self.old_tokenizer.all_special_ids[-1]
        self.debug_mode = debug_mode
        self.unicode_normalizer = UnicodeNormalizer()
        if new_to_old_map_path is not None and old_to_new_map_path is not None and save_maps is False:
            self.new_to_old_map = self._load_token_map(new_to_old_map_path)
            self.old_to_new_map = self._load_token_map(old_to_new_map_path)
            self.replacement_character_map = self._load_token_map(replacement_character_map_path)
        else:
            self.new_to_old_map = self._create_token_map()
            self.old_to_new_map = {v: k for k, v in self.new_to_old_map.items()}
            self.replacement_character_map = self._create_replacement_character_map()
            self.save_maps(new_to_old_map_path, old_to_new_map_path, replacement_character_map_path)
        self.common_token_ids = self._init_common_token_ids()
        self.common_token_ids_map = {id: True for id in self.common_token_ids}
        self.old_tokenizer_arabic_ids = [id for id in self.old_vocab.values() if self._is_target_input(self.old_tokenizer.decode([id]))]
        self.new_tokenizer_arabic_ids = [id for token, id in self.new_vocab.items() if self._is_target_input(self.new_tokenizer.decode([id]))]
        self.new_tokenizer_arabic_ids_mapped = [self.new_to_old_map[id] for id in self.new_tokenizer_arabic_ids]

    @classmethod
    def from_config(cls, config_path):
        """Initialize MappingTokenizer from a YAML config file."""
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return cls(**config)
    
    def _load_token_map(self, path):
        """Load a token map from a JSON file."""
        with open(path) as f:
            token_map = json.load(f)
            # Check if the loaded data is a dictionary
            if isinstance(token_map, dict):
                token_map = {int(k): v for k, v in token_map.items()}
            return token_map
    
    def _init_common_token_ids(self):
        """Initialize common token IDs between the new and old tokenizers."""
        _, common_tokens= self.get_token_sets(self.new_tokenizer, self.old_tokenizer)
        common_token_ids = [self.old_vocab [token] for token in common_tokens]
        return common_token_ids
    
    def get_reusable_token_ids(self):
        """Get reusable token IDs from the specified languages."""
        with open(self.token_id_language_map_path) as f:
            lang_ids_info = json.load(f)
        reusable_ids = []
        reusable_langs_unicode_ranges = {}
        for lang in self.reusable_languages:
            reusable_ids.extend(lang_ids_info[lang]['tokens'])
            reusable_langs_unicode_ranges[lang] = lang_ids_info[lang]['ranges']
        # exclude special tokens and initial bytes from reusable tokens (special tokens till id 8)
        reusable_ids = [id for id in reusable_ids if id > 263]
        logger.debug(f"Found {len(reusable_ids)} reusable token IDs")
        return reusable_ids, reusable_langs_unicode_ranges
    
    def get_token_sets(self, new_tokenizer, original_tokenizer):
        """Get sets of new and common tokens between two tokenizers."""
        new_tokenizer_tokens = set([token for token in new_tokenizer.get_vocab().keys()])
        original_tokenizer_tokens = set([token for token in original_tokenizer.get_vocab().keys()])
        new_pieces = new_tokenizer_tokens - original_tokenizer_tokens
        common_pieces = new_tokenizer_tokens.intersection(original_tokenizer_tokens)
        logger.debug(f"New tokenizer vocabulary size: {len(new_tokenizer_tokens)} tokens")
        logger.debug(f"Common tokens between tokenizers: {len(common_pieces)}")

        return new_pieces, common_pieces
    
    def get_visible_token(self, id, is_old):
        if is_old:
            token_ids = self.old_tokenizer.convert_tokens_to_ids([id])
            decoded_text = self.old_tokenizer.decode(token_ids)
        else:
            token_ids = self.new_tokenizer.convert_tokens_to_ids([id])
            decoded_text = self.new_tokenizer.decode(token_ids)
        return decoded_text

    def _create_token_map(self):
        """Create a mapping between new and old token IDs."""
        new_tokens, common_tokens = self.get_token_sets(self.new_tokenizer, self.old_tokenizer)
        new_tokens = [token for token in new_tokens]

        self.common_token_ids = [self.old_vocab[token] for token in tqdm(common_tokens, desc="Getting common token IDs")]

        if len(new_tokens) > len(self.reusable_token_ids):
            error_msg = f"Not enough old token IDs ({len(self.reusable_token_ids)}) provided for mapping all new tokens ({len(new_tokens)})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"New tokens to be mapped: {len(new_tokens)}")
        logger.debug(f"Common tokens to be preserved: {len(self.common_token_ids)}")     

        mapping = {self.new_vocab[token]: old_id for token, old_id in tqdm(zip(new_tokens, self.reusable_token_ids), desc="Mapping new tokens")}

        mapping.update({
            self.new_vocab[token]: self.old_vocab[token] for token in tqdm(common_tokens, desc="Mapping common tokens")
        })

        new_vocab_by_id = {id: token for token, id in self.new_vocab.items()}
        old_vocab_by_id = {id: token for token, id in self.old_vocab.items()}

        return mapping
    
    def _create_replacement_character_map(self):
        """Create a mapping of token IDs to their decoded text for tokens that contain replacement characters."""
        token_ids = range(len(self.old_tokenizer))
        replacement_character_tokens = {}
        
        for token_id in token_ids:
            decoded = self.old_tokenizer.decode([token_id])
            if "�" in decoded:
                replacement_character_tokens[token_id] = decoded
        return replacement_character_tokens


    def _is_target_input(self, text):
        """Check if the given text contains Arabic characters."""
        arabic_ranges = [
            (0x0600, 0x06FF),   # Arabic
            (0x0750, 0x077F),   # Arabic Supplement
            (0x08A0, 0x08FF),   # Arabic Extended-A
            (0x0870, 0x089F),   # Arabic Extended-B
            (0x10EC0, 0x10EFF), # Arabic Extended-C
            (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
            (0x1EE00, 0x1EEFF)  # Arabic Mathematical Alphabetic Symbols
        ]
        for char in text:
            code_point = ord(char)
            if any(start <= code_point <= end for start, end in arabic_ranges):
                return True
        return False
    
    
    def encode(self, text, **kwargs):
        """Encode the input text to token IDs by segmenting into Arabic and non-Arabic parts."""
        kwargs['add_special_tokens'] = False 

        text = self.unicode_normalizer.normalize(text)

        # First attempt with segmented encoding
        # Initialize variables
        segments = []
        current_segment = []
        is_current_arabic = None
        
        # Process text character by character
        i = 0
        while i < len(text):
            char = text[i]
            is_char_arabic = self._is_target_input(char)

            # add spaces to current segment without further checks
            # they get encoded and decoded according to the segment they got added to
            if char.isspace():
                current_segment.append(char)
                i += 1
                continue
                    
            # handle first non-whitespace character
            if is_current_arabic is None:
                is_current_arabic = is_char_arabic
                current_segment.append(char)
                i += 1
                continue
                    
            # if we're switching between Arabic and non-Arabic (or vice versa)
            if is_char_arabic != is_current_arabic:
                # save current segment if it exists
                if current_segment:
                    segments.append((''.join(current_segment), is_current_arabic))
                current_segment = [char]
                is_current_arabic = is_char_arabic
                i += 1
                continue
            else:
                # Continue current segment
                current_segment.append(char)
                i += 1
        
        # add final segment
        if current_segment:
            segments.append((''.join(current_segment), is_current_arabic))

        if self.debug_mode:
            logger.debug(f"Original text:\n{text}")
            logger.debug(f"Encoding segments:\n{segments}")

        # encode each segment with appropriate tokenizer
        final_encoding = []
        for segment_text, is_arabic in segments:
            if is_arabic:
                # use new tokenizer for Arabic segments
                new_ids = self.new_tokenizer.encode(segment_text, **kwargs)
                # map new ids to old ids
                old_ids = [self.new_to_old_map[id] for id in new_ids]
                final_encoding.extend(old_ids)
                if self.debug_mode:
                    logger.debug(f"Arabic segment:\n{segment_text}")
                    for id in new_ids:
                        logger.debug(f"New ID: {id}, Old ID: {self.new_to_old_map[id]}")
            else:
                # use old tokenizer for non-Arabic segments
                old_ids = self.old_tokenizer.encode(segment_text, **kwargs)
                final_encoding.extend(old_ids)

        return final_encoding

    def basic_decode(self, ids, **kwargs):
        """Decode the input token IDs to text by segmenting based on tokenizer mapping."""        
        old_to_new_map = self.old_to_new_map
                
        segments = []
        current_segment = []
        current_is_mapped = None

        # pre-cast IDs if they are not already ints:
        ids = [int(i) for i in ids]

        # Group IDs into segments
        for id in ids:
            # An ID is mapped only if it's in old_to_new_map 
            is_mapped = id in old_to_new_map

            # Handle first token in sequence
            if current_is_mapped is None:
                current_is_mapped = is_mapped
                current_segment.append(id)
                continue

            # Check if we need to start a new segment
            if is_mapped != current_is_mapped:
                if current_segment:
                    segments.append((current_segment, current_is_mapped))
                current_segment = [id]
                current_is_mapped = is_mapped
            else:
                current_segment.append(id)

        # Add final segment if it exists
        if current_segment:
            segments.append((current_segment, current_is_mapped))
                
        # Decode each segment with the appropriate tokenizer
        decoded_segments = []
        for segment_ids, is_mapped in segments:
            if is_mapped:
                # Convert old IDs to new IDs and decode with the new tokenizer
                new_ids = [self.old_to_new_map[id] for id in segment_ids]
                decoded_text = self.new_tokenizer.decode(new_ids, **kwargs)
            else:
                # Decode directly with the old tokenizer
                decoded_text = self.old_tokenizer.decode(segment_ids, **kwargs)
            decoded_segments.append(decoded_text)
        
        final_text = ''.join(decoded_segments)
        return final_text
    
    def _decode_segment(self, segment, is_mapped, **kwargs):
        """Decode a segment with the appropriate tokenizer based on mapping status."""
        if is_mapped:
            # Convert old IDs to new IDs and decode with the new tokenizer
            new_ids = [self.old_to_new_map[token_id] for token_id in segment]
            return self.new_tokenizer.decode(new_ids, **kwargs)
        else:
            # Decode directly with the old tokenizer
            return self.old_tokenizer.decode(segment, **kwargs)
    
    def _find_optimal_window(self, ids, start_idx, current_segment=None, current_is_mapped=None, **kwargs):
        """Find the optimal window size that successfully decodes replacement characters.
        
        Tries different window sizes (1-4 tokens) to find one that produces text without
        replacement characters, testing both tokenizers as appropriate.
        
        Returns:
            tuple: (best_window_size, best_decoded_text, is_mapped_flag)
        """
        if current_segment is None:
            current_segment = []
        
        # Try to group with up to 3 more tokens to form complete UTF-8 character
        max_window_size = min(4, len(ids) - start_idx)  # At most 4 tokens (current + 3 more)
        best_window_size = 1
        best_decoded = None
        best_is_mapped = None
        
        # Try different window sizes with both tokenizers
        for window_size in range(1, max_window_size + 1):
            test_window = ids[start_idx:start_idx+window_size]
            test_segment = current_segment + test_window
            
            # Check if any token in the window is not in old_to_new_map
            window_has_unmapped = any(tid not in self.old_to_new_map for tid in test_window)
            
            # Try decoding with old tokenizer if any token is unmapped
            if window_has_unmapped or (current_is_mapped is not None and not current_is_mapped):
                decoded = self.old_tokenizer.decode(test_segment, **kwargs)
                if "�" not in decoded:
                    best_window_size = window_size
                    best_decoded = decoded
                    best_is_mapped = False
                    break
            
            # Try decoding with new tokenizer if all tokens are mapped
            if all(tid in self.old_to_new_map for tid in test_segment):
                new_ids = [self.old_to_new_map[tid] for tid in test_segment]
                decoded = self.new_tokenizer.decode(new_ids, **kwargs)
                if "�" not in decoded:
                    best_window_size = window_size
                    best_decoded = decoded
                    best_is_mapped = True
                    break
        
        return best_window_size, best_decoded, best_is_mapped
    
    def decode(self, ids, **kwargs):
        """Decode token IDs to text by handling replacement characters with a sliding window approach."""
        # Try basic decode first
        basic_decoded = self.basic_decode(ids, **kwargs)
        if "�" not in basic_decoded:
            return basic_decoded
        
        # Pre-cache frequently used attributes
        old_to_new_map = self.old_to_new_map
        replacement_character_map = self.replacement_character_map
        common_token_ids_map = self.common_token_ids_map
        old_last_special = self.old_tokenizer_last_special_token_id

        # pre-cast IDs if they are not already ints:
        ids = [int(i) for i in ids]
        
        # Identify segments for decoding
        segments = []
        current_segment = []
        current_is_mapped = None
        
        i = 0
        while i < len(ids):
            token_id = ids[i]
            is_byte = token_id > old_last_special and token_id <= (256 + old_last_special)
            is_replacement = token_id in replacement_character_map or is_byte
            is_common_token = token_id in common_token_ids_map
            is_mapped = token_id in old_to_new_map
            
            if self.debug_mode:
                logger.debug(f"ID: {token_id}, is_replacement: {is_replacement}, is_common_token: {is_common_token}, is_mapped: {is_mapped}, is_byte: {is_byte}")
                logger.debug(f"Current segment: {current_segment}")
                logger.debug(f"Current is mapped: {current_is_mapped}")
            
            # Handle first token in sequence
            if current_is_mapped is None:
                # If the first token is a replacement character and common, try window approach
                if is_replacement:
                    best_window_size, best_decoded, best_is_mapped = self._find_optimal_window(ids, i, current_segment, current_is_mapped, **kwargs)
                    
                    if best_decoded is not None:
                        segments.append((ids[i:i+best_window_size], best_is_mapped))
                        i += best_window_size
                        continue
                
                # Otherwise, handle normally
                current_is_mapped = is_mapped
                current_segment.append(token_id)
                i += 1
                continue
            
            # Special handling for replacement characters
            if is_replacement:
                best_window_size, best_decoded, best_is_mapped = self._find_optimal_window(ids, i, current_segment, current_is_mapped, **kwargs)
                
                # If we found a good window size
                if best_decoded is not None:
                    if best_is_mapped == current_is_mapped:
                        current_segment = current_segment + ids[i:i+best_window_size]
                        i += best_window_size
                    else:
                        segments.append((current_segment, current_is_mapped)) # add the current segment to the segments list
                        current_segment = ids[i:i+best_window_size]
                        current_is_mapped = best_is_mapped
                        i += best_window_size
                else:
                    # If we couldn't resolve the replacement character, add just this token
                    # and continue with normal segmentation
                    if is_mapped != current_is_mapped:
                        if current_segment:
                            segments.append((current_segment, current_is_mapped))
                        current_segment = [token_id]
                        current_is_mapped = is_mapped
                    else:
                        current_segment.append(token_id)
                    i += 1
                continue
            
            # Regular segmentation logic for other tokens
            if is_mapped != current_is_mapped:
                if current_segment:
                    segments.append((current_segment, current_is_mapped))
                current_segment = [token_id]
                current_is_mapped = is_mapped
            else:
                current_segment.append(token_id)
            i += 1
        
        # Add final segment if it exists
        if current_segment:
            segments.append((current_segment, current_is_mapped))
        
        if self.debug_mode:
            logger.debug(f"Decoding segments:\n{segments}")
        
        # Decode each segment with the appropriate tokenizer
        decoded_segments = [self._decode_segment(segment, is_mapped, **kwargs) for segment, is_mapped in segments]
        return ''.join(decoded_segments)
    
    def convert_tok_ids_to_tokens(self, ids):
        """Convert token IDs to tokens."""
        tokens = []
        for id in ids:
            if id in self.old_to_new_map and id in self.new_tokenizer_arabic_ids_mapped:
                tokens.append(self.new_tokenizer.decode([self.old_to_new_map[id]]))
            else:
                tokens.append(self.old_tokenizer.decode([id]))
        return tokens

    def save_maps(self, new_to_old_map_path, old_to_new_map_path, replacement_character_map_path):
        """Save all mapping files."""
        with open(new_to_old_map_path, 'w') as f:
            json.dump(self.new_to_old_map, f)
        with open(old_to_new_map_path, 'w') as f:
            json.dump(self.old_to_new_map, f)
        with open(replacement_character_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.replacement_character_map, f)
            
        logger.info(f"Token mapping files saved to:")
        logger.info(f"- New to old map: {new_to_old_map_path}")
        logger.info(f"- Old to new map: {old_to_new_map_path}")
        logger.info(f"- Replacement character map: {replacement_character_map_path}")
    
    def __getstate__(self):
        """Prepare the object for pickling."""
        state = self.__dict__.copy()
        # remove the tokenizer objects and unicode_normalizer as they might not be pickleable
        state.pop('new_tokenizer', None)
        state.pop('old_tokenizer', None)
        state.pop('unicode_normalizer', None)  # Remove the unicode_normalizer
        # store the paths instead
        state['new_tokenizer_path'] = self.new_tokenizer_path
        state['old_tokenizer_model_id'] = self.old_tokenizer_model_id
        state['new_to_old_map'] = self.new_to_old_map
        state['old_to_new_map'] = self.old_to_new_map
        state['replacement_character_map'] = self.replacement_character_map
        return state

    def __setstate__(self, state):
        """Restore the object from pickling."""
        self.__dict__.update(state)
        # restore the tokenizer objects and unicode_normalizer
        self.new_tokenizer = AutoTokenizer.from_pretrained(self.new_tokenizer_path)
        self.old_tokenizer = AutoTokenizer.from_pretrained(self.old_tokenizer_model_id)
        self.unicode_normalizer = UnicodeNormalizer()  # Recreate the unicode_normalizer

    def to_json(self):
        """Convert the object into a JSON string using the modified __dict__."""
        return json.dumps(self.__getstate__(), ensure_ascii=False).encode('utf8').decode()

    @classmethod
    def from_json(cls, json_str):
        """Recreate an object from a JSON string."""
        state = json.loads(json_str)
        state['new_to_old_map'] = {int(k): v for k,v in state['new_to_old_map'].items()}
        state['old_to_new_map'] = {int(k): v for k,v in state['old_to_new_map'].items()}
        state['replacement_character_map'] = {int(k): v for k,v in state['replacement_character_map'].items()}
        # create a blank instance without invoking __init__
        obj = cls.__new__(cls)
        obj.__setstate__(state)  # restore state
        return obj


def main():
    import os
    parser = argparse.ArgumentParser(description="Prepare a mapping tokenizer")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # READ FROM YAML
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Handle Hugging Face login if API key is provided
    if config.get('hf_token'):
        login(token=config['hf_token'])
        logger.info("Successfully logged in to Hugging Face Hub")
    else:
        logger.warning("No hf_token provided in config. You might have limited access to models.")
    
    # Set up paths
    output_dir = config['output_dir']
    metadata_dir = os.path.join(output_dir, "metadata")
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    cache_dir = os.path.join(output_dir, "cache")
    
    # Set up metadata paths
    token_id_language_map_path = os.path.join(metadata_dir, "token_id_language_map.json")
    token_text_language_map_path = os.path.join(metadata_dir, "token_text_language_map.json")
    reusable_languages_path = os.path.join(metadata_dir, "reusable_languages.txt")
    new_to_old_map_path = os.path.join(metadata_dir, "new_to_old_map.json")
    old_to_new_map_path = os.path.join(metadata_dir, "old_to_new_map.json")
    replacement_character_map_path = os.path.join(metadata_dir, "replacement_character_map.json")
    
    # Get ranges path
    current_script_path = os.path.abspath(__file__)
    rbpe_dir = os.path.dirname(current_script_path)
    src_dir = os.path.dirname(rbpe_dir)
    rbpe_repo_dir = os.path.dirname(src_dir)
    ranges_path = os.path.join(rbpe_repo_dir, "inputs", "unicode_ranges.txt")
    if not os.path.exists(ranges_path):
        default_ranges_path = os.path.join(rbpe_repo_dir, "inputs", "unicode_ranges.txt")
        if os.path.exists(default_ranges_path):
            import shutil
            shutil.copy(default_ranges_path, ranges_path)
            logger.debug(f"Copied ranges file from {default_ranges_path} to {ranges_path}")
        else:
            logger.warning(f"Ranges file not found at {default_ranges_path}")
            ranges_path = default_ranges_path

    token_classifier = TokenClassifier(
        ranges_path=ranges_path,
        token_id_language_map_path=token_id_language_map_path,
        token_text_language_map_path=token_text_language_map_path,
        min_reusable_ids=config.get('min_reusable_count', 20000),
        reusable_languages_path=reusable_languages_path,
        preserved_languages=config.get('preserved_langs', []),
        old_tokenizer_model_id=config['model_id'],
        hf_api_key=config.get('hf_token')
    )

    reusable_languages_with_ranges_dict, total_reusable_language_count = token_classifier.get_reusable_languages_and_count()

    reusable_languages = list(reusable_languages_with_ranges_dict.keys())

    should_create_mapping = config.get('force', {}).get('mapping', False)

    wrapper_tokenizer = MappingTokenizer(
        new_tokenizer_path=tokenizer_dir,
        old_tokenizer_model_id=config['model_id'],
        token_id_language_map_path=token_id_language_map_path,
        reusable_languages=reusable_languages,
        cache_dir=cache_dir,
        new_to_old_map_path=new_to_old_map_path,
        old_to_new_map_path=old_to_new_map_path,
        replacement_character_map_path=replacement_character_map_path,
        save_maps=should_create_mapping,
        debug_mode=args.debug
    )

    while True:
        print("\nOptions:")
        print("1. Encode and decode text with mapping tokenizer")
        print("2. Decode token IDs with mapping tokenizer")
        print("3. Decode token IDs with old tokenizer")
        print("4. Decode token IDs with new tokenizer")
        print("5. Encode and decode text with old tokenizer")
        print("6. Encode and decode text with new tokenizer")
        print("q. Quit")
        
        choice = input("Enter your choice: ")
        
        if choice.lower() == 'q':
            break
        
        if choice == '1':
            user_input = input("Enter text to encode and decode: ")
            encoded_text = wrapper_tokenizer.encode(user_input)
            encoded_tokens = wrapper_tokenizer.convert_tok_ids_to_tokens(encoded_text)
            decoded_text = wrapper_tokenizer.decode(encoded_text)
            
            print(f"Text encoded: {encoded_text}")
            print(f"Tokens: {encoded_tokens}")
            print(f"Text decoded: {decoded_text}")
        
        elif choice == '2':
            ids_input = input("Enter token IDs (comma-separated integers): ")
            try:
                ids = [int(id_str.strip()) for id_str in ids_input.split(',')]
                decoded_text = wrapper_tokenizer.decode(ids)
                tokens = wrapper_tokenizer.convert_tok_ids_to_tokens(ids)
                
                print(f"Tokens: {tokens}")
                print(f"Decoded text (mapping tokenizer): {decoded_text}")
            except ValueError:
                print("Error: Please enter valid comma-separated integers")
        
        elif choice == '3':
            ids_input = input("Enter token IDs (comma-separated integers): ")
            try:
                ids = [int(id_str.strip()) for id_str in ids_input.split(',')]
                decoded_text = wrapper_tokenizer.old_tokenizer.decode(ids)
                
                print(f"Decoded text (original tokenizer): {decoded_text}")
            except ValueError:
                print("Error: Please enter valid comma-separated integers")
                
        elif choice == '4':
            ids_input = input("Enter token IDs (comma-separated integers): ")
            try:
                ids = [int(id_str.strip()) for id_str in ids_input.split(',')]
                mapped_ids = [wrapper_tokenizer.old_to_new_map[id] for id in ids]
                decoded_text = wrapper_tokenizer.new_tokenizer.decode(mapped_ids)
                tokens = wrapper_tokenizer.new_tokenizer.convert_ids_to_tokens(mapped_ids)
                
                print(f"Tokens (new tokenizer): {tokens}")
                print(f"Decoded text (new tokenizer): {decoded_text}")
            except ValueError:
                print("Error: Please enter valid comma-separated integers")
        
        elif choice == '5':
            user_input = input("Enter text to encode and decode with old tokenizer: ")
            encoded_text = wrapper_tokenizer.old_tokenizer.encode(user_input, add_special_tokens=False)
            decoded_text = wrapper_tokenizer.old_tokenizer.decode(encoded_text)
            tokens = wrapper_tokenizer.old_tokenizer.convert_ids_to_tokens(encoded_text)
            
            print(f"Text encoded (old tokenizer): {encoded_text}")
            print(f"Tokens (old tokenizer): {tokens}")
            print(f"Text decoded (old tokenizer): {decoded_text}")
        
        elif choice == '6':
            user_input = input("Enter text to encode and decode with new tokenizer: ")
            encoded_text = wrapper_tokenizer.new_tokenizer.encode(user_input, add_special_tokens=False)
            decoded_text = wrapper_tokenizer.new_tokenizer.decode(encoded_text)
            tokens = wrapper_tokenizer.new_tokenizer.convert_ids_to_tokens(encoded_text)
            
            print(f"Text encoded (new tokenizer): {encoded_text}")
            print(f"Tokens (new tokenizer): {tokens}")
            print(f"Text decoded (new tokenizer): {decoded_text}")
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
