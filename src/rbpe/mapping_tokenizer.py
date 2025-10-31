import yaml
import string

from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import json
import re
import os

from .token_classifier import TokenClassifier
from .utils.unicode_normalizer import UnicodeNormalizer

from huggingface_hub import login

import logging

from .logger_config import setup_logger

logger = logging.getLogger('BPE')
if not logger.handlers:
    logger = setup_logger('BPE')

class MappingTokenizer:
    def __init__(
        self,
        new_tokenizer,
        old_tokenizer,
        token_id_language_map,
        reusable_languages,
        target_language_scripts_ranges,
        new_to_old_map_path=None,
        old_to_new_map_path=None,
        replacement_character_map_path=None,
        new_tokenizer_additional_special_tokens=None,
        apply_normalization=True,
        debug_mode=False,
        new_tokenizer_path: str = None,
        old_tokenizer_path: str = None,
    ):
        """
        Initialize the MappingTokenizer.

        Args:
            new_tokenizer (AutoTokenizer): The new tokenizer object
            old_tokenizer (AutoTokenizer): The old tokenizer object
            token_id_language_map (dict): Dictionary with languages and their corresponding token IDs from the old tokenizer
            reusable_languages (List[str]): List of languages whose token ids will be reused
            target_language_scripts_ranges (List[tuple]): List of Unicode ranges (lower_bound, upper_bound) for target language scripts
            new_to_old_map_path (str): Path to load the new_to_old_map (new ids to old ids)
            old_to_new_map_path (str): Path to load the old_to_new_map (old ids to new ids)
            replacement_character_map_path (str): Path to load the replacement_character_map (old ids to replacement characters),
            new_tokenizer_additional_special_tokens (List[str]): List of additional special tokens from the new tokenizer
            apply_normalization (bool): If True, apply the R-BPE Arabic normalization to text before encoding. Default is True.
            debug_mode (bool): Enable debug mode
            new_tokenizer_path (str): Path to the new tokenizer model (optional)
            old_tokenizer_path (str): Path to the old tokenizer model (optional)
        """
        self.new_tokenizer_path = new_tokenizer_path
        self.old_tokenizer_path = old_tokenizer_path
        if self.new_tokenizer_path and os.path.isdir(self.new_tokenizer_path):
            self.new_tokenizer = AutoTokenizer.from_pretrained(self.new_tokenizer_path)
        else:
            self.new_tokenizer = new_tokenizer
        if self.old_tokenizer_path and os.path.isdir(self.old_tokenizer_path):    
            self.old_tokenizer = AutoTokenizer.from_pretrained(self.old_tokenizer_path)
        else:
            self.old_tokenizer = old_tokenizer
        self.old_vocab = self.old_tokenizer.get_vocab()
        self.new_vocab = self.new_tokenizer.get_vocab()
        self.token_id_language_map = token_id_language_map
        self.reusable_languages = reusable_languages
        self.target_language_scripts_ranges = target_language_scripts_ranges
        self.old_tokenizer_last_special_token_id = self.old_tokenizer.all_special_ids[-1]
        self.debug_mode = debug_mode
        self.apply_normalization = apply_normalization
        self.unicode_normalizer = UnicodeNormalizer() if apply_normalization else None
        if (new_to_old_map_path is not None and 
            old_to_new_map_path is not None and 
            os.path.exists(new_to_old_map_path) and 
            os.path.exists(old_to_new_map_path)):
            self.new_to_old_map = self._load_token_map(new_to_old_map_path)
            self.old_to_new_map = self._load_token_map(old_to_new_map_path)
            self.replacement_character_map = self._load_token_map(replacement_character_map_path)
        else:
            self.reusable_token_ids, self.reusable_langs_unicode_ranges = self.get_reusable_token_ids()
            self.new_to_old_map = self._create_token_map()
            self.old_to_new_map = {v: k for k, v in self.new_to_old_map.items()}
            self.replacement_character_map = self._create_replacement_character_map()
        self.common_token_ids = self._init_common_token_ids()
        self.common_token_ids_map = {id: True for id in self.common_token_ids}
        self.old_tokenizer_target_ids = [id for id in self.old_vocab.values() if self._is_target_input(self.old_tokenizer.decode([id]))]
        self.new_tokenizer_target_ids = [id for token, id in self.new_vocab.items() if self._is_target_input(self.new_tokenizer.decode([id]))]
        self.new_tokenizer_target_ids_mapped = [self.new_to_old_map[id] for id in self.new_tokenizer_target_ids]
        self.new_additional_tokens = new_tokenizer_additional_special_tokens or []
        # if any special tokens are provided, compile a regex to detect them in text.
        if self.new_additional_tokens:
            # sort tokens by length to ensure that if one token is a prefix of another,
            # (e.g., "<extra_id_1>" and "<extra_id_10>"), the longer one is matched first.
            escaped = [re.escape(t) for t in sorted(self.new_additional_tokens, key=len, reverse=True)]
            self._new_specials_regex = re.compile(f"({'|'.join(escaped)})")
        else:
            self._new_specials_regex = None

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
        reusable_ids = []
        reusable_langs_unicode_ranges = {}
        for lang in self.reusable_languages:
            reusable_ids.extend(self.token_id_language_map[lang]['tokens'])
            reusable_langs_unicode_ranges[lang] = self.token_id_language_map[lang]['ranges']
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
        """Check if the given text contains target language characters."""
        for char in text:
            code_point = ord(char)
            if any(start <= code_point <= end for start, end in self.target_language_scripts_ranges):
                return True
        return False

    def _split_by_new_specials(self, text):
        """
        Returns a list of (segment, is_always_new). Segments marked True are the exact
        matched special tokens and must be encoded with the new tokenizer.
        """
        if not self._new_specials_regex:
            return [(text, False)]
        
        # split by special tokens, the regex uses capturing groups, so matched tokens
        # are included in the result list
        parts = self._new_specials_regex.split(text)
        
        # check each part against our special tokens list
        result = []
        for part in parts:
            if part:
                is_special = part in self.new_additional_tokens
                result.append((part, is_special))
        
        return result if result else [(text, False)]
    
    def _segment_input_text(self, text):
        """Segment the input text into target-language and non-target-language segments."""
        # First attempt with segmented encoding
        # Initialize variables
        segments = []
        current_segment = []
        is_current_target = None
        
        # Process text character by character
        i = 0
        while i < len(text):
            char = text[i]
            is_char_target = self._is_target_input(char)

            # add spaces to current segment without further checks
            # they get encoded and decoded according to the segment they got added to
            if char.isspace():
                current_segment.append(char)
                i += 1
                continue
                    
            # handle first non-whitespace character
            if is_current_target is None:
                is_current_target = is_char_target
                current_segment.append(char)
                i += 1
                continue
                    
            # if we're switching between target and non-target (or vice versa)
            if is_char_target != is_current_target:
                # save current segment if it exists
                if current_segment:
                    segments.append((''.join(current_segment), is_current_target))
                current_segment = [char]
                is_current_target = is_char_target
                i += 1
                continue
            else:
                # Continue current segment
                current_segment.append(char)
                i += 1
        
        # add final segment
        if current_segment:
            segments.append((''.join(current_segment), is_current_target))

        if self.debug_mode:
            logger.debug(f"Original text:\n{text}")
            logger.debug(f"Encoding segments:\n{segments}")
        return segments
    
    def encode(self, text, **kwargs):
        """Encode the input text to token IDs by segmenting into target and non-target parts."""
        kwargs['add_special_tokens'] = False 

        # handle additional special tokens from the new tokenizer
        initial_segments = self._split_by_new_specials(text)

        # encode each segment with appropriate tokenizer
        final_encoding = []
        for segment_text, is_always_new in initial_segments:
            if is_always_new:
                # use new tokenizer for always-new segments
                new_ids = self.new_tokenizer.encode(segment_text, **kwargs)
                # map new ids to old ids
                old_ids = [self.new_to_old_map[id] for id in new_ids]
                final_encoding.extend(old_ids)
                if self.debug_mode:
                    logger.debug(f"target segment:\n{segment_text}")
                    for id in new_ids:
                        logger.debug(f"New ID: {id}, Old ID: {self.new_to_old_map[id]}")
                continue
            
            normalized_segment_text = self.unicode_normalizer.normalize(segment_text) if self.apply_normalization else segment_text
            for segment_text, is_target in self._segment_input_text(normalized_segment_text):
                if is_target:
                    # use new tokenizer for target segments
                    new_ids = self.new_tokenizer.encode(segment_text, **kwargs)
                    # map new ids to old ids
                    old_ids = [self.new_to_old_map[id] for id in new_ids]
                    final_encoding.extend(old_ids)
                    if self.debug_mode:
                        logger.debug(f"target segment:\n{segment_text}")
                        for id in new_ids:
                            logger.debug(f"New ID: {id}, Old ID: {self.new_to_old_map[id]}")
                else:
                    # use old tokenizer for non-target segments
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
            tokens.append(self.decode([id]))
        return tokens
    
    def __getstate__(self):
        """Prepare the object for pickling."""
        state = self.__dict__.copy()
        # remove the tokenizer objects and unicode_normalizer as they might not be pickleable
        state.pop('new_tokenizer', None)
        state.pop('old_tokenizer', None)
        state.pop('unicode_normalizer', None)  # remove the unicode_normalizer
        state.pop('_new_specials_regex', None)  # remove the compiled regex pattern
        # store the paths instead
        state['new_tokenizer_path'] = self.new_tokenizer_path
        state['old_tokenizer_path'] = self.old_tokenizer_path
        state['new_to_old_map'] = self.new_to_old_map
        state['old_to_new_map'] = self.old_to_new_map
        state['replacement_character_map'] = self.replacement_character_map
        state['common_token_ids_map'] = self.common_token_ids_map
        return state

    def __setstate__(self, state):
        """Restore the object from pickling."""
        self.__dict__.update(state)
        # restore the tokenizer objects and unicode_normalizer
        self.new_tokenizer = AutoTokenizer.from_pretrained(self.new_tokenizer_path)
        self.old_tokenizer = AutoTokenizer.from_pretrained(self.old_tokenizer_path)
        self.old_vocab = self.old_tokenizer.get_vocab()
        self.new_vocab = self.new_tokenizer.get_vocab()
        # Recreate the unicode_normalizer only if normalization is enabled
        self.unicode_normalizer = UnicodeNormalizer() if self.apply_normalization else None
        
        # recreate the regex pattern from the special tokens list
        if self.new_additional_tokens:
            escaped = [re.escape(t) for t in sorted(self.new_additional_tokens, key=len, reverse=True)]
            self._new_specials_regex = re.compile(f"({'|'.join(escaped)})")
        else:
            self._new_specials_regex = None

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
        state['common_token_ids_map'] = {int(k): v for k,v in state['common_token_ids_map'].items()}
        
        # create a blank instance without invoking __init__
        obj = cls.__new__(cls)
        obj.__setstate__(state)  # restore state
        return obj
