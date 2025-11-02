from typing import List, Optional, Union
from transformers import PreTrainedTokenizerBase, AutoTokenizer

import os
from pathlib import Path

from .mapping_tokenizer import MappingTokenizer
from transformers.utils import TensorType
import os
import json
from huggingface_hub import hf_hub_download

from transformers.tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    EncodedInput,
    EncodedInputPair,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
    PaddingStrategy
)

def create_dynamic_tokenizer(base_class, mapping_tokenizer: MappingTokenizer, config: dict):
    """Creates a new tokenizer class that inherits from the base tokenizer class."""
    
    class DynamicCustomTokenizer(base_class):
        def __init__(self, mapping_tokenizer, *args, **kwargs):
            self.custom_tokenizer_config = config
            model_id = kwargs.get('model_id', None)
            if model_id:
                # If it's a local folder, use its tokenizer.json
                if os.path.isdir(model_id):
                    tok_json = os.path.join(model_id, "tokenizer.json")
                    if not os.path.isfile(tok_json):
                        raise FileNotFoundError(f"Expected tokenizer.json in {model_id}")
                    kwargs['tokenizer_file'] = tok_json
                else:
                    # Fall back to Hub download
                    tokenizer_path = hf_hub_download(
                        repo_id=model_id,
                        filename="tokenizer.json",
                        cache_dir=os.path.join(
                            os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub'
                        )
                    )
                    kwargs['tokenizer_file'] = tokenizer_path
            else:
                raise ValueError("Cannot create dynamic tokenizer without a model_id")
            
            # __init__ of the parent class is not enough to have the tokenizer setup correctly, usually
            # the tokenizer is setup correctly because it is created using the from_pretrained method but since we are using 
            # a custom tokenizer we need to do this step manually by creating the pretrained tokenizer instance and then copying 
            # all attributes from the pretrained tokenizer

            # create the pretrained tokenizer instance
            self.mapping_tokenizer = mapping_tokenizer
            self._base_tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # initialize parent class normally
            super().__init__(*args, **kwargs)

            # copy all attributes from pretrained tokenizer
            for key, value in self._base_tokenizer.__dict__.items():
                setattr(self, key, value)

            # set up special tokens for both custom and old tokenizers
            self._setup_special_tokens(self, self._base_tokenizer, config)
            self._setup_special_tokens(self.mapping_tokenizer.old_tokenizer, self._base_tokenizer, config)

        def _setup_special_tokens(self, tokenizer, source_tokenizer, config):
            """Helper function to set up special tokens for a tokenizer"""
            special_tokens_dict = {
                'pad_token': config.get('pad_token') or source_tokenizer.pad_token,
                'eos_token': config.get('eos_token') or source_tokenizer.eos_token,
                'bos_token': config.get('bos_token') or source_tokenizer.bos_token,
                'unk_token': config.get('unk_token') or source_tokenizer.unk_token,
                'mask_token': config.get('mask_token') or source_tokenizer.mask_token,
                'sep_token': config.get('sep_token') or source_tokenizer.sep_token,
                'cls_token': config.get('cls_token') or source_tokenizer.cls_token,
            }
            
            # clean up None values
            special_tokens_dict = {k: v for k, v in special_tokens_dict.items() if v is not None}

            # add special tokens
            tokenizer.add_special_tokens(special_tokens_dict)
        
        def get_vocab_info(self):
            """
            Returns information about vocabulary sizes and changes.
            
            Returns:
                dict: Contains:
                    - original_vocab_size: Size of the original tokenizer's vocab
                    - current_vocab_size: Current size of the vocab
                    - new_tokens_count: Number of new tokens added
            """
            original_tokenizer = AutoTokenizer.from_pretrained(self.custom_tokenizer_config['model_id'])
            original_vocab_size = len(original_tokenizer.get_vocab())
            current_vocab_size = len(self.get_vocab())
            new_tokens_count = current_vocab_size - original_vocab_size
            
            return {
                "original_vocab_size": original_vocab_size,
                "current_vocab_size": current_vocab_size,
                "new_tokens_count": new_tokens_count
            }

        def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
                List[PreTokenizedInputPair],
                List[EncodedInput],
                List[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            padding_side: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            split_special_tokens: bool = False,
            **kwargs,
        ) -> BatchEncoding:
            # Handle both single texts and text pairs
            if isinstance(batch_text_or_text_pairs[0], (list, tuple)):
                batch_text, batch_text_pair = zip(*batch_text_or_text_pairs)
            else:
                batch_text = batch_text_or_text_pairs
                batch_text_pair = None

            # Encode all texts
            encoded_inputs = {
                "input_ids": [],
                "attention_mask": []
            }
            
            for i, text in enumerate(batch_text):
                # Get the text pair if it exists
                text_pair = batch_text_pair[i] if batch_text_pair is not None else None
                
                # Encode single text
                encoded = self.mapping_tokenizer.encode(text, add_special_tokens=add_special_tokens)
                if text_pair:
                    encoded_pair = self.mapping_tokenizer.encode(text_pair, add_special_tokens=add_special_tokens)
                    if add_special_tokens:
                        encoded = (
                            [self.bos_token_id] + 
                            encoded + 
                            [self.eos_token_id] + 
                            [self.bos_token_id] + 
                            encoded_pair + 
                            [self.eos_token_id]
                        )
                elif add_special_tokens:
                    encoded = [self.bos_token_id] + encoded + [self.eos_token_id]
                    
                encoded_inputs["input_ids"].append(encoded)
                encoded_inputs["attention_mask"].append([1] * len(encoded))
            
            # truncate sequences using the parent class's method
            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length is not None:
                for i, input_ids in enumerate(encoded_inputs["input_ids"]):
                    total_len = len(input_ids)
                    encoded_inputs["input_ids"][i], pair_ids, overflowing_tokens = self.truncate_sequences(
                        input_ids,
                        num_tokens_to_remove=total_len - max_length,
                        truncation_strategy=truncation_strategy,
                        stride=stride,
                    )
                    encoded_inputs["attention_mask"][i] = [1] * len(encoded_inputs["input_ids"][i])
                
                if return_overflowing_tokens:
                    encoded_inputs["overflowing_tokens"] = overflowing_tokens
                    encoded_inputs["num_truncated_tokens"] = total_len - max_length

            # let the parent class handle batching and padding
            batch_outputs = self.pad(
                encoded_inputs,
                padding=padding_strategy,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                verbose=verbose,
            )

            return batch_outputs
        
        def _decode(
            self,
            token_ids: Union[int, List[int]],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
        ) -> str:
            self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)
            # Handle batch input
            if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                return [self._decode(ids, skip_special_tokens, clean_up_tokenization_spaces) for ids in token_ids]

            text = self.mapping_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

            clean_up_tokenization_spaces = (
                clean_up_tokenization_spaces
                if clean_up_tokenization_spaces is not None
                else self.clean_up_tokenization_spaces
            )
            if clean_up_tokenization_spaces:
                clean_text = self.clean_up_tokenization(text)
                return clean_text
            else:
                return text

        def convert_tokens_to_string(self, tokens: List[str]) -> str:
            return self.mapping_tokenizer.decode(tokens)
        
        def convert_ids_to_tokens(self, ids):
            return self.mapping_tokenizer.convert_tok_ids_to_tokens(ids)

        def save_pretrained(self, save_directory: str, *args, **kwargs):
            os.makedirs(save_directory, exist_ok=True)

            # Bundle tokenizers
            new_tok_dir = os.path.join(save_directory, "new_tokenizer")
            old_tok_dir = os.path.join(save_directory, "old_tokenizer")
            os.makedirs(new_tok_dir, exist_ok=True)
            os.makedirs(old_tok_dir, exist_ok=True)

            # Save NEW tokenizer (it already lives on disk)
            self.mapping_tokenizer.new_tokenizer.save_pretrained(new_tok_dir)

            # Save OLD tokenizer snapshot (so loading is offline/self-contained)
            self.mapping_tokenizer.old_tokenizer.save_pretrained(old_tok_dir)
            
            # Save metadata files
            meta_dir = os.path.join(save_directory, "metadata")
            os.makedirs(meta_dir, exist_ok=True)

            # Save token classifier data
            with open(os.path.join(meta_dir, "token_id_language_map.json"), "w") as f:
                json.dump(self.custom_tokenizer_config['token_id_language_map'], f, indent=4)
            
            with open(os.path.join(meta_dir, "token_text_language_map.json"), "w") as f:
                json.dump(self.custom_tokenizer_config['token_text_language_map'], f, indent=4)
            
            with open(os.path.join(meta_dir, "vocabulary_languages.txt"), "w") as f:
                sorted_all_languages = sorted(self.custom_tokenizer_config['vocabulary_languages'], key=lambda x: x[1], reverse=False)
                for language, id_count in sorted_all_languages:
                    f.write(f"{language}\t{id_count}\n")
            
            # Save mapping tokenizer data
            with open(os.path.join(meta_dir, "new_to_old_map.json"), "w") as f:
                json.dump(self.custom_tokenizer_config['new_to_old_map'], f, indent=4)
            with open(os.path.join(meta_dir, "old_to_new_map.json"), "w") as f:
                json.dump(self.custom_tokenizer_config['old_to_new_map'], f, indent=4)
            with open(os.path.join(meta_dir, "replacement_character_map.json"), "w") as f:
                json.dump(self.custom_tokenizer_config['replacement_character_map'], f, indent=4)
            

            # Make mapping_tokenizer JSON-safe
            original_mapping_tokenizer = self.mapping_tokenizer
            mapping_tokenizer_json = self.mapping_tokenizer.to_json()
            self.mapping_tokenizer = mapping_tokenizer_json

            if not hasattr(self, 'init_kwargs'):
                self.init_kwargs = {}
            self.init_kwargs['mapping_tokenizer'] = mapping_tokenizer_json
            self.init_kwargs['custom_tokenizer_config'] = self.custom_tokenizer_config

            # Ensure tokenizer_config.json contains relative pointers
            # Call parent to write standard files (including tokenizer_config.json)
            result = super().save_pretrained(save_directory, *args, **kwargs)

            # restore mapping_tokenizer object
            self.mapping_tokenizer = original_mapping_tokenizer
            return result

    return DynamicCustomTokenizer