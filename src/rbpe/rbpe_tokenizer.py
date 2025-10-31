from typing import List, Optional, Union
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

import os
import sys
import shutil
import pkg_resources
import importlib
from pathlib import Path
from datetime import datetime

from .token_classifier import TokenClassifier
from .data_cleaner import DataCleaner
from .bpe_tokenizer_trainer import BPETokenizerTrainer
from .mapping_tokenizer import MappingTokenizer
from transformers.utils import TensorType
import os
import json
from huggingface_hub import hf_hub_download
import yaml
import argparse

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
from huggingface_hub import login

from .logger_config import setup_logger

logger = setup_logger('BPE')

def _generate_default_output_dir(model_id: str) -> str:
    """Generate default output directory name from model ID.
    
    Args:
        model_id (str): The model ID (e.g., 'microsoft/DialoGPT-medium')
        
    Returns:
        str: Default output directory name (e.g., 'rbpe_microsoft_DialoGPT_medium')
    """
    # Extract model name from model_id (everything after the last '/')
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    # Replace dashes with underscores
    model_name_clean = model_name.replace('-', '_')
    return f"rbpe_{model_name_clean}"

def create_dynamic_tokenizer(base_class, mapping_tokenizer: MappingTokenizer, config: dict):
    """Creates a new tokenizer class that inherits from the base tokenizer class."""
    
    class DynamicCustomTokenizer(base_class):
        def __init__(self, mapping_tokenizer, *args, **kwargs):
            self.custom_tokenizer_config = config
            if kwargs.get('model_id'):
                tokenizer_path = hf_hub_download(
                    repo_id=kwargs['model_id'],
                    filename="tokenizer.json",
                    cache_dir=self.custom_tokenizer_config['cache_dir']
                )
                kwargs['tokenizer_file'] = tokenizer_path
            
            # __init__ of the parent class is not enough to have the tokenizer setup correctly, usually
            # the tokenizer is setup correctly because it is created using the from_pretrained method but since we are using 
            # a custom tokenizer we need to do this step manually by creating the pretrained tokenizer instance and then copying 
            # all attributes from the pretrained tokenizer

            # create the pretrained tokenizer instance
            self.mapping_tokenizer = mapping_tokenizer
            self._base_tokenizer = AutoTokenizer.from_pretrained(self.mapping_tokenizer.old_tokenizer_model_id)
            
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
            original_tokenizer = AutoTokenizer.from_pretrained(self.mapping_tokenizer.old_tokenizer_model_id)
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

        def save_pretrained(self, *args, **kwargs):
            # make sure the mapping_tokenizer is JSON serializable
            original_mapping_tokenizer = self.mapping_tokenizer
            mapping_tokenizer_json = self.mapping_tokenizer.to_json()
            self.mapping_tokenizer = mapping_tokenizer_json
            
            # add mapping_tokenizer and custom_tokenizer_config to the tokenizer config for loading with from_pretrained
            if not hasattr(self, 'init_kwargs'):
                self.init_kwargs = {}
            self.init_kwargs['mapping_tokenizer'] = mapping_tokenizer_json
            self.init_kwargs['custom_tokenizer_config'] = self.custom_tokenizer_config
            
            # save the tokenizer
            result = super().save_pretrained(*args, **kwargs)
            
            # restore original mapping_tokenizer
            self.mapping_tokenizer = original_mapping_tokenizer
            return result

    return DynamicCustomTokenizer

class RBPETokenizer:
    """Factory class to create and prepare a custom BPE tokenizer with minimal configuration requirements."""
    
    @classmethod
    def from_config(cls, config_path: str):
        """Initialize tokenizer from a YAML config file.
        
        Args:
            config_path (str): Path to YAML config file with simplified parameters
            
        Returns:
            RBPETokenizer: Initialized tokenizer instance
        """
        # Load config from YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract parameters from config
        model_id = config.get('model_id')
        output_dir = config.get('output_dir')
        training_data_dir = config.get('training_data_dir')
        cleaned_data_dir = config.get('cleaned_data_dir')
        hf_token = config.get('hf_token')
        min_reusable_count = config.get('min_reusable_count', 20000)
        target_language_scripts = config.get('target_language_scripts', [])
        preserved_languages_scripts = config.get('preserved_languages_scripts', [])
        
        # Generate default output_dir if not provided
        if not output_dir and model_id:
            output_dir = _generate_default_output_dir(model_id)
        
        # Extract force parameters
        force = config.get('force', {})
        force_classify = force.get('classify', False)
        force_clean = force.get('clean', False)
        force_train = force.get('train', False)
        force_mapping = force.get('mapping', False)
        
        # Extract special tokens
        special_tokens = config.get('special_tokens', {})
        additional_special_tokens = special_tokens.pop('additional_special_tokens', [])
        
        # Create instance with parameters
        instance = cls.__new__(cls)
        instance._init_from_params(
            model_id=model_id,
            output_dir=output_dir,
            training_data_dir=training_data_dir,
            cleaned_data_dir=cleaned_data_dir,
            hf_token=hf_token,
            min_reusable_count=min_reusable_count,
            target_language_scripts=target_language_scripts,
            preserved_languages_scripts=preserved_languages_scripts,
            force_classify=force_classify,
            force_clean=force_clean,
            force_train=force_train,
            force_mapping=force_mapping,
            additional_special_tokens=additional_special_tokens,
            **special_tokens
        )
        return instance
    
    @classmethod
    def from_params(cls, 
                  model_id: str,
                  output_dir: str = None,
                  training_data_dir: str = None,
                  cleaned_data_dir: str = None,
                  hf_token: str = None,
                  min_reusable_count: int = 20000,
                  target_language_scripts: list = None,
                  preserved_languages_scripts: list = None,
                  force_classify: bool = False,
                  force_clean: bool = False,
                  force_train: bool = False,
                  force_mapping: bool = False,
                  additional_special_tokens: list = None,
                  pad_token: str = None,
                  unk_token: str = None,
                  bos_token: str = None,
                  eos_token: str = None,
                  mask_token: str = None,
                  sep_token: str = None,
                  cls_token: str = None):
        """Initialize tokenizer with explicit parameters.
        
        Args:
            model_id (str): Base tokenizer model ID
            output_dir (str, optional): Directory to save tokenizer and metadata. If not provided, defaults to 'rbpe_{model_name}'
            training_data_dir (str, optional): Directory containing training data
            cleaned_data_dir (str, optional): Where to store cleaned data
            hf_token (str, optional): Hugging Face API token
            min_reusable_count (int, optional): Min token count for a reusable language. Default is 20000.
            target_language_scripts (list, optional): Target languages of the new tokenizer
            preserved_languages_scripts (list, optional): Languages to preserve
            force_classify (bool, optional): Force token classification even if exists
            force_clean (bool, optional): Force data cleaning even if already exists
            force_train (bool, optional): Force tokenizer training even if exists
            force_mapping (bool, optional): Force mapping creation even if exists
            pad_token (str, optional): Padding token
            unk_token (str, optional): Unknown token
            bos_token (str, optional): Beginning of sequence token
            eos_token (str, optional): End of sequence token
            mask_token (str, optional): Mask token
            sep_token (str, optional): Separator token
            cls_token (str, optional): Classification token
        
        Returns:
            RBPETokenizer: Initialized tokenizer instance
        """
        target_language_scripts = target_language_scripts or []
        preserved_languages_scripts = preserved_languages_scripts or []
        
        # Generate default output_dir if not provided
        if not output_dir and model_id:
            output_dir = _generate_default_output_dir(model_id)
        
        # Create instance and initialize
        instance = cls.__new__(cls)
        instance._init_from_params(
            model_id=model_id,
            output_dir=output_dir,
            training_data_dir=training_data_dir,
            cleaned_data_dir=cleaned_data_dir,
            hf_token=hf_token,
            min_reusable_count=min_reusable_count,
            target_language_scripts=target_language_scripts,
            preserved_languages_scripts=preserved_languages_scripts,
            force_classify=force_classify,
            force_clean=force_clean,
            force_train=force_train,
            force_mapping=force_mapping,
            additional_special_tokens=additional_special_tokens,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            sep_token=sep_token,
            cls_token=cls_token
        )
        return instance
    
    def _init_from_params(self, model_id, output_dir, training_data_dir, cleaned_data_dir=None,
                         hf_token=None, min_reusable_count=20000, target_language_scripts=None, preserved_languages_scripts=None,
                         force_classify=False, force_clean=False, force_train=False, force_mapping=False, additional_special_tokens=None, **special_tokens):
        """Internal method to initialize from parameters."""
        target_language_scripts = target_language_scripts or []
        
        self.tokenizer = None
        preserved_languages_scripts = preserved_languages_scripts or []
        
        # Validate required parameters
        if not model_id:
            raise ValueError("model_id is required")
        
        # Generate default output_dir if not provided
        if not output_dir:
            output_dir = _generate_default_output_dir(model_id)
        if (force_clean or force_train) and not training_data_dir:
            raise ValueError("training_data_dir is required when force_clean or force_train is True")
        if (force_clean or force_train) and not cleaned_data_dir:
            raise ValueError("cleaned_data_dir is required when force_clean or force_train is True")
        
        # Set up directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up cleaned data directory
        if not cleaned_data_dir:
            cleaned_data_dir = os.path.join(output_dir, "cleaned_data")
        os.makedirs(cleaned_data_dir, exist_ok=True)
        
        # Set up tokenizer directory
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # Set up metadata directory
        metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Set up cache directory
        cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set up metadata paths
        token_id_language_map_path = os.path.join(metadata_dir, "token_id_language_map.json")
        token_text_language_map_path = os.path.join(metadata_dir, "token_text_language_map.json")
        vocabulary_languages_path = os.path.join(metadata_dir, "vocabulary_languages.txt")
        new_to_old_map_path = os.path.join(metadata_dir, "new_to_old_map.json")
        old_to_new_map_path = os.path.join(metadata_dir, "old_to_new_map.json")
        replacement_character_map_path = os.path.join(metadata_dir, "replacement_character_map.json")
        output_reusable_samples_path = os.path.join(metadata_dir, "reusable_samples.txt")
        additional_special_tokens = additional_special_tokens or []
        
        # Create internal config
        self.config = {
            "model_id": model_id,
            "output_dir": output_dir,
            "training_data_dir": training_data_dir,
            "cleaned_data_dir": cleaned_data_dir,
            "tokenizer_dir": tokenizer_dir,
            "metadata_dir": metadata_dir,
            "cache_dir": cache_dir,
            "token_id_language_map_path": token_id_language_map_path,
            "token_text_language_map_path": token_text_language_map_path,
            "vocabulary_languages_path": vocabulary_languages_path,
            "new_to_old_map_path": new_to_old_map_path,
            "old_to_new_map_path": old_to_new_map_path,
            "replacement_character_map_path": replacement_character_map_path,
            "output_reusable_samples_path": output_reusable_samples_path,
            "min_reusable_ids": min_reusable_count,
            "target_language_scripts": target_language_scripts,
            "preserved_languages_scripts": preserved_languages_scripts,
            "hf_api_key": hf_token,
            "additional_special_tokens": additional_special_tokens,
            "force_classify": force_classify,
            "force_clean": force_clean,
            "force_train": force_train,
            "force_mapping": force_mapping
        }
        
        # Add special tokens to config
        for token_name, token_value in special_tokens.items():
            if token_value is not None:
                self.config[token_name] = token_value
        
        # Login to HF if token provided
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Successfully logged in to Hugging Face Hub")
            except Exception as e:
                error_msg = f"Failed to log in to Hugging Face Hub: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
    
    def prepare(self) -> PreTrainedTokenizerBase:
        """
        Orchestrates the complete tokenizer preparation process:
        1. Classifies tokens using TokenClassifier
        2. Cleans data using DataCleaner (if needed)
        3. Trains new tokenizer using BPETokenizerTrainer (if needed)
        4. Creates mapping using MappingTokenizer (if needed)
        5. Returns final RBPETokenizer instance
        
        Will auto-detect which steps to run based on existing files, unless
        forced with force_* parameters.
        
        Returns:
            PreTrainedTokenizerBase: The prepared tokenizer
        """
        logger.info("Starting tokenizer preparation process...")

        classifier_files = [self.config['token_id_language_map_path'], self.config['token_text_language_map_path']]
        classifier_files = [f for f in classifier_files if os.path.exists(f)]
        should_classify = self.config.get('force_classify', False) or not classifier_files
        logger.debug(f"Should classify: {should_classify}")

        # Step 1: Token Classification
        logger.info("1. Initializing TokenClassifier...")
        token_classifier = TokenClassifier(
            token_id_language_map_path=self.config['token_id_language_map_path'],
            token_text_language_map_path=self.config['token_text_language_map_path'],
            min_reusable_ids=self.config['min_reusable_ids'],
            vocabulary_languages_path=self.config['vocabulary_languages_path'],
            target_language_scripts=self.config['target_language_scripts'],
            preserved_languages_scripts=self.config['preserved_languages_scripts'],
            old_tokenizer_model_id=self.config['model_id'],
            hf_api_key=self.config.get('hf_api_key'),
            save_classified_tokens=should_classify
        )

        # Get reusable languages and ranges
        reusable_languages_dict, total_reusable_count = token_classifier.get_reusable_languages_and_count()
        target_language_scripts_ranges = token_classifier.get_target_language_scripts_ranges()
        
        # Step 2: Clean Data if needed
        should_clean = self.config.get('force_clean', False)
        
        if should_clean:
            logger.info("2. Starting data cleaning process...")
            
            cleaner = DataCleaner(
                data_dir=self.config['training_data_dir'],
                reusable_languages_with_ranges=reusable_languages_dict,
                output_reusable_samples=self.config.get('output_reusable_samples_path'),
                cleaned_data_dir=self.config['cleaned_data_dir']
            )
            cleaner.process()
            
            logger.info("Data cleaning completed")
        else:
            logger.info("2. Skipping data cleaning - cleaned data already exists")
        
        # Step 3: Train Tokenizer if needed
        tokenizer_files = [f for f in os.listdir(self.config['tokenizer_dir']) 
                          if f.endswith('.json') or f.endswith('.txt')]
        should_train = self.config.get('force_train', False) or not tokenizer_files
        
        if should_train:
            logger.info("3. Training new BPE tokenizer...")
            
            special_tokens_dict = {
                'additional_special_tokens': self.config.get('additional_special_tokens', [])
            }
            
            trainer = BPETokenizerTrainer(
                training_data_dir=self.config['cleaned_data_dir'],
                output_tokenizer_dir=self.config['tokenizer_dir'],
                vocab_size=total_reusable_count,
                model_id=self.config['model_id'],
                special_tokens_dict=special_tokens_dict,
            )
            trainer.run()
            
            logger.info("Tokenizer training completed")
        else:
            logger.info("3. Skipping tokenizer training - tokenizer files already exist")
        
        # Step 4: Create Mapping if needed
        mapping_exists = (os.path.exists(self.config['new_to_old_map_path']) and 
                        os.path.exists(self.config['old_to_new_map_path']) and 
                        os.path.exists(self.config['replacement_character_map_path']))
        should_map = self.config.get('force_mapping', False) or not mapping_exists
        
        if should_map:
            logger.info("4. Creating mapping tokenizer...")
            
            mapping_tokenizer = MappingTokenizer(
                new_tokenizer_path=self.config['tokenizer_dir'],
                old_tokenizer_model_id=self.config['model_id'],
                token_id_language_map_path=self.config['token_id_language_map_path'],
                reusable_languages=list(reusable_languages_dict.keys()),
                target_language_scripts_ranges=target_language_scripts_ranges,
                cache_dir=self.config['cache_dir'],
                new_to_old_map_path=self.config['new_to_old_map_path'],
                old_to_new_map_path=self.config['old_to_new_map_path'],
                replacement_character_map_path=self.config['replacement_character_map_path'],
                new_tokenizer_additional_special_tokens=self.config['additional_special_tokens'],
                save_maps=True,
            )
            
            logger.info("Mapping creation completed")
        else:
            logger.info("4. Skipping mapping creation - mapping files already exist")
            
            # Load mapping tokenizer from existing files
            mapping_tokenizer = MappingTokenizer(
                new_tokenizer_path=self.config['tokenizer_dir'],
                old_tokenizer_model_id=self.config['model_id'],
                token_id_language_map_path=self.config['token_id_language_map_path'],
                reusable_languages=list(reusable_languages_dict.keys()),
                target_language_scripts_ranges=target_language_scripts_ranges,
                cache_dir=self.config['cache_dir'],
                new_to_old_map_path=self.config['new_to_old_map_path'],
                old_to_new_map_path=self.config['old_to_new_map_path'],
                replacement_character_map_path=self.config['replacement_character_map_path'],
                new_tokenizer_additional_special_tokens=self.config['additional_special_tokens'],
                save_maps=False
            )
        
        # Step 5: Create final tokenizer
        logger.info("5. Creating final custom tokenizer...")
        base_tokenizer_class = AutoTokenizer.from_pretrained(self.config['model_id']).__class__
        dynamic_tokenizer_class = create_dynamic_tokenizer(base_tokenizer_class, mapping_tokenizer, self.config)
        
        self.tokenizer = dynamic_tokenizer_class(
            mapping_tokenizer=mapping_tokenizer,
            model_id=self.config['model_id']
        )
        
        logger.info("Tokenizer preparation completed successfully!")
        
        return self.tokenizer
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> PreTrainedTokenizerBase:
        """Load a prepared tokenizer from the given path"""
        config_path = os.path.join(pretrained_path, 'tokenizer_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        custom_tokenizer_config = config['custom_tokenizer_config']
            
        mapping_tokenizer = MappingTokenizer.from_json(config['mapping_tokenizer'])
        base_tokenizer_class = AutoTokenizer.from_pretrained(mapping_tokenizer.old_tokenizer_model_id).__class__
        dynamic_tokenizer_class = create_dynamic_tokenizer(base_tokenizer_class, mapping_tokenizer, custom_tokenizer_config)
        
        return dynamic_tokenizer_class(
            model_id=mapping_tokenizer.old_tokenizer_model_id,
            mapping_tokenizer=mapping_tokenizer,
            **kwargs
        )

def main():
    """CLI entry point for tokenizer preparation"""
    parser = argparse.ArgumentParser(description="Prepare a custom BPE tokenizer")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Add direct parameters as alternative to config file
    parser.add_argument('--model_id', type=str, help='Base tokenizer model ID')
    parser.add_argument('--output_dir', type=str, help='Output directory (defaults to rbpe_{model_name})')
    parser.add_argument('--training_data_dir', type=str, help='Training data directory')
    parser.add_argument('--cleaned_data_dir', type=str, help='Cleaned data directory')
    parser.add_argument('--hf_token', type=str, help='Hugging Face API token')
    parser.add_argument('--min_reusable_count', type=int, default=100000, help='Min reusable language token count')
    parser.add_argument('--force_clean', action='store_true', help='Force data cleaning')
    parser.add_argument('--force_train', action='store_true', help='Force tokenizer training')
    parser.add_argument('--force_mapping', action='store_true', help='Force mapping creation')
    
    # Special tokens
    parser.add_argument('--pad_token', type=str, help='Padding token')
    parser.add_argument('--unk_token', type=str, help='Unknown token')
    parser.add_argument('--bos_token', type=str, help='Beginning of sequence token')
    parser.add_argument('--eos_token', type=str, help='End of sequence token')
    parser.add_argument('--mask_token', type=str, help='Mask token')
    parser.add_argument('--sep_token', type=str, help='Separator token')
    parser.add_argument('--cls_token', type=str, help='Classification token')
    
    args = parser.parse_args()
    
    # Create tokenizer from either config or direct params
    if args.config:
        tokenizer_factory = RBPETokenizer.from_config(args.config)
    else:
        # Extract parameters from args
        kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
        
        # Ensure required parameters are provided
        required = ['model_id', 'training_data_dir', 'hf_token']
        missing = [r for r in required if r not in kwargs or kwargs[r] is None]
        if missing:
            parser.error(f"Missing required parameters: {', '.join(missing)}")
        
        tokenizer_factory = RBPETokenizer.from_params(**kwargs)
    
    # Prepare the tokenizer
    tokenizer = tokenizer_factory.prepare()
    
    # Interactive testing mode
    logger.info("\nTokenizer preparation complete! Entering interactive testing mode.")
    logger.info("Type 'q' to quit, or enter text to encode and decode.")
    
    while True:
        user_input = input("\nEnter text to encode and decode (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        encoded = tokenizer(
            user_input
        )
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        decoded = tokenizer.decode(encoded['input_ids'])
        
        logger.info("Encoded: %s", encoded)
        logger.info("Tokens: %s", tokens)
        logger.info("Decoded: %s", decoded)

if __name__ == "__main__":
    main()
