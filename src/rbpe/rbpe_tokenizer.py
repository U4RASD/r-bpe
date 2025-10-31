from typing import List, Optional, Union
from transformers import PreTrainedTokenizerBase, AutoTokenizer

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
from .dynamic_tokenizer import create_dynamic_tokenizer
import os
import json
import yaml
import argparse

from huggingface_hub import login

from .logger_config import setup_logger

logger = setup_logger('BPE')


class RBPETokenizer:
    """Factory class to create and prepare a custom R-BPE tokenizer with minimal configuration requirements."""
    def __init__(self, model_id, training_data_dir, clean_data=True, cleaned_data_dir=None,
                         hf_token=None, min_reusable_count=20000, target_language_scripts=['arabic'], preserved_languages_scripts=['latin', 'greek'],
                         special_tokens={}, additional_special_tokens=[], apply_rbpe_arabic_norm=True):
        """Initialize an R-BPE tokenizer from parameters.
        
        Args:
            model_id (str): The HuggingFace model id of the original tokenizer.
            training_data_dir (str): The directory where the training data for the new tokenizer is stored.
            clean_data (bool): Whether to clean the training data or not.
            cleaned_data_dir (str): The directory where the cleaned training data for the new tokenizer should be saved. Optional, will only process in memory if not provided.
            hf_token (str): The HuggingFace access token.
            min_reusable_count (int): The minimum number of tokens needed for reuse (threshold ***_h_*** in the paper).
            target_language_scripts (list): The list of the unicode script names or aliases of the target language.
            preserved_languages_scripts (list): the unicode script names or aliases of the languages that must be preserved.
            special_tokens (dict): The dictionary of custom special tokens values for the main special tokens: pad_token, unk_token, bos_token, mask_token, sep_token, cls_token.
            additional_special_tokens (list): The list of additional special tokens the new tokenizer will have.
            apply_rbpe_arabic_norm (bool): Whether to apply the R-BPE Arabic normalization during encoding or not.
        """
        
        self.token_classifier = None
        self.tokenizer = None
        self.old_tokenizer = None
        self.new_tokenizer = None
        self.mapping_tokenizer = None
        self.reusable_languages_dict = None
        self.target_language_scripts_ranges = None

        # Validate required parameters
        if not model_id:
            raise ValueError("model_id is required")
        self.model_id = model_id
        if not training_data_dir:
            raise ValueError("training_data_dir is required to train the new tokenizer")
        if clean_data and not cleaned_data_dir:
            logger.warning("cleaned_data_dir was not provided. Cleaned data will not be saved to disk.")
        self.training_data_dir = training_data_dir
        self.clean_data = clean_data
        self.cleaned_data_dir = cleaned_data_dir if clean_data else None

        if not hf_token:
            raise ValueError("hf_token is required to log in to Hugging Face Hub")
        self.hf_token = hf_token
        self.min_reusable_count = min_reusable_count
        self.target_language_scripts = target_language_scripts
        self.preserved_languages_scripts = preserved_languages_scripts
        self.special_tokens = special_tokens
        self.additional_special_tokens = additional_special_tokens
        self.apply_rbpe_arabic_norm = apply_rbpe_arabic_norm
        
        # Login to HF
        try:
            login(token=hf_token)
            logger.debug("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            error_msg = f"Failed to log in to Hugging Face Hub: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @classmethod
    def from_config(cls, config_path: str):
        """Initialize an R-BPE tokenizer from a YAML config file.
        
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
        training_data_dir = config.get('training_data_dir')
        clean_data = config.get('clean_data', True)
        cleaned_data_dir = config.get('cleaned_data_dir')
        # TODO: make hf_token an environment variable!
        hf_token = config.get('hf_token')
        min_reusable_count = config.get('min_reusable_count', 20000)
        target_language_scripts = config.get('target_language_scripts', [])
        preserved_languages_scripts = config.get('preserved_languages_scripts', [])
        apply_rbpe_arabic_norm = config.get('apply_rbpe_arabic_norm', True)
        
        # Extract special tokens
        special_tokens = config.get('special_tokens', {})
        additional_special_tokens = config.get('additional_special_tokens', [])
        
        # Create instance with parameters
        instance = cls.__new__(cls)
        instance.__init__(
            model_id=model_id,
            training_data_dir=training_data_dir,
            clean_data=clean_data,
            cleaned_data_dir=cleaned_data_dir,
            hf_token=hf_token,
            min_reusable_count=min_reusable_count,
            target_language_scripts=target_language_scripts,
            preserved_languages_scripts=preserved_languages_scripts,
            apply_rbpe_arabic_norm=apply_rbpe_arabic_norm,
            special_tokens=special_tokens,
            additional_special_tokens=additional_special_tokens,
        )
        return instance
    
    def prepare(self) -> PreTrainedTokenizerBase:
        """
        Orchestrates the complete tokenizer preparation process:
        1. Classifies tokens using TokenClassifier
        2. Cleans data using DataCleaner (if needed)
        3. Trains new tokenizer using BPETokenizerTrainer
        4. Creates mappings using MappingTokenizer
        5. Returns final RBPETokenizer instance
        
        Returns:
            PreTrainedTokenizerBase: The prepared tokenizer
        """
        logger.info("Starting tokenizer preparation process...")

        # Token Classification
        logger.info("Initializing TokenClassifier...")
        self.token_classifier = TokenClassifier(
            min_reusable_ids=self.min_reusable_count,
            target_language_scripts=self.target_language_scripts,
            preserved_languages_scripts=self.preserved_languages_scripts,
            old_tokenizer_model_id=self.model_id,
            hf_api_key=self.hf_token,
        )

        # Get reusable languages and ranges
        self.reusable_languages_dict, total_reusable_count = self.token_classifier.get_reusable_languages_and_count()
        self.target_language_scripts_ranges = self.token_classifier.get_target_language_scripts_ranges()
        
        cleaned_dataset = None
        if self.clean_data:
            # Clean Data
            logger.info("Starting data cleaning process...")
            
            cleaner = DataCleaner(
                data_dir=self.training_data_dir,
                reusable_languages_with_ranges=self.reusable_languages_dict,
                cleaned_data_dir=self.cleaned_data_dir
            )
            cleaned_dataset = cleaner.process()
            logger.info("Data cleaning completed")
        
        # Train the new tokenizer
        logger.info("Training new BPE tokenizer...")
        
        special_tokens_dict = {
            'additional_special_tokens': self.additional_special_tokens
        }
        
        if cleaned_dataset:
            trainer = BPETokenizerTrainer(
                dataset=cleaned_dataset,
                vocab_size=total_reusable_count,
                model_id=self.model_id,
                special_tokens_dict=special_tokens_dict,
            )
        else:
            trainer = BPETokenizerTrainer(
                dataset_dir=self.training_data_dir,
                vocab_size=total_reusable_count,
                model_id=self.model_id,
                special_tokens_dict=special_tokens_dict,
            )
        self.new_tokenizer = trainer.run()
        
        logger.info("Tokenizer training completed successfully.")
        
        # TODO: update var names to match paper
        # Create mapping layer
        logger.info("Creating mapping tokenizer...")
        
        self.old_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.mapping_tokenizer = MappingTokenizer(
            new_tokenizer=self.new_tokenizer,
            old_tokenizer=self.old_tokenizer,
            token_id_language_map=self.token_classifier.classified_ids_with_ranges,
            reusable_languages=list(self.reusable_languages_dict.keys()),
            target_language_scripts_ranges=self.target_language_scripts_ranges,
            new_tokenizer_additional_special_tokens=self.additional_special_tokens,
            apply_normalization=self.apply_rbpe_arabic_norm,
        )
        
        logger.info("Mapping creation completed successfully.")
        
        # Prepare config for dynamic tokenizer
        dynamic_tokenizer_config = {
            'model_id': self.model_id,
            'old_to_new_map': self.mapping_tokenizer.old_to_new_map,
            'new_to_old_map': self.mapping_tokenizer.new_to_old_map,
            'replacement_character_map': self.mapping_tokenizer.replacement_character_map,
            'reusable_languages': list(self.reusable_languages_dict.keys()),
            'target_language_scripts_ranges': self.target_language_scripts_ranges,
            'token_id_language_map': self.token_classifier.classified_ids_with_ranges,
            'token_text_language_map': self.token_classifier.classified_tokens_with_ranges,
            'vocabulary_languages': self.token_classifier.all_languages_data
        }

        dynamic_tokenizer_config.update(self.special_tokens)
        
        # Dynamically create the final R-BPE tokenizer based on the original tokenizer's HuggingFace class
        logger.info("Creating final custom tokenizer...")
        base_tokenizer_class = AutoTokenizer.from_pretrained(self.model_id).__class__
        dynamic_tokenizer_class = create_dynamic_tokenizer(base_tokenizer_class, self.mapping_tokenizer, dynamic_tokenizer_config)
        
        self.tokenizer = dynamic_tokenizer_class(
            mapping_tokenizer=self.mapping_tokenizer,
            model_id=self.model_id
        )
        
        logger.info("Tokenizer preparation completed successfully!")
        
        return self.tokenizer
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> PreTrainedTokenizerBase:
        config_path = os.path.join(pretrained_path, 'tokenizer_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        custom_tokenizer_config = config['custom_tokenizer_config']
        mapping_tokenizer_state = config['mapping_tokenizer']
        mapping_tokenizer_dict = json.loads(mapping_tokenizer_state)

        # Rebuild MappingTokenizer from JSON, but override its paths to the bundled ones
        mapping_tokenizer_dict['new_tokenizer_path'] = os.path.join(pretrained_path, 'new_tokenizer')
        mapping_tokenizer_dict['old_tokenizer_path'] = os.path.join(pretrained_path, 'old_tokenizer')
        mapping_tokenizer = MappingTokenizer.from_json(json.dumps(mapping_tokenizer_dict))

        base_tokenizer_class = AutoTokenizer.from_pretrained(custom_tokenizer_config['model_id']).__class__
        dynamic_tokenizer_class = create_dynamic_tokenizer(base_tokenizer_class, mapping_tokenizer, custom_tokenizer_config)

        return dynamic_tokenizer_class(
            model_id=custom_tokenizer_config['model_id'],
            mapping_tokenizer=mapping_tokenizer,
            **kwargs
        )
