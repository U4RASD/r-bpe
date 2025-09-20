import os
import yaml
from datasets import concatenate_datasets, load_from_disk
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    normalizers
)
from transformers import AutoTokenizer
from .token_classifier import TokenClassifier
from huggingface_hub import login
import logging

logger = logging.getLogger('BPE')

class BPETokenizerTrainer:
    def __init__(self, training_data_dir, output_tokenizer_dir, vocab_size, model_id, special_tokens_dict):
        self.training_data_dir = training_data_dir
        self.output_tokenizer_dir = output_tokenizer_dir
        self.vocab_size = vocab_size
        self.model_id = model_id
        self.special_tokens_dict = special_tokens_dict

    def _load_datasets(self):
        datasets = []
        if os.path.isfile(os.path.join(self.training_data_dir, "dataset_info.json")):
            dataset = load_from_disk(self.training_data_dir)
            datasets.extend(dataset.values() if isinstance(dataset, dict) else [dataset])
        else:
            for dataset_name in os.listdir(self.training_data_dir):
                input_dir = os.path.join(self.training_data_dir, dataset_name)
                if os.path.isdir(input_dir):
                    dataset = load_from_disk(input_dir)
                    datasets.extend(dataset.values() if isinstance(dataset, dict) else [dataset])
        return concatenate_datasets(datasets)

    def _get_text_generator(self):
        combined_data = self._load_datasets()
        return (example["text"] for example in combined_data)

    def _train_tokenizer(self, texts):
        special_tokens = self._get_special_tokens()
        tokenizer = self._initialize_tokenizer()
        trainer = self._initialize_trainer(special_tokens)

        try:
            tokenizer.train_from_iterator(texts, trainer=trainer)
            logger.info("Tokenizer training completed.")
        except Exception as e:
            logger.error(f"Failed to train tokenizer: {e}")
            raise ValueError(f"Failed to train tokenizer: {e}")

        return self._wrap_tokenizer(tokenizer)

    def _get_special_tokens(self):
        special_tokens = []
        if self.special_tokens_dict:
            for key, token in self.special_tokens_dict.items():
                if token is not None and token != "None" and token != ["None"]:
                    if isinstance(token, list):
                        special_tokens.extend([t for t in token if t != "None"])
                    else:
                        special_tokens.append(token)
        return special_tokens

    def _initialize_tokenizer(self):
        bpe_args = {"byte_fallback": True}
        if self.special_tokens_dict and self.special_tokens_dict.get('unk_token') is not None and self.special_tokens_dict.get('unk_token') != "None":
            bpe_args["unk_token"] = self.special_tokens_dict['unk_token']

        tokenizer = Tokenizer(models.BPE(**bpe_args))
        
        # Use the exact GPT-4 regex pattern
        gpt4_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=gpt4_pattern,
                behavior='isolated'
            ),
            # Keep ByteLevel as fallback for any unmatched characters
            pre_tokenizers.ByteLevel(add_prefix_space=False)
        ])
        
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    def _initialize_trainer(self, special_tokens):
        adjusted_vocab_size = self.vocab_size + len(special_tokens)
        return trainers.BpeTrainer(
            vocab_size=adjusted_vocab_size,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

    def _wrap_tokenizer(self, tokenizer):
        try:
            tokenizer_class = AutoTokenizer.from_pretrained(self.model_id).__class__ if self.model_id else None
            tokenizer_kwargs = {key: self.special_tokens_dict[key] for key in ['unk_token', 'bos_token', 'eos_token', 'pad_token', 'additional_special_tokens'] 
                              if self.special_tokens_dict.get(key) is not None 
                              and self.special_tokens_dict.get(key) != "None"
                              and self.special_tokens_dict.get(key) != ["None"]}
            tokenizer = tokenizer_class(tokenizer_object=tokenizer, **tokenizer_kwargs)
            logger.debug(f"Tokenizer vocabulary size: {len(tokenizer)}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to create tokenizer: {e}")
            raise ValueError(f"Failed to create tokenizer: {e}")

    def _save_and_load_tokenizer(self, tokenizer):
        try:
            tokenizer.save_pretrained(self.output_tokenizer_dir)
            logger.info(f"Tokenizer saved to {self.output_tokenizer_dir}")

            loaded_tokenizer = AutoTokenizer.from_pretrained(self.output_tokenizer_dir)
            logger.debug("Tokenizer loaded successfully")
            return loaded_tokenizer
        except Exception as e:
            logger.error(f"Error saving or loading tokenizer: {e}")
            raise ValueError(f"Error saving or loading tokenizer: {e}")

    def _analyze_tokenizer(self, tokenizer, example_text="Hello, how are you today?"):
        try:
            vocab = tokenizer.get_vocab()
            vocab_size = len(vocab)
            first_10_tokens = list(vocab.keys())[:10]
            logger.debug(f"Vocabulary size: {vocab_size}")
            logger.debug(f"First 10 tokens: {first_10_tokens}")

            encoded = tokenizer.encode(example_text)
            decoded = tokenizer.decode(encoded)
            tokens = tokenizer.tokenize(example_text)

            logger.debug(f"Example text: {example_text}")
            logger.debug(f"Encoded IDs: {encoded}")
            logger.debug(f"Decoded text: {decoded}")
            logger.debug(f"Tokens: {tokens}")

        except Exception as e:
            logger.error(f"An error occurred during tokenizer analysis: {e}")
            raise ValueError(f"An error occurred during tokenizer analysis: {e}")

    def run(self):
        try:
            texts = self._get_text_generator()

            trained_tokenizer = self._train_tokenizer(texts)
            if trained_tokenizer is None:
                error_msg = "Tokenizer training failed."
                logger.error(error_msg)
                raise ValueError(error_msg)

            loaded_tokenizer = self._save_and_load_tokenizer(trained_tokenizer)
            if loaded_tokenizer is None:
                error_msg = "Failed to save and load the tokenizer."
                logger.error(error_msg)
                raise ValueError(error_msg)

            self._analyze_tokenizer(loaded_tokenizer)

            logger.info("Tokenizer training and testing completed successfully.")

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise ValueError(f"An unexpected error occurred: {e}")
