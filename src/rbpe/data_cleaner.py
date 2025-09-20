import os
import json
from tqdm import tqdm
import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from .token_classifier import TokenClassifier
from huggingface_hub import login
import logging

logger = logging.getLogger('BPE')

class DataCleaner:
    def __init__(
        self,
        data_dir: str,
        reusable_languages_with_ranges: dict,
        output_reusable_samples: str = None,
        cleaned_data_dir: str = None
    ):
        """
        Initializes the DataCleaner with the dataset and reusable languages configurations.

        Args:
            data_dir (str): Path to the dataset directory.
            reusable_languages_with_ranges (dict): Dictionary containing reusable languages and their character ranges.
            output_reusable_samples (str, optional): Path to save samples containing reusable languages.
            cleaned_data_dir (str, optional): Directory to save the cleaned dataset.
        """
        logger.info(f"Initializing DataCleaner with {len(reusable_languages_with_ranges)} reusable languages")
        self.dataset = self.load_dataset(data_dir)
        self.language_ranges = reusable_languages_with_ranges
        self.reusable_file = self._initialize_reusable_file(output_reusable_samples)
        self.cleaned_data_dir = cleaned_data_dir

    def _initialize_reusable_file(self, output_path: str):
        """Initializes the samples containing reusable languages file if a path is provided."""
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logger.debug(f"Initializing samples containing reusable languages file at: {output_path}")
            return open(output_path, 'w', encoding='utf-8')
        logger.debug("No samples containing reusable languages file path provided, skipping reusable sample logging")
        return None

    def detect_languages(self, text: str) -> set:
        """
        Detects possible languages in the given text based on character ranges.

        Args:
            text (str): The text to analyze.

        Returns:
            set: A set of detected language names.
        """
        text_chars = set(ord(c) for c in text)
        matching_languages = set()
        for language, ranges in self.language_ranges.items():
            for start, end in ranges:
                if any(start <= char <= end for char in text_chars):
                    matching_languages.add(language)
                    break
        return matching_languages

    def load_dataset(self, data_dir: str):
        """
        Loads the dataset from the specified directory.

        Args:
            data_dir (str): Path to the dataset directory.

        Returns:
            Dataset or DatasetDict: The loaded dataset.
        """
        try:
            dataset = load_from_disk(data_dir)
            if isinstance(dataset, DatasetDict):
                logger.debug(f"Loaded dataset with splits: {list(dataset.keys())}")
                for split_name, split_dataset in dataset.items():
                    logger.debug(f"Split '{split_name}' has {len(split_dataset)} samples")
            else:
                logger.debug(f"Loaded single dataset with {len(dataset)} samples")
            return dataset
        except Exception as e:
            raise ValueError(f"Could not load dataset from {data_dir}: {str(e)}")

    def process(self):
        """
        Processes the dataset by filtering out samples containing reusable languages.
        Saves the filtered dataset and optionally logs samples containing reusable languages.
        """
        try:
            if isinstance(self.dataset, DatasetDict):
                self._process_with_splits()
            else:
                self._process_single_split()
        finally:
            if self.reusable_file:
                self.reusable_file.close()

        if self.cleaned_data_dir:
            os.makedirs(self.cleaned_data_dir, exist_ok=True)
            logger.info(f"\nSaving filtered dataset to: {self.cleaned_data_dir}")
            self.filtered_dataset.save_to_disk(self.cleaned_data_dir)
            logger.debug("Dataset saved successfully!")

    def _process_with_splits(self):
        """Processes each split in the dataset individually."""
        filtered_dataset = DatasetDict()
        total_reusable_count = 0
        total_samples = sum(len(split) for split in self.dataset.values())

        for split_name, split_dataset in self.dataset.items():
            filtered_samples = []
            reusable_count = 0

            logger.info(f"\nProcessing split: {split_name}")
            # Check if this is an SFT dataset by looking for 'messages' column
            is_sft = 'messages' in split_dataset.features and 'text' not in split_dataset.features

            for idx, sample in tqdm(
                enumerate(split_dataset),
                desc=f"Processing {split_name} split",
                total=len(split_dataset)
            ):
                if is_sft:
                    # Process SFT dataset
                    has_reusable_lang = False
                    for turn in sample['messages']:
                        text = turn['content']
                        detected_langs = self.detect_languages(text)
                        if detected_langs.intersection(self.language_ranges.keys()):
                            has_reusable_lang = True
                            break
                    
                    if has_reusable_lang:
                        reusable_count += 1
                        if self.reusable_file:
                            self._log_reusable_sample(split_name, idx, detected_langs, str(sample['messages']))
                    else:
                        filtered_samples.append(sample)
                else:
                    # Process regular dataset
                    text = sample['text']
                    detected_langs = self.detect_languages(text)
                    
                    if detected_langs.intersection(self.language_ranges.keys()):
                        reusable_count += 1
                        if self.reusable_file:
                            self._log_reusable_sample(split_name, idx, detected_langs, text)
                    else:
                        filtered_samples.append(sample)

            filtered_dataset[split_name] = Dataset.from_list(filtered_samples)
            total_reusable_count += reusable_count

            logger.info(f"Split '{split_name}' filtering complete:")
            logger.info(f"- Original samples: {len(split_dataset)}")
            logger.info(f"- Samples with reusable languages: {reusable_count}")
            logger.info(f"- Remaining samples: {len(filtered_samples)}")

        logger.info(f"\nOverall filtering complete:")
        logger.info(f"- Total original samples: {total_samples}")
        logger.info(f"- Total samples with reusable languages: {total_reusable_count}")
        remaining = sum(len(split) for split in filtered_dataset.values())
        logger.info(f"- Total remaining samples: {remaining}")

        self.filtered_dataset = filtered_dataset

    def _process_single_split(self):
        """Processes a single split dataset."""
        filtered_samples = []
        reusable_count = 0
        total_samples = len(self.dataset)

        # Check if this is an SFT dataset by looking for 'messages' column
        is_sft = 'messages' in self.dataset.features and 'text' not in self.dataset.features

        logger.info(f"Starting with {total_samples} total samples")
        for idx, sample in tqdm(
            enumerate(self.dataset),
            desc="Processing dataset",
            total=total_samples
        ):
            if is_sft:
                # Process SFT dataset
                has_reusable_lang = False
                for turn in sample['messages']:
                    text = turn['content']
                    detected_langs = self.detect_languages(text)
                    if detected_langs.intersection(self.language_ranges.keys()):
                        has_reusable_lang = True
                        break
                
                if has_reusable_lang:
                    reusable_count += 1
                    if self.reusable_file:
                        self._log_reusable_sample(None, idx, detected_langs, str(sample['messages']))
                else:
                    filtered_samples.append(sample)
            else:
                # Process regular dataset
                text = sample['text']
                detected_langs = self.detect_languages(text)
                
                if detected_langs.intersection(self.language_ranges.keys()):
                    reusable_count += 1
                    if self.reusable_file:
                        self._log_reusable_sample(None, idx, detected_langs, text)
                else:
                    filtered_samples.append(sample)

        self.filtered_dataset = Dataset.from_list(filtered_samples)

        logger.info(f"\nFiltering complete:")
        logger.info(f"- Original samples: {total_samples}")
        logger.info(f"- Samples with reusable languages: {reusable_count}")
        logger.info(f"- Remaining samples: {len(filtered_samples)}")

    def _log_reusable_sample(self, split_name, idx, detected_langs, text):
        """Logs details of samples containing reusable languages."""
        if split_name:
            self.reusable_file.write(f"Split: {split_name}\n")
        self.reusable_file.write(f"Index: {idx}\n")
        self.reusable_file.write(f"Detected Languages: {', '.join(detected_langs)}\n")
        self.reusable_file.write(f"Text: {text}\n")
        self.reusable_file.write("-" * 80 + "\n")
        self.reusable_file.flush()
