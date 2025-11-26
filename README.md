# R-BPE: Improving BPE-Tokenizers with Token Reuse

This repository accompanies <a href="https://aclanthology.org/2025.emnlp-main.1169/" target="_blank">the paper introducing R-BPE</a>, a lightweight framework for adapting existing Byte-Pair Encoding (BPE) tokenizers to better support a specified target language. The method is demonstrated using Arabic as the target language. R-BPE reuses tokens from user-excluded languages and creates ID-based maps to resolve the new tokens of the chosen language. It is compatible with HuggingFace interfaces and thereby readily applicable to a wide range of existing models.

## Overview
The `RBPETokenizer` orchestrates the entire process of:
1. Classifying vocabulary tokens languages via `TokenClassifier`.
2. Cleaning training data using `DataCleaner`.
3. Training a new BPE tokenizer with `BPETokenizerTrainer`.
4. Creating mappings between the original and new tokenizer with `MappingTokenizer`.
5. Returning a final `RBPETokenizer` adapted to the target language.

## Prerequisites

### Installation from GitHub

#### Using pip
```bash
pip install rbpe
```

#### Using uv
```bash
uv add rbpe 
```

### Installation from Local Directory

#### Using pip

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package:
```bash
pip install .
```

#### Using uv

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install the package:
```bash
uv venv venv
source venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:
```bash
uv sync
```

## Creating an R-BPE Tokenizer

You can create an R-BPE tokenizer either through the command-line interface (CLI) or programmaticaly through the Python API.

#### Configuration Parameters

R-BPE uses the following configuration parameters:

| Parameter | Meaning | Necessity | Default Value|
|-----|-------|-------| -------|
| model_id | The HuggingFace model id of the original tokenizer. e.g. `meta-llama/Llama-3.1-8B` | Required | None |
| training_data_dir | The directory where the training data for the new tokenizer is stored. | Required | None |
| clean_data| Whether to clean the training data or not. Warning: only set to false if you are sure that your training data does not include any non-preserved languages. | Required | True |
| cleaned_data_dir | The directory where the cleaned training data for the new tokenizer should be saved. | Optional | None |
| hf_token | The HuggingFace access token. | Required | None |
| min_reusable_count | The minimum number of tokens needed for reuse (threshold ***_h_*** in the paper). The size of the new tokenizer vocabulary will be <= `min_reusable_count` depending on how many reusable tokens are found in the specified original tokenizer. | Optional | 20000 |
| target_language_scripts | List of the unicode script names or aliases of the target language. See [this](unicode_scripts.md) table for possible values. | Optional | Arabic |
| preserved_languages_scripts | List of the unicode script names or aliases of the languages that must be preserved. The target language scripts are preserved by default. See [this](unicode_scripts.md) table for possible values. | Optional | Latin, Greek |
| special_tokens | Dictionary of custom special tokens values for the main special tokens: `pad_token`, `unk_token`, `bos_token`, `mask_token`, `sep_token`, `cls_token`. | Optional | None |
| additional_special_tokens | List of additional special tokens the _new_ tokenizer will have. | Optional | None |
| apply_rbpe_arabic_norm | Whether to apply the R-BPE Arabic normalization during encoding or not. | optional | True |

#### Using the CLI

You have to supply `output_dir` which is the path where the created `RBPETokenizer` should be saved.

```bash
rbpe create-tokenizer --config path/to/config.yaml --output_dir path/to/tokenizer_output_dir
```
or 

```bash
rbpe create-tokenizer --output_dir path/to/tokenizer_output_dir --model_id meta-llama/Llama-3.1-8B --output_dir ./rbpe_tokenizer --training_data_dir ./data --hf_token YOUR_TOKEN
```

#### Using the Python API

```python
from rbpe import RBPETokenizer

# From a YAML config file
tokenizer_factory = RBPETokenizer.from_config('path/to/config.yaml')

# Or with explicit parameters
tokenizer_factory = RBPETokenizer(
    model_id='meta-llama/Llama-3.1-8B',
    training_data_dir='./data',
    cleaned_data_dir='./data_cleaned',
    target_language_scripts=['arabic'],
    preserved_languages_scripts=['latin', 'greek'],
)

# Prepare the tokenizer
tokenizer = tokenizer_factory.prepare()

# You can directly use the tokenizer now

# Save the prepared R-BPE tokenizer for future use
tokenizer.save_pretrained('./rbpe_llama3_8b_tokenizer')
```

## Using an R-BPE tokenizer

Once you have created your R-BPE tokenizer, you can use it the same way you use any HuggingFace tokenizer:

```python
from rbpe import RBPETokenizer

tokenizer = RBPETokenizer.from_pretrained('./rbpe_llama3_8b_tokenizer')

text = 'مرحبا'
encoded = tokenizer(text)
decoded = tokenizer.decode(encoded['input_ids'])

print('Encoded:', encoded)
print('Decoded:', decoded)
```

## Citation

If you use R-BPE, please cite:
```bibtex
@inproceedings{hamdan-etal-2025-r,
    title = "{R}-{BPE}: Improving {BPE}-Tokenizers with Token Reuse",
    author = "Hamdan, Nancy  and
      Rakan Al Mraikhat, Osama  and
      Zaraket, Fadi A.",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1169/",
    doi = "10.18653/v1/2025.emnlp-main.1169",
    pages = "22951--22959",
    ISBN = "979-8-89176-332-6",
    abstract = "This paper presents R-BPE, a lightweight framework for adapting existing Byte-Pair Encoding (BPE) tokenizers to better support a specified target language. It reuses tokens from user-excluded languages and creates ID-based maps to resolve the new tokens of the chosen language. We evaluate R-BPE on Arabic as a target language. R-BPE reduced subword fertility by an average of 24.4{\%} across the LLaMA 3.1 8B, Command R 35B, and Qwen 3 8B models. Applied to LLaMA 3.1 8B in continued pretraining mode, R-BPE yields a 7.33{\%} reduction in training time. On the ArabicMMLU benchmark, the resulting model improved by 5.09 points on five in-domain topics and matched the original model{'}s overall performance. It also preserved performance on EnglishMMLU. R-BPE effectively leverages existing models' tokenizers, embedding layers, and performance to better support target languages without incurring model size changes. We release an R-BPE implementation that is compatible with HuggingFace interfaces and thereby readily applicable to a wide range of existing models at \url{https://acr.ps/1L9GPmL}."
}
```



