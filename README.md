# R-BPE: Improving BPE-Tokenizers with Token Reuse

This repository accompanies the paper introducing R-BPE, a lightweight framework for adapting existing Byte-Pair Encoding (BPE) tokenizers to better support a specified target language. The method is demonstrated using Arabic as the target language. R-BPE reuses tokens from
user-excluded languages and creates ID-based maps to resolve the new tokens of the chosen language. It is compatible with HuggingFace interfaces and thereby readily applicable to a wide range of existing models.

## Overview
The RBPETokenizer orchestrates the entire process of:
1. Classifying tokens via TokenClassifier.
2. Cleaning data using DataCleaner.
3. Training a new BPE tokenizer with BPETokenizerTrainer.
4. Creating mappings between the old and new tokenizer with MappingTokenizer.
5. Returning a final R-BPE tokenizer that can be used for training models.

## Prerequisites

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install the package:
```bash
pip install .
```

3. Create a YAML config file with your desired parameters. An example config file is provided in the `inputs` directory as `tokenizer_config.yaml`.

## Configuration Format

The tokenizer uses the following YAML configuration format:

```yaml
model_id: CohereForAI/c4ai-command-r-v01  # Base tokenizer model ID from Hugging Face
output_dir: /path/to/output/directory     # Main output directory
training_data_dir: /path/to/training/data # Directory containing training data
cleaned_data_dir: /path/to/cleaned/data   # Where to store cleaned training data (optional)
hf_token: your_huggingface_token          # Hugging Face API token
min_reusable_count: 100000                  # Minimum token count for reusable IDs (threshold h)
preserved_langs:                           # Languages to exclude from reuse
  - arabic
  - latin
  - greek
special_tokens:                           # Custom special tokens (optional)
  pad_token: <PAD>
  unk_token: <UNK>
  bos_token: <BOS_TOKEN>
  eos_token: <EOS_TOKEN>
  mask_token: <MASK_TOKEN>
  sep_token: <SEP>
  cls_token: <CLS>
force:                                    # Force specific processing steps
  classify: true                          # Force Token classification even if output exists
  clean: true                             # Force data cleaning even if output exists
  train: true                             # Force tokenizer training even if trained
  mapping: true                           # Force mapping creation even if exists
```

## Output Directory Structure

The tokenizer automatically creates and maintains the following directory structure:

```
output_dir/
├── tokenizer/      # Trained tokenizer files
├── metadata/       # Token mapping and classification files
├── cache/          # Cache for downloaded models
```

## Using the Tokenizer

### Command Line Usage

You can create a tokenizer from the command line:

```bash
python -m rbpe.rbpe_tokenizer --config path/to/config.yaml
```

Or with explicit parameters:

```bash
python -m rbpe.rbpe_tokenizer --model_id CohereForAI/c4ai-command-r-v01 --output_dir ./outputs --training_data_dir ./data --force_clean --force_train
```

### Python API Usage

```python
from rbpe.rbpe_tokenizer import RBPETokenizer

# From a YAML config file
tokenizer_factory = RBPETokenizer.from_config('path/to/config.yaml')

# Or with explicit parameters
tokenizer_factory = RBPETokenizer.from_params(
    model_id='CohereForAI/c4ai-command-r-v01',
    training_data_dir='./data',
    force_clean=True,
    force_train=True
)

# Prepare the tokenizer
tokenizer = tokenizer_factory.prepare()

# Test the tokenizer
text = 'مرحبا'
encoded = tokenizer(text)
decoded = tokenizer.decode(encoded['input_ids'])

print('Encoded:', encoded)
print('Decoded:', decoded)
```

### Loading a Pretrained Tokenizer

Once your tokenizer is prepared, you can load it later:

```python
from rbpe.rbpe_tokenizer import RBPETokenizer

tokenizer = RBPETokenizer.from_pretrained('path/to/tokenizer_output_dir')
```
