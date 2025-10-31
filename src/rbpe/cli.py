import click
import json
from rbpe import RBPETokenizer, TokenClassifier

@click.group()
def rbpe():
    "Command-line client for the R-BPE package."
    pass

@rbpe.command()
@click.option("--config", help="The R-BPE configuration file")
@click.option("--output_dir", help="The directory to save the tokenizer", required=True)
@click.option("--model_id", help="The HuggingFace model ID")
@click.option("--training_data_dir", help="The directory containing the training data")
@click.option("--clean_data", help="Whether to clean the training data", default=True, type=click.BOOL)
@click.option("--cleaned_data_dir", help="The directory containing the cleaned data", default=None)
@click.option("--hf_token", help="The HuggingFace API token")
@click.option("--min_reusable_count", help="The minimum number of reusable tokens", default=20000)
@click.option("--target_language_scripts", help="The target language scripts (comma-separated)", default="arabic")
@click.option("--preserved_languages_scripts", help="The preserved languages scripts (comma-separated)", default="latin,greek")
@click.option("--apply_rbpe_arabic_norm", help="Whether to apply the R-BPE Arabic normalization during encoding", default=True)
@click.option("--special_tokens", help="Custom special tokens as JSON string, e.g. '{\"pad_token\": \"[PAD]\", \"unk_token\": \"[UNK]\"}'", default=None)
@click.option("--additional_special_tokens", help="Additional special tokens (comma-separated)", default=None)
def create_tokenizer(config, model_id, output_dir, training_data_dir, clean_data, cleaned_data_dir, hf_token, min_reusable_count, 
            target_language_scripts=None, preserved_languages_scripts=None, apply_rbpe_arabic_norm=None, special_tokens=None, additional_special_tokens=None):
    "Create an R-BPE tokenizer."
    if target_language_scripts:
        target_language_scripts = [s.strip() for s in target_language_scripts.split(',')]
    if preserved_languages_scripts:
        preserved_languages_scripts = [s.strip() for s in preserved_languages_scripts.split(',')]
    
    if special_tokens:
        try:
            special_tokens = json.loads(special_tokens)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"special_tokens must be valid JSON: {e}")
    else:
        special_tokens = {}
    
    if additional_special_tokens:
        additional_special_tokens = [s.strip() for s in additional_special_tokens.split(',')]
    else:
        additional_special_tokens = []
    
    if config:
        tokenizer_factory = RBPETokenizer.from_config(config)
    else:
        if not hf_token:
            raise click.BadParameter("hf_token is required")
        if not model_id:
            raise click.BadParameter("model_id is required")
        if not training_data_dir:
            raise click.BadParameter("training_data_dir is required")
        
        tokenizer_factory = RBPETokenizer(model_id=model_id, 
                                          training_data_dir=training_data_dir, 
                                          clean_data=clean_data, 
                                          cleaned_data_dir=cleaned_data_dir, 
                                          hf_token=hf_token, 
                                          min_reusable_count=min_reusable_count, 
                                          target_language_scripts=target_language_scripts, 
                                          preserved_languages_scripts=preserved_languages_scripts, 
                                          apply_rbpe_arabic_norm=apply_rbpe_arabic_norm, 
                                          special_tokens=special_tokens, 
                                          additional_special_tokens=additional_special_tokens)

    tokenizer = tokenizer_factory.prepare()
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")


@rbpe.command()
@click.option("--model_id", help="The HuggingFace model ID", required=True)
@click.option("--min_reusable_count", help="The minimum number of reusable tokens", default=20000)
@click.option("--target_language_scripts", help="The target language scripts (comma-separated)", default="arabic")
@click.option("--preserved_languages_scripts", help="The preserved languages scripts (comma-separated)", default="latin,greek")
@click.option("--hf_token", help="The HuggingFace API token", required=True)
def classify_tokens(model_id, min_reusable_count, target_language_scripts, preserved_languages_scripts, hf_token):
    "Get Language Classifications of a Tokenizer's Vocabulary."
    if target_language_scripts:
        target_language_scripts = [s.strip() for s in target_language_scripts.split(',')]
    if preserved_languages_scripts:
        preserved_languages_scripts = [s.strip() for s in preserved_languages_scripts.split(',')]
    
    token_classifier = TokenClassifier(min_reusable_ids=min_reusable_count, 
                                old_tokenizer_model_id=model_id, 
                                target_language_scripts=target_language_scripts, 
                                preserved_languages_scripts=preserved_languages_scripts, 
                                hf_api_key=hf_token)
    _, _ = token_classifier.get_reusable_languages_and_count()
    vocabulary_languages = sorted(token_classifier.all_languages_data, key=lambda x: x[1], reverse=False)
    result_dict = {}
    for language, id_count in vocabulary_languages:
        result_dict[language] = id_count
    print(json.dumps(result_dict, indent=4))
