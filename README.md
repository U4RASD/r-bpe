# R-BPE: Improving BPE-Tokenizers with Token Reuse

This repository accompanies the paper introducing R-BPE, a lightweight framework for adapting existing Byte-Pair Encoding (BPE) tokenizers to better support a specified target language. The method is demonstrated using Arabic as the target language. R-BPE reuses tokens from
user-excluded languages and creates ID-based maps to resolve the new tokens of the chosen language. It is compatible with HuggingFace interfaces and thereby readily applicable to a wide range of existing models.

## Overview
The RBPETokenizer orchestrates the entire process of:
1. Classifying tokens via TokenClassifier.
2. Cleaning data using DataCleaner.
3. Training a new BPE tokenizer with BPETokenizerTrainer.
4. Creating mappings between the original and new tokenizer with MappingTokenizer.
5. Returning a final R-BPE tokenizer adapted to the target language.

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

## Creating an R-BPE Tokenizer

You can create an R-BPE tokenizer either through the command-line interface (CLI) or programmaticaly through the Python API.

#### Configuration Parameters

R-BPE uses the following configuration parameters:

| Parameter | Meaning | Necessity | Default Value|
|-----|-------|-------| -------|
| model_id | The HuggingFace model id of the original tokenizer's model. | Required | None |
| output_dir | The output directory where the R-BPE tokenizer will be saved. | Optional | rbpe_{model_id} |
| training_data_dir | The directory where the training data for the new tokenizer is. | Required | None |
| cleaned_data_dir | The directory where the cleaned training data for the new tokenizer is, or should be saved (if it's empty). | Optional | {output_dir}/{cleaned_data} |
| min_reusable_count | Minimum number of tokens needed for reuse (threshold ***_h_*** in the paper). | Optional | 20000 |
| target_language_scripts | The unicode script names or aliases of the target language. | Optional | Arabic |
| preserved_languages_scripts | The unicode script names or aliases of the languages that must be preserved. The target language scripts are preserved by default. | Optional | Latin, Greek |
| special_tokens | Custom special tokens for the main special tokens like bos_token, eos_token, pad_token, etc. | Optional | None |
| additional_special_tokens | Any additional special tokens the _new_ tokenizer will have. | Optional | None |
| force_classify | Force classify vocabulary tokens. | Optional | False |
| force_clean | Force clean training data. | Optional | False |
| force_train | Force train a new tokenizer. | Optional | False |
| force_mapping | Force create the new to old and old to new maps. | Optional | False |

The current Unicode data R-BPE uses is the [Unicode 17](https://www.unicode.org/versions/Unicode17.0.0/) data. You can refer to [this](#specifying-language-scripts) table for all the language scripts you can specify.

#### Using the CLI

```bash
python -m rbpe.rbpe_tokenizer --config path/to/config.yaml
```
or 

```bash
python -m rbpe.rbpe_tokenizer --model_id CohereForAI/c4ai-command-r-v01 --output_dir ./rbpe_tokenizer --training_data_dir ./data  --cleaned_data_dir ./data_cleaned --hf_token YOUR_TOKEN
```

#### Using the Python API

```python
from rbpe.rbpe_tokenizer import RBPETokenizer

# From a YAML config file
tokenizer_factory = RBPETokenizer.from_config('path/to/config.yaml')

# Or with explicit parameters
tokenizer_factory = RBPETokenizer.from_params(
    model_id='CohereForAI/c4ai-command-r-v01',
    output_dir='./rbpe_tokenizer'
    training_data_dir='./data',
    cleaned_data_dir='./data_cleaned',
    target_language_scripts=['arabic'],
    preserved_languages_scripts=['latin', 'greek'],
    force_clean=True,
    force_train=True
)

# Prepare the tokenizer
tokenizer = tokenizer_factory.prepare()

# you can directly use the tokenizer now

# Save the prepared R-BPE tokenizer for future use
tokenizer.save_pretrained('./rbpe_tokenizer')
```

## R-BPE Output Structure

Creating an R-BPE tokenizer will result in the following output directory encapsulating the R-BPE tokenizer:

```
output_dir/
├── tokenizer/      # The new tokenizer
├── metadata/       # R-BPE tokenizer metadata including new to old and old to new maps.
├── cache/          # Cache for downloaded tokenizers.
```

## Using an R-BPE tokenizer

Once you have created your R-BPE tokenizer, you can use it the same way you use any HuggingFace tokenizer:

```python
from rbpe.rbpe_tokenizer import RBPETokenizer

tokenizer = RBPETokenizer.from_pretrained('path/to/tokenizer_output_dir')

text = 'مرحبا'
encoded = tokenizer(text)
decoded = tokenizer.decode(encoded['input_ids'])

print('Encoded:', encoded)
print('Decoded:', decoded)
```

## Specifying Language Scripts

Language script specification is case insensitive. The following table shows all possible values you can use which are derived from the [Unicode 17](https://www.unicode.org/versions/Unicode17.0.0/) data:

| Language Script Name | Alias |
|-----|-------|
| adlam | adlm |
| ahom | ahom |
| anatolian_hieroglyphs | hluw |
| arabic | arab |
| armenian | armn |
| avestan | avst |
| balinese | bali |
| bamum | bamu |
| bassa_vah | bass |
| batak | batk |
| bengali | beng |
| beria_erfe | berf |
| bhaiksuki | bhks |
| bopomofo | bopo |
| brahmi | brah |
| braille | brai |
| buginese | bugi |
| buhid | buhd |
| canadian_aboriginal | cans |
| carian | cari |
| caucasian_albanian | aghb |
| chakma | cakm |
| cham | cham |
| cherokee | cher |
| chorasmian | chrs |
| common | zyyy |
| coptic | copt |
| cuneiform | xsux |
| cypriot | cprt |
| cypro_minoan | cpmn |
| cyrillic | cyrl |
| deseret | dsrt |
| devanagari | deva |
| dives_akuru | diak |
| dogra | dogr |
| duployan | dupl |
| egyptian_hieroglyphs | egyp |
| elbasan | elba |
| elymaic | elym |
| ethiopic | ethi |
| garay | gara |
| georgian | geor |
| glagolitic | glag |
| gothic | goth |
| grantha | gran |
| greek | grek |
| gujarati | gujr |
| gunjala_gondi | gong |
| gurmukhi | guru |
| gurung_khema | gukh |
| han | hani |
| hangul | hang |
| hanifi_rohingya | rohg |
| hanunoo | hano |
| hatran | hatr |
| hebrew | hebr |
| hiragana | hira |
| imperial_aramaic | armi |
| inherited | zinh |
| inscriptional_pahlavi | phli |
| inscriptional_parthian | prti |
| javanese | java |
| kaithi | kthi |
| kannada | knda |
| katakana | kana |
| katakana_or_hiragana | hrkt |
| kawi | kawi |
| kayah_li | kali |
| kharoshthi | khar |
| khitan_small_script | kits |
| khmer | khmr |
| khojki | khoj |
| khudawadi | sind |
| kirat_rai | krai |
| lao | laoo |
| latin | latn |
| lepcha | lepc |
| limbu | limb |
| linear_a | lina |
| linear_b | linb |
| lisu | lisu |
| lycian | lyci |
| lydian | lydi |
| mahajani | mahj |
| makasar | maka |
| malayalam | mlym |
| mandaic | mand |
| manichaean | mani |
| marchen | marc |
| masaram_gondi | gonm |
| medefaidrin | medf |
| meetei_mayek | mtei |
| mende_kikakui | mend |
| meroitic_cursive | merc |
| meroitic_hieroglyphs | mero |
| miao | plrd |
| modi | modi |
| mongolian | mong |
| mro | mroo |
| multani | mult |
| myanmar | mymr |
| nabataean | nbat |
| nag_mundari | nagm |
| nandinagari | nand |
| new_tai_lue | talu |
| newa | newa |
| nko | nkoo |
| nushu | nshu |
| nyiakeng_puachue_hmong | hmnp |
| ogham | ogam |
| ol_chiki | olck |
| ol_onal | onao |
| old_hungarian | hung |
| old_italic | ital |
| old_north_arabian | narb |
| old_permic | perm |
| old_persian | xpeo |
| old_sogdian | sogo |
| old_south_arabian | sarb |
| old_turkic | orkh |
| old_uyghur | ougr |
| oriya | orya |
| osage | osge |
| osmanya | osma |
| pahawh_hmong | hmng |
| palmyrene | palm |
| pau_cin_hau | pauc |
| phags_pa | phag |
| phoenician | phnx |
| psalter_pahlavi | phlp |
| rejang | rjng |
| runic | runr |
| samaritan | samr |
| saurashtra | saur |
| sharada | shrd |
| shavian | shaw |
| siddham | sidd |
| sidetic | sidt |
| signwriting | sgnw |
| sinhala | sinh |
| sogdian | sogd |
| sora_sompeng | sora |
| soyombo | soyo |
| sundanese | sund |
| sunuwar | sunu |
| syloti_nagri | sylo |
| syriac | syrc |
| tagalog | tglg |
| tagbanwa | tagb |
| tai_le | tale |
| tai_tham | lana |
| tai_viet | tavt |
| tai_yo | tayo |
| takri | takr |
| tamil | taml |
| tangsa | tnsa |
| tangut | tang |
| telugu | telu |
| thaana | thaa |
| thai | thai |
| tibetan | tibt |
| tifinagh | tfng |
| tirhuta | tirh |
| todhri | todr |
| tolong_siki | tols |
| toto | toto |
| tulu_tigalari | tutg |
| ugaritic | ugar |
| unknown | zzzz |
| vai | vaii |
| vithkuqi | vith |
| wancho | wcho |
| warang_citi | wara |
| yezidi | yezi |
| yi | yiii |
| zanabazar_square | zanb |