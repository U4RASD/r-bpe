# R-BPE test suite

This directory contains the test suite for R-BPE. It is organised one file per module-under-test and driven by [language packs](packs/README.md) so adding tests for a new target language is a drop-in directory and does not requier any code changes.

## Layout

```
tests/
├── conftest.py                  # pack discovery + tokenizer fixtures
├── packs/
│   ├── README.md                # language-pack contract
│   └── arabic/                  # the reference pack (Arabic, the paper's target)
│       ├── pack.yaml
│       ├── basic.json
│       ├── diacritics.json
│       ├── target_numerals.json
│       ├── ascii_numbers.json
│       ├── preserved_language.json
│       ├── mixed_with_preserved.json
│       ├── emoji.json
│       ├── target_with_emoji.json
│       ├── emoji_with_target_numerals.json
│       ├── emoji_coverage.json
│       ├── punctuation.json
│       ├── target_with_punctuation.json
│       ├── replacement_chars.json
│       ├── replacement_char_stress.json
│       └── chat_samples.json
├── test_mapping_tokenizer.py    # tests for rbpe.mapping_tokenizer.MappingTokenizer
└── test_rbpe_tokenizer.py       # tests for rbpe.rbpe_tokenizer.RBPETokenizer
```

## Running

Tests need a prepared R-BPE tokenizer.

```bash
# Point at a saved R-BPE tokenizer directory:
RBPE_PRETRAINED_PATH=/path/to/saved_rbpe_tokenizer pytest

# Select the language pack (defaults to the first pack alphabetically):
RBPE_LANG_PACK=arabic pytest
```

CLI flags are also accepted: `--rbpe-pretrained`, `--rbpe-lang-pack`.

If none of the above is set, the tokenizer-dependent tests are skipped.

## Language-pack model

Test data lives on disk as JSON arrays of strings (or, for chat, arrays of
message objects). Each pack declares its target + preserved scripts in
`pack.yaml`. See [`packs/README.md`](packs/README.md) for the contract and
file-by-file explanation.

One pack is active per `pytest` invocation. `conftest.py`'s
`pytest_generate_tests` expands the active pack into per-sample parametrized
cases, so a failing sample is reported with a stable id like
`test_roundtrip[arabic-diacritics-0]`.

