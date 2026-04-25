# Language packs

A **language pack** is a self-contained directory that describes one target
language for the R-BPE test suite. Adding support for a new target language
means dropping in a new directory here, no code changes required.

The suite runs against one pack at a time. Select it with:

```bash
export RBPE_LANG_PACK=arabic    # or: pytest --rbpe-lang-pack arabic
```

If unset, the alphabetically-first pack is used.

## Directory layout

```
packs/<name>/
├── pack.yaml                        # manifest (required)
├── basic.json                       # target-language text
├── diacritics.json                  # target script with diacritics/marks
├── target_numerals.json             # text with target-script digit system
├── ascii_numbers.json               # text with ASCII digits and math
├── preserved_language.json          # text in a preserved (non-target) language
├── mixed_with_preserved.json        # target + preserved language mixed
├── emoji.json                       # emoji-only samples
├── target_with_emoji.json           # target-language text interleaved with emoji
├── emoji_with_target_numerals.json  # emoji adjacent to target-script digits
├── emoji_coverage.json              # broad emoji sweep (singles + pairs + triplets)
├── punctuation.json                 # punctuation-heavy samples (any language)
├── target_with_punctuation.json     # target language with punctuation
├── replacement_chars.json           # samples that exercise the replacement-char map
├── replacement_char_stress.json     # stress cases: must not decode to "�"
└── chat_samples.json                # list of chat-template message arrays
```

Every sample file except `chat_samples.json` is a JSON array of strings.
`chat_samples.json` is a JSON array of message arrays — each message is an
object with `role` and `content` fields, exactly as accepted by
`transformers.PreTrainedTokenizer.apply_chat_template`.

Any category file may be omitted; the corresponding tests are simply skipped
for that pack.

## `pack.yaml`

```yaml
name: arabic
target_language_scripts:
  - arabic
preserved_languages_scripts:
  - latin
  - greek
```

`target_language_scripts` and `preserved_languages_scripts` use the same
script names/aliases accepted by `RBPETokenizer` (see [unicode_scripts.md](../../unicode_scripts.md)).

## Adding a new pack

1. Create `packs/<lang>/pack.yaml` with the manifest fields above.
2. Drop in whichever category files you have sample text for. You can start with a
   `basic.json` and `preserved_language.json`; add the rest as you grow
   coverage.
3. Load an R-BPE tokenizer adapted for `<lang>` and run:
   ```bash
   RBPE_LANG_PACK=<lang> RBPE_PRETRAINED_PATH=/path/to/tokenizer pytest
   ```
4. Everything that has data runs automatically.
