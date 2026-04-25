import string
import unicodedata
import warnings
from unicodedata import normalize

from rbpe.mapping_tokenizer import MappingTokenizer


def test_encode_empty_string(mapping_tokenizer):
    encoded = mapping_tokenizer.encode("")
    assert isinstance(encoded, list)
    assert encoded == []


def test_decode_empty_tokens(mapping_tokenizer):
    decoded = mapping_tokenizer.decode([])
    assert isinstance(decoded, str)
    assert decoded == ""


def test_roundtrip(mapping_tokenizer, roundtrip_case):
    """encode then decode must equal the input under NFC, for every pack sample."""
    category, index, text = roundtrip_case
    encoded = mapping_tokenizer.encode(text)
    decoded = mapping_tokenizer.decode(encoded)
    assert normalize("NFC", decoded) == normalize("NFC", text), (
        f"[{category} #{index}] round-trip mismatch\n"
        f"  original: {text!r}\n"
        f"  decoded : {decoded!r}"
    )


def test_no_replacement_char_in_decoded(mapping_tokenizer, replacement_stress_case):
    """Stress samples, but must never produce U+FFFD."""
    category, index, text = replacement_stress_case
    decoded = mapping_tokenizer.decode(mapping_tokenizer.encode(text))
    assert "�" not in decoded, (
        f"[{category} #{index}] decoded text contains U+FFFD\n"
        f"  original: {text!r}\n"
        f"  decoded : {decoded!r}"
    )


# Punctuation characters we treat as "must be encoded separately from the
# target script". Extend in a pack-specific manner if a future pack needs more.
PUNCT_CHARS = string.punctuation + "،؛؟٪)،,٪،,٪.,٪),#,"
REPLACEMENT_CHAR = "�"


def _is_ignorable_mark(char: str) -> bool:
    """Marks (diacritics, combining) and modifier letters (e.g. Arabic tatweel).

    These may legitimately appear in a token that also contains punctuation
    but they do not violate the 'punctuation is separated' property.
    """
    cat = unicodedata.category(char)
    return cat.startswith("M") or cat == "Lm"


def test_punctuation_separated_in_encoding(mapping_tokenizer, punctuation_case):
    """Every token produced by ``encode(text)`` that contains a punctuation
    character must contain only punctuation / whitespace characters."""
    category, index, text = punctuation_case
    ids = mapping_tokenizer.encode(text)
    tokens = mapping_tokenizer.convert_tok_ids_to_tokens(ids)
    for token in tokens:
        has_punct = any(ch in PUNCT_CHARS for ch in token)
        if not has_punct:
            continue
        has_non_punct = any(ch not in PUNCT_CHARS and not ch.isspace() for ch in token)
        assert not has_non_punct, (
            f"[{category} #{index}] punctuation not encoded separately\n"
            f"  input : {text!r}\n"
            f"  token : {token!r}\n"
            f"  tokens: {tokens!r}"
        )


def _report_mixed_punct_tokens(mapping_tokenizer, inner_tokenizer, label: str):
    """Report vocabulary tokens that mix target script with punctuation."""
    vocab = inner_tokenizer.get_vocab()
    specials = set(inner_tokenizer.special_tokens_map.values())
    problematic = []
    for token_id in vocab.values():
        decoded = inner_tokenizer.decode([token_id])
        if decoded in specials:
            continue
        punct_chars = [c for c in decoded if c in PUNCT_CHARS]
        if not punct_chars:
            continue
        # punctuation that is part of a special token of the outer tokenizer
        # is OK (e.g. '<' or '|' inside '<|eot|>')
        if all(any(p in s for s in specials) for p in punct_chars):
            continue
        non_punct_non_mark = [
            c
            for c in decoded
            if c not in PUNCT_CHARS and not c.isspace() and not _is_ignorable_mark(c)
        ]
        if non_punct_non_mark and any(
            mapping_tokenizer._is_target_input(c) for c in non_punct_non_mark
        ):
            problematic.append((token_id, decoded))
    if problematic:
        sample = ", ".join(f"{tid}={tok!r}" for tid, tok in problematic[:5])
        more = "" if len(problematic) <= 5 else f" (+{len(problematic) - 5} more)"
        warnings.warn(
            f"[{label}] {len(problematic)} vocabulary tokens mix target "
            f"script with punctuation: {sample}{more}",
            stacklevel=2,
        )


def test_new_tokenizer_vocab_punct_report(mapping_tokenizer):
    _report_mixed_punct_tokens(
        mapping_tokenizer, mapping_tokenizer.new_tokenizer, "new_tokenizer"
    )


def test_old_tokenizer_vocab_punct_report(mapping_tokenizer):
    _report_mixed_punct_tokens(
        mapping_tokenizer, mapping_tokenizer.old_tokenizer, "old_tokenizer"
    )


def test_target_script_range_coverage(mapping_tokenizer):
    """For every codepoint in the tokenizer's target-script ranges, encoding
    and decoding a small stress context containing that codepoint must not produce
    U+FFFD."""
    ranges = mapping_tokenizer.target_language_scripts_ranges
    assert ranges, "target_language_scripts_ranges is empty — nothing to test"

    failures: list[dict] = []
    for start, end in ranges:
        for code_point in range(start, end + 1):
            char = chr(code_point)
            probe = f"{char}🤠{char}🤠"
            try:
                decoded = mapping_tokenizer.decode(mapping_tokenizer.encode(probe))
            except Exception as exc:
                failures.append({"code_point": hex(code_point), "error": repr(exc)})
                continue
            if REPLACEMENT_CHAR in decoded:
                failures.append({"code_point": hex(code_point), "decoded": decoded})

    if failures:
        lines = ["codepoints in target-script ranges that did not round-trip cleanly:"]
        for f in failures[:20]:
            if "error" in f:
                lines.append(f"  {f['code_point']}: {f['error']}")
            else:
                lines.append(f"  {f['code_point']}: decoded={f['decoded']!r}")
        if len(failures) > 20:
            lines.append(f"  ... (+{len(failures) - 20} more)")
        raise AssertionError("\n".join(lines))


# JSON serialization
def test_json_attributes(mapping_tokenizer):
    """to_json/from_json preserves all load-bearing attributes."""
    loaded = MappingTokenizer.from_json(mapping_tokenizer.to_json())

    assert loaded.new_tokenizer_path == mapping_tokenizer.new_tokenizer_path
    assert loaded.old_tokenizer_path == mapping_tokenizer.old_tokenizer_path
    assert loaded.new_to_old_map == mapping_tokenizer.new_to_old_map
    assert loaded.old_to_new_map == mapping_tokenizer.old_to_new_map
    assert (
        loaded.replacement_character_map == mapping_tokenizer.replacement_character_map
    )
    assert loaded.common_token_ids_map == mapping_tokenizer.common_token_ids_map


def test_json_functional(mapping_tokenizer, serialization_case):
    """After JSON round-trip, encoding, decoding and token-conversion are maintained."""
    _, _, text = serialization_case
    loaded = MappingTokenizer.from_json(mapping_tokenizer.to_json())

    original_ids = mapping_tokenizer.encode(text)
    loaded_ids = loaded.encode(text)
    assert original_ids == loaded_ids, "encoding diverged after JSON round-trip"

    assert mapping_tokenizer.decode(original_ids) == loaded.decode(loaded_ids), (
        "decoding diverged after JSON round-trip"
    )

    assert mapping_tokenizer.convert_tok_ids_to_tokens(
        original_ids
    ) == loaded.convert_tok_ids_to_tokens(loaded_ids), (
        "token-id conversion diverged after JSON round-trip"
    )
