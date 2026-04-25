import os
import tempfile

import pytest

from rbpe.rbpe_tokenizer import RBPETokenizer


@pytest.fixture(scope="module")
def hf_sample_texts(pack) -> list[str]:
    """Short sample texts."""
    texts = ["Hello world!", "Multiple tokens here to test"]
    target = pack.samples("basic")[:1]
    mixed = pack.samples("mixed_with_preserved")[:1]
    return texts + target + mixed


@pytest.fixture(scope="module")
def hf_padding_texts() -> list[str]:
    """Three texts with deliberately different lengths for padding tests."""
    return [
        "Short text",
        "This is a longer text with more tokens",
        "Very very very long text that should be longer than the others",
    ]


# Encoding format vs. the reference HF tokenizer
ENCODING_CONFIGS = [
    pytest.param({"return_tensors": None}, id="default-lists"),
    pytest.param(
        {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": 28,
        },
        id="pt-padmax",
    ),
    pytest.param(
        {
            "return_tensors": "np",
            "padding": "max_length",
            "truncation": True,
            "max_length": 28,
        },
        id="np-padmax",
    ),
    pytest.param({"padding": "max_length", "max_length": 28}, id="padmax"),
    pytest.param({"add_special_tokens": False}, id="no-specials"),
    pytest.param({"return_attention_mask": True}, id="attn-mask"),
]


def _assert_same_structure(custom, reference, *, context: str) -> None:
    assert custom.keys() == reference.keys(), (
        f"{context}: output keys differ ({set(custom)} vs {set(reference)})"
    )
    for key in custom.keys():
        c, r = custom[key], reference[key]
        assert type(c) is type(r), (
            f"{context}: value type for {key!r} differs ({type(c)} vs {type(r)})"
        )
        if hasattr(c, "dtype"):
            assert c.dtype == r.dtype, (
                f"{context}: dtype for {key!r} differs ({c.dtype} vs {r.dtype})"
            )
            assert c.shape == r.shape, (
                f"{context}: shape for {key!r} differs ({c.shape} vs {r.shape})"
            )


@pytest.mark.parametrize("config", ENCODING_CONFIGS)
def test_encoding_format_single(
    tokenizer, reference_tokenizer, hf_sample_texts, config
):
    """Single-text encoding output has the same keys, dtypes and shapes as HF."""
    text = hf_sample_texts[0]
    custom = tokenizer(text, **config)
    reference = reference_tokenizer(text, **config)
    _assert_same_structure(custom, reference, context=f"single/{config}")


@pytest.mark.parametrize("config", ENCODING_CONFIGS)
def test_encoding_format_batch(tokenizer, reference_tokenizer, hf_sample_texts, config):
    """Batch encoding output matches HF structure and keeps a correct batch dimension."""
    custom = tokenizer(hf_sample_texts, **config)
    reference = reference_tokenizer(hf_sample_texts, **config)
    _assert_same_structure(custom, reference, context=f"batch/{config}")
    for key in custom.keys():
        value = custom[key]
        if hasattr(value, "shape"):
            assert value.shape[0] == len(hf_sample_texts), (
                f"batch/{config}: batch dim mismatch for {key!r}"
            )


# Padding behaviour
PADDING_CONFIGS = [
    pytest.param(
        {"padding": True, "truncation": False, "max_length": None},
        id="longest-default",
    ),
    pytest.param(
        {"padding": "longest", "truncation": False, "max_length": None},
        id="longest-explicit",
    ),
    pytest.param(
        {"padding": "max_length", "max_length": 20, "truncation": True},
        id="max-length-20",
    ),
    pytest.param(
        {"padding": True, "max_length": 12, "truncation": True},
        id="longest-after-trunc-12",
    ),
]


@pytest.mark.parametrize("config", PADDING_CONFIGS)
def test_padding_behavior(tokenizer, reference_tokenizer, hf_padding_texts, config):
    custom = tokenizer(hf_padding_texts, **config)
    reference = reference_tokenizer(hf_padding_texts, **config)

    input_ids = custom["input_ids"]
    attention_mask = custom["attention_mask"]

    unique_lengths = {len(seq) for seq in input_ids}
    assert len(unique_lengths) == 1, f"rows have different lengths: {unique_lengths}"

    pad_id = tokenizer.pad_token_id
    for seq, mask in zip(input_ids, attention_mask):
        non_pad = sum(1 for t in seq if t != pad_id)
        assert non_pad == sum(mask), "attention mask does not match padding"

    max_length = config.get("max_length")
    if max_length is not None:
        if config.get("padding") == "max_length":
            assert all(len(seq) == max_length for seq in input_ids), (
                f"not padded to exactly max_length={max_length}"
            )
        else:
            longest_content = max(
                sum(1 for t in seq if t != pad_id) for seq in input_ids
            )
            assert longest_content <= max_length, (
                f"content length {longest_content} exceeds max_length={max_length}"
            )
    elif config.get("padding") in (True, "longest"):
        content_lengths = [sum(1 for t in seq if t != pad_id) for seq in input_ids]
        assert len(input_ids[0]) == max(content_lengths), (
            "row length does not match longest content"
        )

    assert custom.keys() == reference.keys()


def test_pad_token_placement_max_length(tokenizer, hf_padding_texts):
    max_length = 20
    encoded = tokenizer(
        hf_padding_texts,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
        truncation=True,
    )
    assert all(len(seq) == max_length for seq in encoded["input_ids"])
    for seq, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        for pos, m in enumerate(mask):
            if m == 0:
                assert seq[pos] == tokenizer.pad_token_id, (
                    f"expected pad_token_id at masked position {pos}, got {seq[pos]}"
                )


def test_pad_token_placement_longest(tokenizer, hf_padding_texts):
    encoded = tokenizer(hf_padding_texts, padding=True, return_attention_mask=True)
    max_len = max(len(seq) for seq in encoded["input_ids"])
    assert all(len(seq) == max_len for seq in encoded["input_ids"])
    for seq, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        non_pad = sum(1 for t in seq if t != tokenizer.pad_token_id)
        assert non_pad == sum(mask)


# Special tokens
def test_special_tokens_single(tokenizer):
    """BOS and EOS bracket the encoded sequence when add_special_tokens=True."""
    ids = tokenizer("Hello world!", add_special_tokens=True)["input_ids"]
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id


def test_special_tokens_batch(tokenizer):
    """In a padded batch, BOS/EOS bracket the non-pad region of every row."""
    encoded = tokenizer(
        ["Hello", "World", "Test"],
        padding=True,
        add_special_tokens=True,
    )
    for seq in encoded["input_ids"]:
        non_pad = [i for i, t in enumerate(seq) if t != tokenizer.pad_token_id]
        assert non_pad, "sequence is entirely padding"
        assert seq[non_pad[0]] == tokenizer.bos_token_id
        assert seq[non_pad[-1]] == tokenizer.eos_token_id


def test_special_tokens_initialization(
    tokenizer, reference_tokenizer, tokenizer_config_dict
):
    """Config-declared special tokens override the reference tokenizer's values;
    unspecified ones fall back to the reference."""
    config_specials = tokenizer_config_dict.get("special_tokens", {}) or {}
    for name in [
        "pad_token",
        "eos_token",
        "bos_token",
        "unk_token",
        "mask_token",
        "sep_token",
        "cls_token",
    ]:
        expected = config_specials.get(name) or getattr(reference_tokenizer, name)
        actual = getattr(tokenizer, name)
        assert actual == expected, f"{name}: expected {expected!r}, got {actual!r}"

    for id_attr in ["eos_token_id", "bos_token_id", "unk_token_id", "pad_token_id"]:
        tok_id = getattr(tokenizer, id_attr)
        assert tok_id is not None, f"{id_attr} is None"
        tok_name = id_attr.removesuffix("_id")
        expected = getattr(tokenizer, tok_name)
        assert tokenizer.decode([tok_id]).strip() == expected, (
            f"{id_attr}={tok_id} decodes to {tokenizer.decode([tok_id])!r}, expected {expected!r}"
        )


# Chat template
def test_chat_template(tokenizer, reference_tokenizer, chat_case):
    """``apply_chat_template`` rendering round-trips the same way as the reference tokenizer."""
    messages = chat_case

    reference_render = reference_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    custom_render = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    ref_ids_of_ref = reference_tokenizer(reference_render, add_special_tokens=False)[
        "input_ids"
    ]
    cus_ids_of_ref = tokenizer(reference_render, add_special_tokens=False)["input_ids"]
    ref_ids_of_cus = reference_tokenizer(custom_render, add_special_tokens=False)[
        "input_ids"
    ]
    cus_ids_of_cus = tokenizer(custom_render, add_special_tokens=False)["input_ids"]

    ref_decoded_of_ref = reference_tokenizer.decode(
        ref_ids_of_ref, skip_special_tokens=False
    )
    cus_decoded_of_ref = tokenizer.decode(cus_ids_of_ref, skip_special_tokens=False)
    ref_decoded_of_cus = reference_tokenizer.decode(
        ref_ids_of_cus, skip_special_tokens=False
    )
    cus_decoded_of_cus = tokenizer.decode(cus_ids_of_cus, skip_special_tokens=False)

    assert cus_decoded_of_ref.strip() == ref_decoded_of_ref.strip(), (
        "decoded strings diverge on the reference-rendered template"
    )
    assert cus_decoded_of_cus.strip() == ref_decoded_of_cus.strip(), (
        "decoded strings diverge on the custom-rendered template"
    )


# save_pretrained / from_pretrained
def test_save_and_load_pretrained(tokenizer, pack):
    """save_pretrained writes the expected layout and from_pretrained reproduces behaviour."""
    basic = pack.samples("basic")
    preserved = pack.samples("preserved_language")
    sample = (preserved + basic + ["Hello world!"])[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)

        assert os.path.exists(os.path.join(tmp_dir, "tokenizer_config.json"))
        assert os.path.exists(os.path.join(tmp_dir, "tokenizer.json"))

        loaded = RBPETokenizer.from_pretrained(tmp_dir)

        assert hasattr(loaded, "mapping_tokenizer")
        assert loaded.mapping_tokenizer is not None

        original_ids = tokenizer.encode(sample)
        loaded_ids = loaded.encode(sample)
        assert original_ids == loaded_ids, "encoding diverged after save/load"

        assert tokenizer.decode(original_ids) == loaded.decode(loaded_ids), (
            "decoding diverged after save/load"
        )
