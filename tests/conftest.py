"""Shared fixtures and language-pack discovery for the R-BPE test suite."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pytest
import yaml
from transformers import AutoTokenizer

from rbpe.rbpe_tokenizer import RBPETokenizer


PACKS_DIR = Path(__file__).parent / "packs"


@dataclass(frozen=True)
class LangPack:
    """A language pack: a directory of sample texts + a pack.yaml manifest."""

    name: str
    dir: Path
    meta: dict = field(default_factory=dict)

    @property
    def target_language_scripts(self) -> list[str]:
        return list(self.meta.get("target_language_scripts", []))

    @property
    def preserved_languages_scripts(self) -> list[str]:
        return list(self.meta.get("preserved_languages_scripts", []))

    def has(self, category: str) -> bool:
        return (self.dir / f"{category}.json").exists()

    def samples(self, category: str) -> list[Any]:
        """Load samples for a category. Returns [] if the category file is absent."""
        path = self.dir / f"{category}.json"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"pack {self.name!r}: {path} must contain a JSON array")
        return data


def _discover_packs() -> dict[str, LangPack]:
    packs: dict[str, LangPack] = {}
    if not PACKS_DIR.is_dir():
        return packs
    for manifest in sorted(PACKS_DIR.glob("*/pack.yaml")):
        with manifest.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        name = manifest.parent.name
        packs[name] = LangPack(name=name, dir=manifest.parent, meta=meta)
    return packs


PACKS = _discover_packs()


def pytest_addoption(parser):
    parser.addoption(
        "--rbpe-pretrained",
        action="store",
        default=None,
        help="Path to a saved R-BPE tokenizer directory (as accepted by RBPETokenizer.from_pretrained). "
        "Falls back to the RBPE_PRETRAINED_PATH environment variable.",
    )
    parser.addoption(
        "--rbpe-lang-pack",
        action="store",
        default=None,
        help="Name of the language pack to run the suite against (default: RBPE_LANG_PACK env var, "
        "or the first pack found alphabetically).",
    )


def _opt(config, name: str, env: str) -> str | None:
    return config.getoption(name) or os.getenv(env)


def _active_pack_name(config) -> str | None:
    name = _opt(config, "--rbpe-lang-pack", "RBPE_LANG_PACK")
    if name:
        return name
    return sorted(PACKS)[0] if PACKS else None


@pytest.fixture(scope="session")
def pack(request) -> LangPack:
    name = _active_pack_name(request.config)
    if not name:
        pytest.skip("no language packs found under tests/packs/")
    if name not in PACKS:
        pytest.skip(f"language pack {name!r} not found. Available: {sorted(PACKS)}")
    return PACKS[name]


@pytest.fixture(scope="session")
def _pretrained_path(request) -> str:
    pretrained = _opt(request.config, "--rbpe-pretrained", "RBPE_PRETRAINED_PATH")
    if not pretrained:
        pytest.skip(
            "set RBPE_PRETRAINED_PATH (or --rbpe-pretrained) to a saved R-BPE "
            "tokenizer directory to run the tokenizer test suite."
        )
    return pretrained


@pytest.fixture(scope="session")
def tokenizer(_pretrained_path):
    """The prepared R-BPE tokenizer under test (session-scoped — built once)."""
    return RBPETokenizer.from_pretrained(_pretrained_path)


@pytest.fixture(scope="session")
def mapping_tokenizer(tokenizer):
    """The MappingTokenizer embedded in the R-BPE tokenizer."""
    assert tokenizer.mapping_tokenizer is not None, "mapping_tokenizer is None"
    return tokenizer.mapping_tokenizer


@pytest.fixture(scope="session")
def tokenizer_config_dict(_pretrained_path) -> dict:
    """The source-of-truth ``custom_tokenizer_config`` from the saved tokenizer."""
    cfg_path = os.path.join(_pretrained_path, "tokenizer_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)["custom_tokenizer_config"]


@pytest.fixture(scope="session")
def reference_tokenizer(tokenizer_config_dict):
    """The original HuggingFace tokenizer that the R-BPE tokenizer was adapted from."""
    return AutoTokenizer.from_pretrained(tokenizer_config_dict["model_id"])


# ---------------------------------------------------------------------------
# Parametrization: expand pack categories into per-sample test cases.
# Each test module declares which categories it consumes by naming a fixture:
#   roundtrip_case, punctuation_case, replacement_stress_case, chat_case,
#   serialization_case.
# ---------------------------------------------------------------------------

ROUNDTRIP_CATEGORIES = [
    "basic",
    "diacritics",
    "target_numerals",
    "ascii_numbers",
    "preserved_language",
    "mixed_with_preserved",
    "emoji",
    "target_with_emoji",
    "emoji_with_target_numerals",
    "emoji_coverage",
    "punctuation",
    "target_with_punctuation",
    "replacement_chars",
]


def _expand(pack: LangPack, categories: Iterable[str]) -> tuple[list, list[str]]:
    cases, ids = [], []
    for category in categories:
        for idx, text in enumerate(pack.samples(category)):
            cases.append((category, idx, text))
            ids.append(f"{pack.name}-{category}-{idx}")
    return cases, ids


def pytest_generate_tests(metafunc):
    pack_name = _active_pack_name(metafunc.config)
    if pack_name is None or pack_name not in PACKS:
        return  # `pack` fixture will skip at runtime
    active_pack = PACKS[pack_name]

    if "roundtrip_case" in metafunc.fixturenames:
        cases, ids = _expand(active_pack, ROUNDTRIP_CATEGORIES)
        metafunc.parametrize("roundtrip_case", cases, ids=ids)

    if "punctuation_case" in metafunc.fixturenames:
        cases, ids = _expand(active_pack, ["punctuation", "target_with_punctuation"])
        metafunc.parametrize("punctuation_case", cases, ids=ids)

    if "replacement_stress_case" in metafunc.fixturenames:
        cases, ids = _expand(active_pack, ["replacement_char_stress"])
        metafunc.parametrize("replacement_stress_case", cases, ids=ids)

    if "chat_case" in metafunc.fixturenames:
        samples = active_pack.samples("chat_samples")
        ids = [f"{active_pack.name}-chat-{i}" for i in range(len(samples))]
        metafunc.parametrize("chat_case", samples, ids=ids)

    if "serialization_case" in metafunc.fixturenames:
        cases, ids = _expand(
            active_pack,
            ["basic", "preserved_language", "mixed_with_preserved"],
        )
        metafunc.parametrize("serialization_case", cases, ids=ids)
