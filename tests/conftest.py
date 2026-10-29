from pathlib import Path

import pytest

from medaibot.corpus import RawLabel, build_chunks, load_raw_dir
from medaibot.drugs import DrugCatalog
from medaibot.embeddings import HashingEmbedder
from medaibot.index import HybridIndex
from medaibot.models import Chunk

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def catalog() -> DrugCatalog:
    return DrugCatalog.from_manifest(ROOT / "data" / "sources.json")


@pytest.fixture(scope="session")
def raw_labels(catalog: DrugCatalog) -> list[RawLabel]:
    return load_raw_dir(FIXTURES, catalog)


@pytest.fixture(scope="session")
def chunks(raw_labels: list[RawLabel]) -> list[Chunk]:
    return build_chunks(raw_labels)


@pytest.fixture(scope="session")
def index(chunks: list[Chunk]) -> HybridIndex:
    return HybridIndex.build(chunks, HashingEmbedder())
