"""Wires settings into the catalogue, index, tools, LLM provider and agent."""

from dataclasses import dataclass, field
from functools import cached_property

from medaibot.config import Settings
from medaibot.drugs import DrugCatalog
from medaibot.embeddings import make_embedder
from medaibot.index import HybridIndex
from medaibot.query import NormalisedQuery, normalise_query


class IndexNotBuilt(Exception):
    pass


@dataclass
class Runtime:
    settings: Settings
    catalog: DrugCatalog
    index: HybridIndex
    _extras: dict[str, object] = field(default_factory=dict)

    def normalise(self, question: str) -> NormalisedQuery:
        return normalise_query(question, self.catalog)

    @cached_property
    def agent(self):  # type: ignore[no-untyped-def]
        from medaibot.agent import build_agent

        return build_agent(self.settings, self.catalog, self.index)


def load_runtime(settings: Settings) -> Runtime:
    catalog = DrugCatalog.from_manifest(settings.sources_path)
    if not (settings.index_dir / "meta.json").exists():
        raise IndexNotBuilt(f"no index in {settings.index_dir}; run `medaibot fetch` then `medaibot ingest`")
    index = HybridIndex.load(settings.index_dir, make_embedder(settings))
    return Runtime(settings=settings, catalog=catalog, index=index)
