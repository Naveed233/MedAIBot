"""Drug catalogue loaded from data/sources.json, with brand, generic and Japanese aliases."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from medaibot.text import is_cjk, normalize


@dataclass(frozen=True)
class Drug:
    key: str
    brand: str
    generic: str
    set_id: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.brand, self.generic, *self.aliases)

    @property
    def display_name(self) -> str:
        return f"{self.brand} ({self.generic})"


class DrugCatalog:
    def __init__(self, drugs: list[Drug]) -> None:
        self._drugs = {d.key: d for d in drugs}
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        for drug in drugs:
            for name in drug.names:
                norm = normalize(name)
                # Japanese has no word boundaries, so match the name anywhere. A Latin name only
                # needs no Latin letter or digit beside it: \b fails in "tagrissoの", because Python
                # counts Japanese characters as word characters.
                pattern = re.escape(norm) if is_cjk(norm) else rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])"
                self._patterns.append((re.compile(pattern), drug.key))
        self._by_set_id = {d.set_id: d for d in drugs}

    @classmethod
    def from_manifest(cls, path: Path) -> "DrugCatalog":
        data = json.loads(path.read_text(encoding="utf-8"))
        drugs = [
            Drug(
                key=d["key"],
                brand=d["brand"],
                generic=d["generic"],
                set_id=d["set_id"],
                aliases=tuple(d.get("aliases", [])),
            )
            for d in data["drugs"]
        ]
        return cls(drugs)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._drugs.values())

    def __len__(self) -> int:
        return len(self._drugs)

    def get(self, key_or_name: str) -> Drug | None:
        if key_or_name in self._drugs:
            return self._drugs[key_or_name]
        found = self.find_in_text(key_or_name)
        return found[0] if len(found) == 1 else None

    def by_set_id(self, set_id: str) -> Drug | None:
        return self._by_set_id.get(set_id)

    def find_in_text(self, text: str) -> list[Drug]:
        """Every catalogue drug mentioned in the text, in catalogue order."""
        norm = normalize(text)
        keys = {key for pattern, key in self._patterns if pattern.search(norm)}
        return [d for k, d in self._drugs.items() if k in keys]

    @property
    def keys(self) -> list[str]:
        return list(self._drugs)
