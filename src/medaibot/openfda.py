"""Small client for the openFDA drug label endpoint."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from medaibot.drugs import Drug

logger = logging.getLogger(__name__)

LABEL_PATH = "/drug/label.json"


class OpenFDAError(Exception):
    """Any failure talking to openFDA. `status` is the HTTP status when there was one."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LabelNotFound(OpenFDAError):
    pass


def dailymed_url(set_id: str) -> str:
    return f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}"


class OpenFDAClient:
    def __init__(
        self,
        base_url: str = "https://api.fda.gov",
        api_key: str | None = None,
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def close(self) -> None:
        self._http.close()

    def _search(self, search: str, limit: int = 1, sort: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"search": search, "limit": limit}
        if sort:
            params["sort"] = sort
        if self._api_key:
            params["api_key"] = self._api_key
        try:
            response = self._http.get(LABEL_PATH, params=params)
        except httpx.TimeoutException as exc:
            raise OpenFDAError(f"openFDA timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise OpenFDAError(f"openFDA request failed: {exc}") from exc

        # openFDA answers "no results" with a 404 and an error body.
        if response.status_code == 404:
            raise LabelNotFound(f"no label matches {search!r}", status=404)
        if response.status_code >= 400:
            raise OpenFDAError(f"openFDA returned HTTP {response.status_code}", status=response.status_code)
        try:
            body = response.json()
        except ValueError as exc:
            raise OpenFDAError("openFDA returned a non-JSON body", status=response.status_code) from exc
        results = body.get("results") or []
        if not results:
            raise LabelNotFound(f"no label matches {search!r}", status=response.status_code)
        return list(results)

    def label_by_set_id(self, set_id: str) -> dict[str, Any]:
        return self._search(f'set_id:"{set_id}"')[0]

    def find_label(self, name: str) -> dict[str, Any]:
        """Newest label whose openFDA brand or generic name matches `name`."""
        quoted = name.replace('"', "").upper()
        search = f'openfda.brand_name:"{quoted}" OR openfda.generic_name:"{quoted}"'
        return self._search(search, limit=1, sort="effective_time:desc")[0]


def fetch_label(client: OpenFDAClient, drug: Drug, raw_dir: Path) -> Path:
    """Fetch one label by its pinned set_id and save it with provenance."""
    label = client.label_by_set_id(drug.set_id)
    if label.get("set_id") != drug.set_id:
        raise OpenFDAError(f"asked for set_id {drug.set_id}, got {label.get('set_id')}")
    record = {
        "provenance": {
            "source": "openFDA drug label API",
            "request": f'{LABEL_PATH}?search=set_id:"{drug.set_id}"',
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "set_id": label["set_id"],
            "spl_id": label.get("id"),
            "version": label.get("version"),
            "effective_time": label.get("effective_time"),
            "dailymed_url": dailymed_url(drug.set_id),
        },
        "label": label,
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{drug.key}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("saved label", extra={"fields": {"drug": drug.key, "path": str(path)}})
    return path
