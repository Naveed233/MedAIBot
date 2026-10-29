"""Command line entry point: `medaibot fetch | ingest | search | ask | eval | serve`."""

import argparse
import logging
import os
import sys
import time

from medaibot.config import Settings, get_settings
from medaibot.logging_setup import configure_logging, log_event

logger = logging.getLogger("medaibot.cli")


def cmd_fetch(settings: Settings, args: argparse.Namespace) -> int:
    from medaibot.drugs import DrugCatalog
    from medaibot.openfda import OpenFDAClient, fetch_label

    catalog = DrugCatalog.from_manifest(settings.sources_path)
    key = settings.openfda_api_key.get_secret_value() if settings.openfda_api_key else None
    client = OpenFDAClient(settings.openfda_base_url, api_key=key, timeout_s=30)
    try:
        for drug in catalog:
            path = settings.raw_dir / f"{drug.key}.json"
            if path.exists() and not args.force:
                print(f"skip {drug.key}: {path} exists (use --force to re-fetch)")
                continue
            fetch_label(client, drug, settings.raw_dir)
            print(f"fetched {drug.key} -> {path}")
    finally:
        client.close()
    return 0


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    from medaibot.corpus import build_chunks, load_raw_dir
    from medaibot.drugs import DrugCatalog
    from medaibot.embeddings import make_embedder
    from medaibot.index import HybridIndex

    catalog = DrugCatalog.from_manifest(settings.sources_path)
    raws = load_raw_dir(settings.raw_dir, catalog)
    if not raws:
        print(f"no raw labels in {settings.raw_dir}; run `medaibot fetch` first", file=sys.stderr)
        return 1
    started = time.perf_counter()
    chunks = build_chunks(raws)
    versions = {r.drug.key: f"v{r.provenance['version']} ({r.provenance['effective_time']})" for r in raws}
    index = HybridIndex.build(chunks, make_embedder(settings), label_versions=versions)
    index.save(settings.index_dir)
    log_event(
        logger,
        "index built",
        chunks=len(chunks),
        labels=len(raws),
        embedder=index.meta["embedder"],
        seconds=round(time.perf_counter() - started, 1),
    )
    print(f"indexed {len(chunks)} chunks from {len(raws)} labels into {settings.index_dir}")
    return 0


def cmd_search(settings: Settings, args: argparse.Namespace) -> int:
    from medaibot.runtime import load_runtime

    runtime = load_runtime(settings)
    nq = runtime.normalise(args.query)
    hits = runtime.index.search(nq.text, k=args.k, drug=nq.drug, mode=args.mode, rrf_k=settings.rrf_k)
    print(f"query: {nq.text!r} drug={nq.drug} section_hint={nq.section}")
    for hit in hits:
        c = hit.chunk
        print(f"{hit.score:.4f} bm25#{hit.bm25_rank} dense#{hit.dense_rank} {c.chunk_id} | {c.subsection or ''}")
        print(f"    {c.text[:160]}")
    return 0


def cmd_ask(settings: Settings, args: argparse.Namespace) -> int:
    from medaibot.render import render_answer
    from medaibot.runtime import load_runtime

    runtime = load_runtime(settings)
    answer = runtime.agent.run(args.question)
    if args.json:
        print(answer.model_dump_json(indent=2))
    else:
        print(render_answer(answer))
    return 0


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    from medaibot.evaluation import run_evaluation

    return run_evaluation(settings, retrieval_only=args.retrieval_only, limit=args.limit)


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("medaibot.api:create_app", factory=True, host=args.host, port=args.port, log_config=None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medaibot", description="Cited Q&A over drug labels.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the labels listed in data/sources.json")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("ingest", help="chunk the raw labels and build the hybrid index")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="run retrieval only and print the hits")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with the agent")
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the evaluation set and write the failure report")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="default: $PORT or 8000")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        return int(args.func(settings, args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception("command failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
