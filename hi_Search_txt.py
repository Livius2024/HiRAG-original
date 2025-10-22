"""Command line helper for running HiRAG on local `.txt` documents.

This script wires together the configuration defined in ``config.yaml``
and exposes a small CLI that can index a plain text file (such as the
Romanian *Codul fiscal*) and run hierarchical or naive RAG queries over it.

Example
-------
```
python hi_Search_txt.py /path/to/Codul_fiscal.txt \
    --question "Care sunt obligațiile de TVA pentru microîntreprinderi?"
```

The script assumes that the relevant API keys (for example
``OPENAI_API_KEY``) are either stored in ``config.yaml`` or provided via
environment variables.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import yaml
from openai import AsyncOpenAI

from hirag import HiRAG, QueryParam
from hirag._utils import (
    EmbeddingFunc,
    compute_args_hash,
    wrap_embedding_func_with_attrs,
)


LOGGER = logging.getLogger("hi_Search_txt")


def _clean_config_value(value: Optional[str]) -> Optional[str]:
    """Return ``None`` for placeholder or empty config values."""

    if value is None:
        return None
    value = str(value).strip()
    if not value or value == "***":
        return None
    return value


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' does not exist.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_openai_embedding(
    *,
    client: AsyncOpenAI,
    model_name: str,
    embedding_dim: int,
    max_token_size: int,
) -> EmbeddingFunc:
    @wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim, max_token_size=max_token_size
    )
    async def _embedding(texts: Iterable[str]) -> np.ndarray:
        response = await client.embeddings.create(
            model=model_name, input=list(texts), encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])

    return _embedding


def _build_openai_chat(
    *,
    client: AsyncOpenAI,
    model_name: str,
):
    async def _chat(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        **kwargs,
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        hashing_kv = kwargs.pop("hashing_kv", None)
        if hashing_kv is not None:
            args_hash = compute_args_hash(model_name, messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        response = await client.chat.completions.create(
            model=model_name, messages=messages, **kwargs
        )
        content = response.choices[0].message.content

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": content, "model": model_name}}
            )
        return content

    return _chat


def _ensure_api_key(key_from_config: Optional[str]) -> str:
    api_key = _clean_config_value(key_from_config) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not provided. Set it in config.yaml or as an environment variable."
        )
    return api_key


def _read_text_file(path: Path, encoding: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Document '{path}' was not found.")
    return path.read_text(encoding=encoding)


def build_graph(
    *,
    config_path: Path,
    document_path: Path,
    encoding: str,
    working_dir_override: Optional[Path],
    skip_index: bool,
    force_reindex: bool,
) -> HiRAG:
    config = _load_config(config_path)

    openai_cfg = config.get("openai", {})
    hirag_cfg = config.get("hirag", {})
    model_params_cfg = config.get("model_params", {})

    api_key = _ensure_api_key(openai_cfg.get("api_key"))
    base_url = _clean_config_value(openai_cfg.get("base_url"))

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    embedding_func = _build_openai_embedding(
        client=client,
        model_name=_clean_config_value(openai_cfg.get("embedding_model"))
        or "text-embedding-3-small",
        embedding_dim=int(model_params_cfg.get("openai_embedding_dim", 1536)),
        max_token_size=int(model_params_cfg.get("max_token_size", 8192)),
    )

    chat_model_name = _clean_config_value(openai_cfg.get("model")) or "gpt-4o"
    chat_func = _build_openai_chat(client=client, model_name=chat_model_name)

    working_dir = (
        working_dir_override
        if working_dir_override is not None
        else Path(hirag_cfg.get("working_dir", "./hirag_workdir"))
    )

    if force_reindex and working_dir.exists():
        LOGGER.info("Removing existing working directory '%s'", working_dir)
        for item in working_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil

                shutil.rmtree(item)

    graph = HiRAG(
        working_dir=str(working_dir),
        enable_llm_cache=bool(hirag_cfg.get("enable_llm_cache", True)),
        enable_hierachical_mode=bool(
            hirag_cfg.get("enable_hierachical_mode", True)
        ),
        embedding_batch_num=int(hirag_cfg.get("embedding_batch_num", 32)),
        embedding_func_max_async=int(
            hirag_cfg.get("embedding_func_max_async", 8)
        ),
        enable_naive_rag=bool(hirag_cfg.get("enable_naive_rag", False)),
        embedding_func=embedding_func,
        best_model_func=chat_func,
        cheap_model_func=chat_func,
    )

    if not skip_index:
        LOGGER.info("Indexing document '%s'", document_path)
        content = _read_text_file(document_path, encoding=encoding)
        graph.insert(content)
    else:
        LOGGER.info("Skipping indexing step as requested.")

    return graph


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index a .txt document (e.g. Codul fiscal) and run HiRAG queries."
    )
    parser.add_argument(
        "document",
        type=Path,
        help="Path to the text document that should be indexed.",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question to ask after indexing. If omitted you will be prompted interactively.",
    )
    parser.add_argument(
        "--mode",
        choices=["hi", "naive", "hi_local", "hi_global", "hi_bridge", "hi_nobridge"],
        default="hi",
        help="Retrieval mode to use when querying.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file (defaults to ./config.yaml).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding used when reading the document.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        help="Override the working directory used for the vector stores and caches.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip (re-)indexing the document and reuse existing data in the working directory.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete cached data in the working directory before indexing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    try:
        graph = build_graph(
            config_path=args.config,
            document_path=args.document,
            encoding=args.encoding,
            working_dir_override=args.working_dir,
            skip_index=args.skip_index,
            force_reindex=args.force_reindex,
        )
    except Exception as exc:  # pragma: no cover - CLI error surface
        LOGGER.error("Failed to prepare HiRAG: %s", exc)
        return 1

    question = args.question
    if not question:
        try:
            question = input("Enter your question: ").strip()
        except EOFError:  # pragma: no cover - interactive guard
            question = ""

    if not question:
        LOGGER.info("No question provided; exiting after indexing.")
        return 0

    LOGGER.info("Running query in '%s' mode", args.mode)
    response = graph.query(question, param=QueryParam(mode=args.mode))
    print("\n=== HiRAG Response ===\n")
    print(response)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

