"""The guard's hint index: each graph.json node's label, file and location.

A hook-guard hint names the graph nodes behind a search or read, on every hooked
tool call, and parsing a large graph.json for three fields per node is most of
what that costs. The index keeps just those fields in
``<out>/cache/hint-index.json``, stamped with the graph's size and mtime, so a
stale index is never read. A build writes it beside each graph.json it
publishes, from the nodes it already holds; the guard rebuilds it from the graph
when the stamp does not match (a graph written some other way, or by an older
graphify).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

FIELDS = ("label", "source_file", "source_location")


def index_path(graph: Path) -> Path:
    return graph.parent / "cache" / "hint-index.json"


def _stamp(graph: Path) -> list[int]:
    stat = graph.stat()
    return [stat.st_size, stat.st_mtime_ns]


def read(graph: Path) -> list[dict] | None:
    """The nodes of ``graph`` as `FIELDS`, from an index written for this very
    graph, else None."""
    try:
        kept = json.loads(index_path(graph).read_text(encoding="utf-8"))
        rows = kept["nodes"] if kept.get("graph") == _stamp(graph) else None
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return None
    if not isinstance(rows, list) or not all(isinstance(r, list) and len(r) == len(FIELDS) for r in rows):
        return None
    return [dict(zip(FIELDS, r)) for r in rows]


def write(graph: Path, nodes: object = None) -> list[dict]:
    """Index ``graph`` and return its nodes as `FIELDS`. ``nodes`` are the graph's
    own nodes when the caller holds them; otherwise the graph is parsed. The stamp
    is taken first, so a graph rewritten meanwhile leaves an index that no longer
    matches rather than a wrong one. A failed write only costs the next hint a
    parse."""
    stamp = _stamp(graph)
    if nodes is None:
        nodes = json.loads(graph.read_text(encoding="utf-8")).get("nodes")
    rows = [[n.get(f) for f in FIELDS] for n in nodes if isinstance(n, dict)] if isinstance(nodes, list) else []
    index = index_path(graph)
    try:
        index.parent.mkdir(parents=True, exist_ok=True)
        tmp = index.with_name(f"{index.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"graph": stamp, "nodes": rows}, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, index)
    except OSError:
        pass
    return [dict(zip(FIELDS, r)) for r in rows]


def publish(graph: str | Path, nodes: object) -> None:
    """`write` for a build that has just put ``graph`` in place: never raises, so
    an index can never cost the build."""
    try:
        write(Path(graph), nodes)
    except Exception:
        pass
