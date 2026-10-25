"""Load graph.json for a one-shot CLI lookup (`explain`, `path`, `query`).

A lookup on a large graph spent most of its time before it looked anything up:
parsing the JSON, building the networkx graph one `add_edge` at a time, and
indexing the labels, with Python's cyclic garbage collector re-scanning the
growing heap throughout. On an 80,000-node graph that was about 4 s of a 4.3 s
`explain`, and `query` built a trigram index on top. This module

- pauses the collector while it loads (nothing it builds is cyclic garbage);
- builds the graph straight into networkx's dicts, as `node_link_graph` would
  with `add_node` and `add_edge`;
- keeps what a lookup builds in the user's cache directory, keyed by the graph
  file's real path, size, modification time and inode (and the Python and
  networkx versions): the directed graph's dicts, the links' order, and the
  exact and trigram indexes. The next lookup on an unchanged graph loads that
  and makes its own graph from it: a `DiGraph` for `explain`, a
  `MultiDiGraph` for `path`, a `Graph` for `query`, each as its own parse
  would make it (`path` and `query` parse a file that links a pair twice or
  keys a link, whose links those shapes keep apart);
- builds `path`'s sorted traversal graph the same way.

On that graph `explain` went from about 4.0 s to 1.2 s, `query` from 7.1 s
to 2.1 s and `path` from 5.7 s to 2.9 s.

The cache is `marshal` data, plain dicts, lists and bytes, which loading cannot
run code from, as unpickling could. On a miss a forked child writes it (where
there is no fork, the lookup does), parsing
the file again and building both indexes, so no lookup waits on it; the child
lets go of the output streams first, so neither does a caller reading them to
their end. A cache whose header does not match the graph file as it is now is
not read. `GRAPHIFY_NO_LOOKUP_CACHE=1` turns it off.
"""
from __future__ import annotations

import gc
import hashlib
import json
import marshal
import os
import sys
import tempfile
from array import array
from contextlib import contextmanager
from pathlib import Path

import networkx as nx

_CACHE_FORMAT = 4


@contextmanager
def _collector_paused():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _build(data: dict, *, directed: bool, multigraph: bool):
    """`node_link_graph(dict(data, directed=directed, multigraph=multigraph))`
    for graphify's node-link shape (the caller's flags win over the file's, as
    every lookup forces them), writing networkx's adjacency dicts directly. Each
    record becomes its node's or edge's attribute dict, its `id` or `source`,
    `target` (and a multigraph's `key`) popped, so `data`'s nodes and links are
    spent. A node id that is not a string goes through networkx itself."""
    from networkx.readwrite import json_graph

    nodes = data.get("nodes", [])
    links = data.get("links", [])
    if not all(isinstance(n.get("id"), str) for n in nodes):
        data = dict(data, directed=directed, multigraph=multigraph)
        try:
            return json_graph.node_link_graph(data, edges="links")
        except TypeError:
            return json_graph.node_link_graph(data)
    if multigraph:
        G = nx.MultiDiGraph() if directed else nx.MultiGraph()
    else:
        G = nx.DiGraph() if directed else nx.Graph()
    G.graph = dict(data.get("graph", {}))
    node = G._node
    succ = G._succ if directed else G._adj
    pred = G._pred if directed else succ
    for n in nodes:
        nid = n.pop("id")
        node[nid] = n
        succ[nid] = {}
        if directed:
            pred[nid] = {}
    # one string object per node for every edge naming it: less memory, and
    # marshal writes a repeated object once
    canon = {nid: nid for nid in node}
    for e in links:
        u, v = e.pop("source"), e.pop("target")
        for end in (u, v):
            if end not in node:  # an edge to a node the file does not list
                node[end] = {}
                succ[end] = {}
                if directed:
                    pred[end] = {}
                canon[end] = end
        u, v = canon[u], canon[v]
        if multigraph:
            keys = succ[u].get(v)
            if keys is None:
                keys = {}
                succ[u][v] = keys
                pred[v][u] = keys
            key = e.pop("key", None)
            if key is None:  # MultiGraph.new_edge_key
                key = len(keys)
                while key in keys:
                    key += 1
            if key in keys:
                keys[key].update(e)
            else:
                keys[key] = e
            continue
        held = succ[u].get(v)
        if held is not None:  # a repeated pair: later attributes win, as add_edge's
            held.update(e)
            continue
        succ[u][v] = e
        pred[v][u] = e
    return G


def sorted_graph(nodes, *batches, directed: bool):
    """`add_nodes_from(sorted(nodes))`, then `add_edges_from(sorted(batch))`
    for each batch in turn, on a new `DiGraph` (or `Graph`), without networkx's
    per-edge calls: the canonical traversal graph `path` searches, whose
    neighbour order decides between routes of equal length. An edge repeated
    keeps its one empty attribute dict, as `add_edges_from` does."""
    with _collector_paused():
        G = nx.DiGraph() if directed else nx.Graph()
        node = G._node
        succ = G._succ if directed else G._adj
        pred = G._pred if directed else succ

        def add_node(n) -> None:
            if n not in node:
                node[n] = {}
                succ[n] = {}
                if directed:
                    pred[n] = {}
        for n in sorted(nodes):
            add_node(n)
        for batch in batches:
            for u, v in sorted(batch):
                add_node(u)
                add_node(v)
                if v not in succ[u]:
                    d = {}
                    succ[u][v] = d
                    pred[v][u] = d
        return G


def edge_dicts(G):
    """(u, v, data) for every edge of `G`, a multigraph's parallel ones each,
    straight from its adjacency: `G.edges(data=True)` in a different order,
    for a caller that sorts what it makes of them."""
    if G.is_multigraph():
        for u, nbrs in G._adj.items():
            for v, keys in nbrs.items():
                for d in keys.values():
                    yield u, v, d
    else:
        for u, nbrs in G._adj.items():
            for v, d in nbrs.items():
                yield u, v, d


def read_graph_data(gp: Path) -> dict:
    """graph.json parsed, with its edges under `links` whichever writer wrote it."""
    with _collector_paused():
        raw = json.loads(gp.read_text(encoding="utf-8"))
    if "links" not in raw and "edges" in raw:
        raw = dict(raw, links=raw["edges"])
    return raw


def graph_from_data(raw: dict, *, directed: bool, multigraph: bool = False):
    with _collector_paused():
        return _build(raw, directed=directed, multigraph=multigraph)


def _meta(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k not in ("nodes", "links", "edges")}


def _mark_direction(links) -> None:
    """`query`'s undirected graph keeps each link's stored direction as
    `_src`/`_tgt`; markers already in the file win (#2309)."""
    for link in links:
        link.setdefault("_src", link.get("source"))
        link.setdefault("_tgt", link.get("target"))


# -- the cache ---------------------------------------------------------------

def _cache_path(gp: Path) -> Path | None:
    if os.environ.get("GRAPHIFY_NO_LOOKUP_CACHE"):
        return None
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    digest = hashlib.sha1(str(gp.resolve()).encode("utf-8", "replace")).hexdigest()[:20]
    return Path(base) / "graphify" / "lookup" / f"{digest}.marshal"


def _stamp(gp: Path) -> list:
    st = gp.stat()
    return [_CACHE_FORMAT, str(gp.resolve()), st.st_size, st.st_mtime_ns, st.st_ino,
            list(sys.version_info[:2]), nx.__version__]


# what `path` and `query` need beyond `explain`'s, in a second record that
# `explain` does not read
_TRAVERSAL_KEYS = ("order", "trigrams")


def _read_cache(cache: Path, stamp: list, *, traversal: bool) -> dict | None:
    try:
        with cache.open("rb") as f:
            if json.loads(f.readline()) != stamp:
                return None
            # each record is its length then its marshal bytes, read whole:
            # marshal.load on a file reads it a few bytes at a time
            with _collector_paused():
                payload = marshal.loads(f.read(int.from_bytes(f.read(8), "little")))
                if traversal:
                    payload.update(marshal.loads(f.read(int.from_bytes(f.read(8), "little"))))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, EOFError, TypeError, AttributeError):
        return None


def _cache_payload(gp: Path) -> dict:
    """What the cache holds for `gp`: the directed graph's dicts (marshal
    writes an object met twice once, so each edge's dict comes back shared by
    `succ` and `pred`, as networkx keeps it), each link's ends in the file's
    order, whether two links share their ends or one carries a `key` (then
    only `explain` may use it: the other shapes keep both links), and both
    indexes, the trigram postings as bytes."""
    from graphify.serve import _get_exact_index, _get_trigram_index

    raw = read_graph_data(gp)
    links = raw.get("links", [])
    ends = [(link.get("source"), link.get("target")) for link in links]
    plain = (len(set(ends)) == len(ends)
             and len({frozenset(e) for e in ends}) == len(ends)
             and not any("key" in link for link in links))
    meta = _meta(raw)
    G = graph_from_data(raw, directed=True)
    with _collector_paused():
        canon = {nid: nid for nid in G._node}
        order = [(canon[u], canon[v]) for u, v in ends]
        exact = _get_exact_index(G)
        trigrams = _get_trigram_index(G)
        graph = {k: v for k, v in G.graph.items() if k not in ("_exact_index", "_trigram_index")}
        return {
            "node": G._node, "succ": G._succ, "pred": G._pred, "graph": graph, "meta": meta,
            "order": order, "plain": plain, "exact": exact,
            "trigrams": {g: a.tobytes() for g, a in trigrams["postings"].items()},
        }


def _write_cache_in_background(gp: Path, cache: Path, stamp: list) -> None:
    """Write the cache, in a forked child where the platform can fork one,
    else here."""
    pid = os.fork() if hasattr(os, "fork") else -1
    if pid > 0:
        return
    try:
        if pid == 0:
            devnull = os.open(os.devnull, os.O_RDWR)
            for fd in (0, 1, 2):
                os.dup2(devnull, fd)
        payload = _cache_payload(gp)
        if _stamp(gp) != stamp:
            return  # rewritten while this ran
        cache.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache.parent, prefix=".tmp-", suffix=".marshal")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(json.dumps(stamp).encode("utf-8") + b"\n")
                with _collector_paused():
                    for record in ({k: v for k, v in payload.items() if k not in _TRAVERSAL_KEYS},
                                   {k: payload[k] for k in _TRAVERSAL_KEYS}):
                        data = marshal.dumps(record)
                        f.write(len(data).to_bytes(8, "little"))
                        f.write(data)
            os.replace(tmp, cache)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        pass  # no cache costs the next lookup a parse, nothing more
    finally:
        if pid == 0:
            os._exit(0)


def _from_cache(c: dict, kind: str):
    """`kind`'s graph made from a cache, as a parse of the file would make it,
    or None when the file's links do not allow it."""
    node, succ, order = c["node"], c["succ"], c.get("order")
    with _collector_paused():
        if kind == "explain":
            G = nx.DiGraph()
            G._node = node
            G._adj = G._succ = succ
            G._pred = c["pred"]
        elif not c["plain"]:
            return None
        elif kind == "path":
            G = nx.MultiDiGraph()
            G._node = node
            msucc = {n: {} for n in node}
            mpred = {n: {} for n in node}
            for u, v in order:
                keys = {0: succ[u][v]}
                msucc[u][v] = keys
                mpred[v][u] = keys
            G._adj = G._succ = msucc
            G._pred = mpred
        else:  # query: undirected, each link's direction kept on it
            G = nx.Graph()
            G._node = node
            adj = {n: {} for n in node}
            for u, v in order:
                d = succ[u][v]
                d.setdefault("_src", u)
                d.setdefault("_tgt", v)
                adj[u][v] = d
                adj[v][u] = d
            G._adj = adj
            G.graph["_trigram_index"] = {
                "ids": list(node),
                "postings": {g: array("i", b) for g, b in c["trigrams"].items()},
                "set_cache": {},
            }
        G.graph.update(c["graph"])
        G.graph["_exact_index"] = c["exact"]
    return G


def load_lookup_graph(gp: Path, kind: str) -> tuple[nx.Graph, dict]:
    """The graph `kind` ("explain", "path" or "query") looks things up in, and
    graph.json's top-level keys other than its nodes and links: from the
    cache when it matches the file, else parsed, the cache then written in the
    background. `explain` gets a `DiGraph`, `path` a `MultiDiGraph` (parallel
    links survive, #2074), `query` an undirected `Graph` whose edges carry
    their stored direction as `_src`/`_tgt`."""
    cache = _cache_path(gp)
    stamp = _stamp(gp) if cache is not None else None
    if cache is not None:
        c = _read_cache(cache, stamp, traversal=kind != "explain")
        if c is not None:
            G = _from_cache(c, kind)
            if G is not None:
                return G, c["meta"]
        else:
            _write_cache_in_background(gp, cache, stamp)
    raw = read_graph_data(gp)
    meta = _meta(raw)
    if kind == "query":
        _mark_direction(raw.get("links", []))
    G = graph_from_data(raw, directed=kind != "query", multigraph=kind == "path")
    return G, meta
