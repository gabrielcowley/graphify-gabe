"""The one-shot lookups' graph loader: a direct build equal to networkx's
`node_link_graph`, and the cache each lookup's graph is made from while the
graph file is unchanged."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest
from networkx.readwrite import json_graph

from graphify import lookup_graph
from graphify.lookup_graph import graph_from_data, load_lookup_graph

_PLAIN = {
    "directed": False,
    "multigraph": False,
    "graph": {"name": "g"},
    "built_at_commit": "abc",
    "nodes": [
        {"id": "a", "label": "A()", "source_file": "a.c"},
        {"id": "b", "label": "B()", "source_file": "b.c"},
        {"id": "c", "label": "C", "source_file": "a.c"},
    ],
    "links": [
        {"source": "a", "target": "b", "relation": "calls", "weight": 1.0},
        # stored the other way round from its markers (#2309)
        {"source": "a", "target": "c", "relation": "references", "_src": "c", "_tgt": "a"},
        # an edge to a node the file does not list
        {"source": "b", "target": "ghost", "relation": "calls"},
    ],
}
# a pair linked twice, once each way: a multigraph keeps both
_REPEATED = dict(copy.deepcopy(_PLAIN), links=copy.deepcopy(_PLAIN["links"]) + [
    {"source": "a", "target": "b", "relation": "calls", "context": "again"},
    {"source": "b", "target": "a", "relation": "references"},
    # a multigraph's own key: the edge's key there, an attribute elsewhere;
    # the same key again updates that edge
    {"source": "c", "target": "b", "relation": "calls", "key": "k"},
    {"source": "c", "target": "b", "context": "same key", "key": "k"},
])
# each of these alone keeps path or query from the cache
_REVERSED = dict(copy.deepcopy(_PLAIN), links=copy.deepcopy(_PLAIN["links"]) + [
    {"source": "b", "target": "a", "relation": "references"},
])
_KEYED = dict(copy.deepcopy(_PLAIN), links=copy.deepcopy(_PLAIN["links"]) + [
    {"source": "c", "target": "b", "relation": "calls", "key": "k"},
])


def _networkx(data: dict, kind: str):
    data = copy.deepcopy(data)
    if kind == "query":
        for link in data["links"]:
            link.setdefault("_src", link["source"])
            link.setdefault("_tgt", link["target"])
    return json_graph.node_link_graph(
        dict(data, directed=kind != "query", multigraph=kind == "path"), edges="links")


def _same(a, b) -> bool:
    def adj(g, which):
        return [(n, [(m, d) for m, d in nbrs.items()]) for n, nbrs in which(g).items()]
    return (type(a) is type(b)
            and list(a.nodes(data=True)) == list(b.nodes(data=True))
            and adj(a, lambda g: g.adj) == adj(b, lambda g: g.adj)
            and (not a.is_directed() or adj(a, lambda g: g.pred) == adj(b, lambda g: g.pred))
            and {k: v for k, v in a.graph.items() if not k.startswith("_")} == b.graph)


@pytest.mark.parametrize("data", [_PLAIN, _REPEATED], ids=["plain", "repeated"])
@pytest.mark.parametrize("kind", ["explain", "path", "query"])
def test_the_direct_build_is_node_link_graphs(data, kind):
    raw = copy.deepcopy(data)
    if kind == "query":
        lookup_graph._mark_direction(raw["links"])
    got = graph_from_data(raw, directed=kind != "query", multigraph=kind == "path")
    assert _same(got, _networkx(data, kind))


def test_an_edge_dict_is_shared_by_both_ends():
    G = graph_from_data(copy.deepcopy(_PLAIN), directed=True)
    assert G.succ["a"]["b"] is G.pred["b"]["a"]


def _graph_file(tmp_path: Path, data: dict) -> Path:
    gp = tmp_path / "graph.json"
    gp.write_text(json.dumps(data), encoding="utf-8")
    return gp


def _cached(gp: Path) -> Path:
    cache = lookup_graph._cache_path(gp)
    deadline = time.time() + 10
    while not cache.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert cache.exists(), "the cache was never written"
    return cache


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("GRAPHIFY_NO_LOOKUP_CACHE", raising=False)
    return tmp_path / "cache"


def _no_parse(monkeypatch):
    def refuse(_gp):
        raise AssertionError("parsed graph.json although the cache matched")
    monkeypatch.setattr(lookup_graph, "read_graph_data", refuse)


@pytest.mark.parametrize("kind", ["explain", "path", "query"])
def test_each_lookup_makes_its_graph_from_the_cache(tmp_path, cache_home, monkeypatch, kind):
    gp = _graph_file(tmp_path, _PLAIN)
    parsed, meta = load_lookup_graph(gp, kind)  # a miss: parsed, the cache written aside
    assert _same(parsed, _networkx(_PLAIN, kind))
    assert meta == {"directed": False, "multigraph": False, "graph": {"name": "g"},
                    "built_at_commit": "abc"}
    assert lookup_graph._cache_path(gp).parent.parent.parent == cache_home
    _cached(gp)
    _no_parse(monkeypatch)
    G, meta2 = load_lookup_graph(gp, kind)
    assert _same(G, _networkx(_PLAIN, kind)) and meta2 == meta
    from graphify.serve import _get_exact_index, _get_trigram_index
    fresh = _networkx(_PLAIN, kind)
    assert G.graph["_exact_index"] == _get_exact_index(fresh)
    if kind == "query":
        want = _get_trigram_index(fresh)
        assert G.graph["_trigram_index"]["ids"] == want["ids"]
        assert {g: list(a) for g, a in G.graph["_trigram_index"]["postings"].items()} \
            == {g: list(a) for g, a in want["postings"].items()}
    else:
        assert "_trigram_index" not in G.graph
    if kind == "explain":
        assert G.succ["a"]["b"] is G.pred["b"]["a"]


@pytest.mark.parametrize("data", [_REPEATED, _REVERSED, _KEYED],
                         ids=["repeated", "reversed", "keyed"])
def test_such_links_leave_path_and_query_to_a_parse(tmp_path, cache_home, monkeypatch, data):
    gp = _graph_file(tmp_path, data)
    load_lookup_graph(gp, "explain")
    _cached(gp)
    for kind in ("path", "query"):  # parsed, so both links of a pair survive
        assert _same(load_lookup_graph(gp, kind)[0], _networkx(data, kind))
    _no_parse(monkeypatch)
    assert _same(load_lookup_graph(gp, "explain")[0], _networkx(data, "explain"))


def test_a_cache_built_from_a_file_rewritten_meanwhile_is_not_kept(tmp_path, cache_home):
    gp = _graph_file(tmp_path, _PLAIN)
    cache = lookup_graph._cache_path(gp)
    stamp = lookup_graph._stamp(gp)
    data = copy.deepcopy(_PLAIN)
    data["nodes"].append({"id": "d", "label": "D()", "source_file": "d.c"})
    gp.write_text(json.dumps(data), encoding="utf-8")
    lookup_graph._write_cache_in_background(gp, cache, stamp)
    time.sleep(0.5)
    assert not cache.exists()


def test_a_changed_graph_is_parsed_again(tmp_path, cache_home):
    gp = _graph_file(tmp_path, _PLAIN)
    load_lookup_graph(gp, "explain")
    _cached(gp)
    data = copy.deepcopy(_PLAIN)
    data["nodes"].append({"id": "d", "label": "D()", "source_file": "d.c"})
    gp.write_text(json.dumps(data), encoding="utf-8")
    G, _ = load_lookup_graph(gp, "explain")
    assert "d" in G


def test_a_cache_whose_header_does_not_match_is_not_read(tmp_path, cache_home):
    gp = _graph_file(tmp_path, _PLAIN)
    cache = lookup_graph._cache_path(gp)
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'["not", "this", "graph"]\n' + b"\x00garbage")
    G, _ = load_lookup_graph(gp, "explain")
    assert set(G) == {"a", "b", "c", "ghost"}


def test_the_cache_can_be_turned_off(tmp_path, cache_home, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_NO_LOOKUP_CACHE", "1")
    gp = _graph_file(tmp_path, _PLAIN)
    load_lookup_graph(gp, "explain")
    time.sleep(0.3)
    assert not (cache_home / "graphify").exists()


@pytest.mark.parametrize("directed", [True, False])
def test_sorted_graph_is_add_nodes_and_edges_from_sorted(directed):
    import networkx as nx

    from graphify.lookup_graph import sorted_graph
    nodes = ["c", "a", "b"]
    first = [("b", "a"), ("a", "c"), ("b", "a"), ("a", "z")]  # repeated, and a node not listed
    second = [("c", "a"), ("a", "b")]
    want = nx.DiGraph() if directed else nx.Graph()
    want.add_nodes_from(sorted(nodes))
    want.add_edges_from(sorted(first))
    want.add_edges_from(sorted(second))
    got = sorted_graph(nodes, first, second, directed=directed)
    assert _same(got, want)
