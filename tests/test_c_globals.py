"""C file-scope variables are nodes, with the functions that read and write
them and the globals whose initializers name them.

A variable a file defines at file scope is a node (`_c_var`: "global", or
"static" when only its file sees it); an `extern` declaration, a prototype, a
typedef and a header's `T x;` without a value are not. A name a function uses
is the global of its own file when it defines one, else the one definition in
the corpus that is not another file's `static`; a local or parameter of the
name is not the global. Changing the variable's own storage is a write, through
`[i]`, `.` and parentheses (`gTasks[id].func = f`); through `->` it is the
pointer that is read. An initializer naming a global `references` it.
INFERRED, one edge per pair.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

_H = (
    "struct Pos { int x; };\n"                                              # 1
    "extern struct Pos *gPosPtr;\n"                                         # 2
    "extern int gCount;\n"                                                  # 3
    "int gTentative;\n"                                                     # 4
)
_A = (
    '#include "g.h"\n'                                                      # 1
    "EWRAM_DATA int gCount = 0;\n"                                          # 2
    "COMMON_DATA const struct Pos *gPosPtr = NULL;\n"                       # 3
    "static EWRAM_DATA u8 sState;\n"                                        # 4
    "static void (*sCb)(void);\n"                                           # 5
    "struct Task gTasks[4];\n"                                              # 6
    "static const u8 sPic[] = {1};\n"                                       # 7
    "static const u8 *const sPics[] = { sPic, NULL };\n"                    # 8
    "int gTentative = 1;\n"                                                 # 9
    "void F(void)\n"                                                        # 10
    "{\n"                                                                   # 11
    "    gCount++;\n"                                                       # 12
    "    gPosPtr->x = 1;\n"                                                 # 13
    "    sState = gCount;\n"                                                # 14
    "    (gTasks[0]).func = F;\n"                                           # 15
    "    sCb();\n"                                                          # 16
    "    Draw(sPics[sState]);\n"                                            # 17
    "}\n"                                                                   # 18
    "static EWRAM_DATA ALIGNED(4) u8 sAligned = 0;\n"                       # 19
    "EWRAM_DATA u16 ALIGNED(4) gBuf[2] = {0};\n"                            # 20
    "void Z(void) { sAligned = gBuf[0]; }\n"                                # 21
)
_B = (
    '#include "g.h"\n'                                                      # 1
    "static u8 sState;\n"                                                   # 2
    "int G(int sPic) { int gCount = 0; return gCount + sState + sPic + gPosPtr->x; }\n"  # 3
    "void H(void) { gPosPtr = 0; }\n"                                       # 4
    "int K(void) { return sCb != 0; }\n"                                    # 5
    "extern int gCount;\n"                                                  # 6
    "void M(void)\n"                                                        # 7
    "{\n"                                                                   # 8
    "    if (gCount)\n"                                                     # 9
    "        gCount = 0;\n"                                                 # 10
    "}\n"                                                                   # 11
)


def _graph(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for name, text in (("g.h", _H), ("a.c", _A), ("b.c", _B)):
        (src / name).write_text(text, encoding="utf-8")
    r = extract([src / "g.h", src / "a.c", src / "b.c"], root=tmp_path, cache_root=tmp_path / "out")
    nodes = {n["id"]: n for n in r["nodes"]}
    edges = {
        (nodes[e["source"]]["label"], e["relation"], nodes[e["target"]]["label"],
         nodes[e["target"]]["source_file"].rsplit("/", 1)[-1], e["source_location"])
        for e in r["edges"]
        if e.get("context") == "global" and e["source"] in nodes and e["target"] in nodes
    }
    return nodes, edges


def test_a_file_scope_definition_is_a_node_and_a_declaration_is_not(tmp_path: Path):
    nodes, _ = _graph(tmp_path)
    got = sorted((n["label"], n["source_file"].rsplit("/", 1)[-1], n["_c_var"])
                 for n in nodes.values() if n.get("_c_var"))
    assert got == [
        ("gBuf", "a.c", "global"),       # behind EWRAM_DATA u16 ALIGNED(4)
        ("gCount", "a.c", "global"),     # behind EWRAM_DATA
        ("gPosPtr", "a.c", "global"),    # behind COMMON_DATA const struct
        ("gTasks", "a.c", "global"),
        ("gTentative", "a.c", "global"),  # not the header's `int gTentative;`
        ("sAligned", "a.c", "static"),   # behind EWRAM_DATA ALIGNED(4)
        ("sCb", "a.c", "static"),        # a function pointer, not a prototype
        ("sPic", "a.c", "static"),
        ("sPics", "a.c", "static"),
        ("sState", "a.c", "static"),
        ("sState", "b.c", "static"),
    ]   # b.c's `extern int gCount;` is not one


def test_writing_the_variable_and_reading_it(tmp_path: Path):
    _, edges = _graph(tmp_path)
    assert ("F()", "writes", "gCount", "a.c", "L12") in edges    # ++ is a write
    assert ("F()", "reads", "gPosPtr", "a.c", "L13") in edges    # -> reads the pointer
    assert ("F()", "writes", "sState", "a.c", "L14") in edges
    assert ("F()", "writes", "gTasks", "a.c", "L15") in edges    # through [i], () and .
    assert ("F()", "reads", "sCb", "a.c", "L16") in edges        # called through
    assert ("H()", "writes", "gPosPtr", "a.c", "L4") in edges
    # the line after `static EWRAM_DATA ALIGNED(4) u8 sAligned` still parses
    assert ("Z()", "writes", "sAligned", "a.c", "L21") in edges
    assert ("Z()", "reads", "gBuf", "a.c", "L21") in edges


def test_one_edge_per_function_and_global_a_write_kept_over_a_read(tmp_path: Path):
    _, edges = _graph(tmp_path)
    assert {(r, loc) for s, r, t, _f, loc in edges if s == "F()" and t == "gCount"} \
        == {("writes", "L12")}
    # a read, then a write: the edge is the write
    assert {(r, loc) for s, r, t, _f, loc in edges if s == "M()"} == {("writes", "L10")}


def test_a_static_is_its_files_and_a_local_is_not_the_global(tmp_path: Path):
    _, edges = _graph(tmp_path)
    g = {(r, t, f) for s, r, t, f, _l in edges if s == "G()"}
    # b.c's own sState; the local gCount and the parameter sPic are not globals
    assert g == {("reads", "sState", "b.c"), ("reads", "gPosPtr", "a.c")}
    # a.c's static sCb is not b.c's to read
    assert not any(s == "K()" for s, *_ in edges)


def test_an_initializer_naming_a_global_references_it(tmp_path: Path):
    _, edges = _graph(tmp_path)
    assert ("sPics", "references", "sPic", "a.c", "L8") in edges
    assert ("F()", "reads", "sPics", "a.c", "L17") in edges


def test_an_unchanged_files_global_still_binds_a_changed_files_use(tmp_path: Path):
    import json

    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in (("g.h", _H), ("a.c", _A), ("b.c", _B)):
        (corpus / name).write_text(text, encoding="utf-8")
    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False) is True
    b = corpus / "b.c"
    b.write_text(_B.replace("gPosPtr = 0;", "gPosPtr = 0; gCount = 2;"), encoding="utf-8")
    assert _rebuild_code(corpus, changed_paths=[b], no_cluster=True, acquire_lock=False) is True
    g = json.loads((corpus / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    got = {(nodes[e["source"]]["label"], e["relation"], nodes[e["target"]]["label"],
            nodes[e["target"]]["source_file"].rsplit("/", 1)[-1])
           for e in g["links"] if e.get("context") == "global"}
    # a.c's gCount and gPosPtr, not b.c's (it has none), from the unchanged graph
    assert ("H()", "writes", "gCount", "a.c") in got
    assert ("H()", "writes", "gPosPtr", "a.c") in got
    assert ("G()", "reads", "sState", "b.c") in got
