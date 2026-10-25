"""C field reads and writes: `x.f` and `x->f` are edges to the field `f` of the
struct `x`'s type names.

The receiver is typed from where its name was declared: a parameter or local
(`struct Sprite *sprite`), a global (`extern struct SaveBlock2
*gSaveBlock2Ptr;` in a header, or the file's own `static EWRAM_DATA struct
{ … } *sMenu`), a field of another receiver, a function's return, or a cast.
`[i]`, `*`, `&` and parentheses keep the type. An object-like macro can stand
for a receiver (`#define eContest (*gResources->contest)`) or for a field
(`#define tState data[0]`). Assigning or incrementing is a write; anything else
a read. INFERRED, one edge per function and field.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify.extract import extract

_H = (
    "struct Pos { int x; int y; };\n"                                       # 1
    "struct Task { void (*func)(unsigned char); short data[16]; };\n"       # 2
    "struct Mon { int hp; struct Pos pos; union { int a; short b; } u; };\n"  # 3
    "struct Resources { struct Mon *mon; };\n"                              # 4
    "extern struct Mon *gMonPtr;\n"                                         # 5
    "extern struct Task gTasks[16];\n"                                      # 6
    "extern struct Resources *gResources;\n"                                # 7
    "struct Mon *GetMon(void);\n"                                           # 8
    "#define eMon (*gResources->mon)\n"                                     # 9
    "#define pX x\n"                                                         # 10
)
_C = (
    '#include "mon.h"\n'                                                    # 1
    "#define tState data[0]\n"                                              # 2
    "static EWRAM_DATA struct {\n"                                          # 3
    "    int timer;\n"                                                      # 4
    "} *sMenu = NULL;\n"                                                    # 5
    "void Heal(struct Mon *mon)\n"                                          # 6
    "{\n"                                                                   # 7
    "    mon->hp = 10;\n"                                                   # 8
    "    gMonPtr->pos.x++;\n"                                               # 9
    "    gTasks[0].tState = mon->u.a;\n"                                    # 10
    "    sMenu->timer = GetMon()->hp + eMon.hp;\n"                          # 11
    "    ((struct Pos *)0)->y = (*mon).hp;\n"                               # 12
    "}\n"                                                                   # 13
    "int Peek(int hp) { struct Mon local; return local.hp + hp; }\n"       # 14
    "int Macro(void) { return eMon.hp; }\n"                                # 15
    "int Ret(void) { return GetMon()->pos.y; }\n"                          # 16
    "void Bump(struct Mon *m) { int h = m->hp; m->hp = h + 1; }\n"         # 17
    "void Move(void) { gMonPtr->pos.pX = 1; }\n"                           # 18
)


def _edges(tmp_path: Path) -> set[tuple[str, str, str, str]]:
    src = tmp_path / "src"
    src.mkdir()
    (src / "mon.h").write_text(_H, encoding="utf-8")
    (src / "battle.c").write_text(_C, encoding="utf-8")
    r = extract([src / "mon.h", src / "battle.c"], root=tmp_path, cache_root=tmp_path / "out")
    nodes = {n["id"]: n for n in r["nodes"]}
    return {
        (nodes[e["source"]]["label"], e["relation"],
         nodes[e["target"]]["label"], e["source_location"])
        for e in r["edges"]
        if e["relation"] in ("reads", "writes") and e["source"] in nodes and e["target"] in nodes
    }


def test_a_parameter_a_global_and_a_chain_type_their_fields(tmp_path: Path):
    edges = _edges(tmp_path)
    assert ("Heal()", "writes", "Mon.hp", "L8") in edges
    assert ("Heal()", "reads", "Mon.pos", "L9") in edges
    assert ("Heal()", "writes", "Pos.x", "L9") in edges    # ++ is a write


def test_an_array_element_a_macro_field_and_an_anonymous_union(tmp_path: Path):
    edges = _edges(tmp_path)
    assert ("Heal()", "writes", "Task.data", "L10") in edges   # tState is data[0]
    assert ("Heal()", "reads", "Mon.u", "L10") in edges
    assert ("Heal()", "reads", "u.a", "L10") in edges


def test_an_anonymous_struct_global_a_return_a_macro_receiver_and_a_cast(tmp_path: Path):
    edges = _edges(tmp_path)
    assert ("Ret()", "reads", "Mon.pos", "L16") in edges   # GetMon() returns a Mon
    assert ("Ret()", "reads", "Pos.y", "L16") in edges
    assert ("Heal()", "writes", "sMenu.timer", "L11") in edges
    assert ("Macro()", "reads", "Mon.hp", "L15") in edges   # eMon, #defined
    assert ("Heal()", "writes", "Pos.y", "L12") in edges


def test_one_edge_per_function_and_field_a_read_kept_beside_a_write(tmp_path: Path):
    edges = _edges(tmp_path)
    heal_hp = {(r, loc) for s, r, t, loc in edges if s == "Heal()" and t == "Mon.hp"}
    assert heal_hp == {("writes", "L8")}
    # a read then a write: the edge is the write
    assert {(r, t) for s, r, t, _l in edges if s == "Bump()"} == {("writes", "Mon.hp")}


def test_a_local_struct_is_typed_and_a_plain_parameter_is_not_a_receiver(tmp_path: Path):
    edges = _edges(tmp_path)
    assert {(s, r, t) for s, r, t, _l in edges if s == "Peek()"} == {("Peek()", "reads", "Mon.hp")}


def test_an_unchanged_header_still_types_a_changed_file(tmp_path: Path):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "mon.h").write_text(_H, encoding="utf-8")
    c = corpus / "battle.c"
    c.write_text(_C, encoding="utf-8")
    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False) is True
    c.write_text(_C.replace("mon->hp = 10;", "mon->hp = 11;"), encoding="utf-8")
    assert _rebuild_code(corpus, changed_paths=[c], no_cluster=True, acquire_lock=False) is True
    g = json.loads((corpus / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    labels = {n["id"]: n["label"] for n in g["nodes"]}
    got = {(labels[e["source"]], e["relation"], labels[e["target"]])
           for e in g["links"] if e["relation"] in ("reads", "writes")}
    # the header's struct, global and macro come from the unchanged graph
    assert ("Heal()", "writes", "Mon.hp") in got
    assert ("Heal()", "writes", "Pos.x") in got
    assert ("Heal()", "writes", "Task.data") in got
    assert ("Move()", "writes", "Pos.x") in got   # through the header's `#define pX x`


def test_a_global_whose_own_type_is_a_typedef_name_takes_the_headers(tmp_path: Path):
    # `COMMON_DATA LINK_MANAGER lman` in its .c, `extern struct linkManagerTag
    # lman` in the header: LINK_MANAGER names no type node, the struct does
    src = tmp_path / "src"
    src.mkdir()
    (src / "lman.h").write_text(
        "typedef struct linkManagerTag { int state; } LINK_MANAGER;\n"
        "extern struct linkManagerTag lman;\n", encoding="utf-8")
    (src / "lman.c").write_text(
        '#include "lman.h"\nCOMMON_DATA LINK_MANAGER lman = {0};\n'
        "void Reset(void) { lman.state = 0; }\n", encoding="utf-8")
    (src / "rfu.c").write_text(
        '#include "lman.h"\nint State(void) { return lman.state; }\n', encoding="utf-8")
    files = [src / "lman.h", src / "lman.c", src / "rfu.c"]
    r = extract(files, root=tmp_path, cache_root=tmp_path / "out")
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    got = {(labels[e["source"]], e["relation"], labels[e["target"]])
           for e in r["edges"] if e.get("context") == "field"}
    assert ("Reset()", "writes", "linkManagerTag.state") in got
    assert ("State()", "reads", "linkManagerTag.state") in got
