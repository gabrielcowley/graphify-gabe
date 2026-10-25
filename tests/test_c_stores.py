"""A C global or field passed where a call stores is written.

`CpuFill16(0, &gX, size)`, `memcpy(&gX[2], ...)` and `StringCopy(gStringVar1,
...)` change the variable's own storage, as an assignment does. Which argument
a call stores through is `_C_DESTINATION_ARGS`: the C library's, the GBA SDK
macros' and the pret decomps' string, decompression and `GetMonData` helpers.
Its address passed there (`&gX`, `&gX[i]`, `&gX.f`) writes it; passed itself, it
is written only when it is an array (`_c_array`), since through a pointer it is
the pointer that is read; an element of it (`gNames[1]`) is left a read.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify.extract import extract

_A = (
    "struct Pos { int x; u8 name[8]; u8 *tag; };\n"                          # 1
    "u8 gStringVar1[16];\n"                                                   # 2
    "struct Pos gPos;\n"                                                      # 3
    "struct Pos *gPosPtr;\n"                                                  # 4
    "u16 gArr[4];\n"                                                          # 5
    "u8 gNames[2][8];\n"                                                      # 6
    "u16 gSrc[4];\n"                                                          # 7
    "u16 gDst[4];\n"                                                          # 8
    "void Fill(void) { CpuFill16(0, &gPos, sizeof(gPos)); }\n"                # 9
    "void Name(const u8 *s) { StringCopy(gStringVar1, s); }\n"                # 10
    "void Ptr(const u8 *s) { memcpy(gPosPtr, s, 4); }\n"                      # 11
    "void Elem(const u8 *s) { memcpy((void *)&gArr[2], s, 2); }\n"            # 12
    "void Field(const u8 *s) { StringCopy(gPosPtr->name, s); StringCopy(gPosPtr->tag, s); }\n"  # 13
    "void Row(const u8 *s) { StringCopy(gNames[1], s); }\n"                   # 14
    "void Dma(void) { DmaCopy16(3, &gSrc, (gDst), sizeof(gDst)); }\n"         # 15
    "void Addr(void) { Clear(&gArr); CpuFill16(0, &gPos.x, 4); }\n"           # 16
)


def _edges(r) -> set:
    nodes = {n["id"]: n for n in r["nodes"]}
    return {(nodes[e["source"]]["label"], e["relation"], nodes[e["target"]]["label"],
             e["source_location"])
            for e in r["edges"]
            if e.get("context") in ("global", "field")
            and e["source"] in nodes and e["target"] in nodes}


def test_a_global_passed_where_a_call_stores_is_written(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.c").write_text(_A, encoding="utf-8")
    r = extract([src / "a.c"], root=tmp_path, cache_root=tmp_path / "out")
    arrays = sorted(n["label"] for n in r["nodes"] if n.get("_c_array"))
    assert arrays == ["Pos.name", "gArr", "gDst", "gNames", "gSrc", "gStringVar1"]
    edges = _edges(r)
    assert ("Fill()", "writes", "gPos", "L9") in edges        # its address
    assert ("Name()", "writes", "gStringVar1", "L10") in edges  # an array, itself
    assert ("Ptr()", "reads", "gPosPtr", "L11") in edges      # a pointer: what it points at
    assert ("Elem()", "writes", "gArr", "L12") in edges       # through a cast and [i]
    assert ("Field()", "writes", "Pos.name", "L13") in edges  # an array field
    assert ("Field()", "reads", "Pos.tag", "L13") in edges    # a pointer field
    assert ("Field()", "reads", "gPosPtr", "L13") in edges
    assert ("Row()", "reads", "gNames", "L14") in edges       # an element is left a read
    assert ("Dma()", "reads", "gSrc", "L15") in edges         # the source
    assert ("Dma()", "writes", "gDst", "L15") in edges        # in parentheses
    assert ("Addr()", "reads", "gArr", "L16") in edges        # not a call that stores
    assert ("Addr()", "writes", "gPos", "L16") in edges       # `&gPos.x`
    assert ("Addr()", "writes", "Pos.x", "L16") in edges


def test_a_changed_file_fills_an_unchanged_files_array(tmp_path: Path):
    from graphify.watch import _rebuild_code

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.c").write_text(_A, encoding="utf-8")
    b = corpus / "b.c"
    b.write_text("void B(void) { }\n", encoding="utf-8")
    assert _rebuild_code(corpus, no_cluster=True, acquire_lock=False) is True
    b.write_text("void B(const u8 *s) { StringCopy(gStringVar1, s); memcpy(gPosPtr, s, 1); }\n",
                 encoding="utf-8")
    assert _rebuild_code(corpus, changed_paths=[b], no_cluster=True, acquire_lock=False) is True
    g = json.loads((corpus / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    got = {(nodes[e["source"]]["label"], e["relation"], nodes[e["target"]]["label"])
           for e in g["links"] if e.get("context") == "global"}
    # whether gStringVar1 is an array comes from the unchanged a.c
    assert ("B()", "writes", "gStringVar1") in got
    assert ("B()", "reads", "gPosPtr") in got
