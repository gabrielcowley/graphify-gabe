"""C struct, union and enum definitions are type nodes, with their members.

The C extractor had no type support, so a plain C header such as
pokeemerald's include/global.h contributed nothing but its file node, and
only a header that happened to contain a C++-looking token (`template` in
`SpriteTemplate`) was routed to the C++ extractor and kept its structs. Now a
definition is a node with a `defines` edge per field, function-pointer fields
included, and an enum a `case_of` edge per enumerator. A typedef'd anonymous
struct takes its typedef's name. A forward declaration or a parameter's
`struct X *` is a use, not a definition, so it mints no node, and a struct
named like a function of its file gets an id of its own.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

_SRC = (
    "struct Sprite;\n"                                                  # 1
    "struct Sprite\n"                                                   # 2
    "{\n"                                                               # 3
    "    u16 x, y;\n"                                                   # 4
    "    Coords (*callback)(struct Sprite *);\n"                        # 5
    "    u8 ALIGNED(2) name[8];\n"                                      # 6
    "    union { u8 a; u16 b; } data;\n"                                # 7
    "};\n"                                                              # 8
    "typedef struct { int unused; } Coords;\n"                         # 9
    "enum Weather { WEATHER_NONE, WEATHER_RAIN = 3 };\n"                # 10
    "struct CableCar { u8 timer; };\n"                                  # 11
    "void CableCar(void) {}\n"                                          # 12
    "void Move(struct Sprite *sprite) { sprite->callback(sprite); }\n"   # 13
    "struct Timers\n"                                                   # 14
    "{\n"                                                               # 15
    "    u8 ALIGNED(4) state;\n"                                         # 16
    "    u16 ALIGNED(4) timer;\n"                                        # 17
    "    u8 UNUSED spare;\n"                                             # 18
    "};\n"                                                              # 19
)


def _graph(tmp_path: Path):
    p = tmp_path / "src" / "sprite.h"
    p.parent.mkdir(parents=True)
    p.write_text(_SRC, encoding="utf-8")
    other = tmp_path / "src" / "use.c"
    other.write_text("void Use(struct Sprite *s) { s->callback(s); }\n", encoding="utf-8")
    r = extract([p, other], root=tmp_path, cache_root=tmp_path / "out")
    nodes = {n["id"]: n for n in r["nodes"]}
    edges = {(nodes[e["source"]]["label"], e["relation"], nodes[e["target"]]["label"],
              e["source_location"])
             for e in r["edges"] if e["source"] in nodes and e["target"] in nodes}
    return list(nodes.values()), edges


def test_a_struct_is_a_node_at_its_definition_with_its_fields(tmp_path: Path):
    nodes, edges = _graph(tmp_path)
    sprites = [n for n in nodes if n["label"] == "Sprite"]
    # one node, at the definition, not at the forward declaration or a use
    assert [(n["source_file"], n["source_location"]) for n in sprites] == [("src/sprite.h", "L2")]
    fields = {(t, loc) for s, r, t, loc in edges if s == "Sprite" and r == "defines"}
    assert fields == {("Sprite.x", "L4"), ("Sprite.y", "L4"), ("Sprite.callback", "L5"),
                      ("Sprite.name", "L6"), ("Sprite.data", "L7")}


def test_an_attribute_macro_does_not_hide_a_field(tmp_path: Path):
    _nodes, edges = _graph(tmp_path)
    fields = {(t, loc) for s, r, t, loc in edges if s == "Timers" and r == "defines"}
    assert fields == {("Timers.state", "L16"), ("Timers.timer", "L17"), ("Timers.spare", "L18")}


def test_a_call_through_a_function_pointer_field_is_no_call_of_the_field(tmp_path: Path):
    _nodes, edges = _graph(tmp_path)
    assert not any(s in ("Move()", "Use()") and r == "calls" for s, r, _t, _l in edges)


def test_a_typedef_struct_and_an_enum(tmp_path: Path):
    _nodes, edges = _graph(tmp_path)
    assert ("Coords", "defines", "Coords.unused", "L9") in edges
    # a function-pointer field is a field, and references its return type
    assert ("Sprite", "references", "Coords", "L5") in edges
    assert ("Weather", "case_of", "WEATHER_NONE", "L10") in edges
    assert ("Weather", "case_of", "WEATHER_RAIN", "L10") in edges


def test_a_struct_named_like_a_function_keeps_both(tmp_path: Path):
    nodes, edges = _graph(tmp_path)
    labels = [n["label"] for n in nodes]
    assert "CableCar" in labels and "CableCar()" in labels
    assert ("CableCar", "defines", "CableCar.timer", "L11") in edges


def test_a_struct_named_by_its_tag_is_referenced(tmp_path: Path):
    _nodes, edges = _graph(tmp_path)
    # Move's parameter (here) and Use's (in use.c) name `struct Sprite`
    assert ("Move()", "references", "Sprite", "L13") in edges
    assert ("Use()", "references", "Sprite", "L1") in edges


def test_a_c_header_naming_templates_stays_c(tmp_path: Path):
    from graphify.extract import _get_extractor
    h = tmp_path / "sprite.h"
    h.write_text(
        "// a class of sprites, not C++, as in Gen3::Sprite; protected: by nothing\n"
        "struct SpriteTemplate { int tileTag; };\n"
        "extern const struct SpriteTemplate gDummySpriteTemplate;\n",
        encoding="utf-8",
    )
    assert _get_extractor(h).__name__ == "extract_c"
    cpp = tmp_path / "box.h"
    cpp.write_text("template <typename T> struct Box { T v; };\n", encoding="utf-8")
    assert _get_extractor(cpp).__name__ == "extract_cpp"


def test_a_field_is_looked_up_as_type_dot_field_or_by_its_bare_name(tmp_path: Path):
    from graphify.build import build_from_json
    from graphify.serve import _resolve_single_node

    p = tmp_path / "src" / "save.h"
    p.parent.mkdir(parents=True)
    p.write_text("struct SaveBlock2 { u8 playerName[8]; u8 playerGender; };\n"
                 "struct Mail { u8 playerName[8]; };\n", encoding="utf-8")
    G = build_from_json(extract([p], root=tmp_path, cache_root=tmp_path / "out"))
    nid, _ = _resolve_single_node(G, "SaveBlock2.playerName")
    assert nid is not None and G.nodes[nid]["label"] == "SaveBlock2.playerName"
    nid, _ = _resolve_single_node(G, "playerGender")
    assert nid is not None and G.nodes[nid]["label"] == "SaveBlock2.playerGender"
    nid, msg = _resolve_single_node(G, "playerName")   # two structs have one
    assert nid is None and msg.startswith("Ambiguous")
