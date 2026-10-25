"""C definitions the parser or the id recipe used to lose.

- Names that differ only in case or in a leading or trailing underscore
  (`Footer_` and `Footer`) share an id under the recipe, so one of the two was
  lost and the other took both bodies' edges. Such a group is salted, the one
  bare name keeping the plain id; fields of one struct likewise.
- A capitalised macro between a declaration's type and its name
  (`static bool8 UNUSED F(u32 x)`) is blanked, but not a type after a keyword
  (`static const INIT_PARAM sX`, `struct FOO x`).
- In a definition the parser rejected, an `#if` chain inside a body is reduced
  to the branch a build defining none of its macros compiles, and a file-level
  `name(...)` line with no `;` (a macro expanding to definitions) is blanked.
  Definitions that parsed keep every branch.
- A function returning a function pointer is named by its inner declarator.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

_C = (
    "struct Feed { u8 monSpriteId_; u8 monSpriteId; u8 Hp; u8 hp; };\n"             # 1
    "static u8 sTimer;\n"                                                         # 2
    "static u8 sTimer_;\n"                                                        # 3
    "static void Footer_(void) { Draw(); }\n"                                     # 4
    "void Footer(void) { Footer_(); }\n"                                          # 5
    "static void _Patch(void) {}\n"                                               # 6
    "void Patch(void) { _Patch(); }\n"                                            # 7
    "void Lower(void) {}\n"                                                       # 8
    "void lower(void) { Lower(); }\n"                                             # 9
    "void Use(struct Feed *f) { f->monSpriteId_ = sTimer_; f->hp = sTimer; }\n"   # 10
    "static bool8 UNUSED TakePowder_(u32 cost) { return Spend(cost); }\n"         # 11
    "static const INIT_PARAM sTemplate = { 0 };\n"                                # 12
    "struct FOO sFoo;\n"                                                          # 13
    "static void Split(void)\n"                                                   # 14
    "{\n"                                                                         # 15
    "#ifndef UBFIX\n"                                                             # 16
    "    if (Vanilla())\n"                                                        # 17
    "#else\n"                                                                     # 18
    "    if (Fixed())\n"                                                          # 19
    "#endif\n"                                                                    # 20
    "        Then();\n"                                                           # 21
    "#ifdef BUGFIX\n"                                                             # 22
    "    Fix();\n"                                                                # 23
    "#else\n"                                                                     # 24
    "    Bug();\n"                                                                # 25
    "#endif\n"                                                                    # 26
    "}\n"                                                                         # 27
    "static void Whole(void)\n"                                                   # 28
    "{\n"                                                                         # 29
    "#ifdef BUGFIX\n"                                                             # 30
    "    u8 UNUSED spare; Fix();\n"                                               # 31
    "#else\n"                                                                     # 32
    "    Bug();\n"                                                                # 33
    "#endif\n"                                                                    # 34
    "}\n"                                                                         # 35
    "movement_type_empty_callback(MovementType_None)\n"                           # 36
    "movement_type_def(MovementType_Wander, gMovementTypeFuncs_Wander)\n"         # 37
    "bool8 MovementType_Wander_Step0(struct Sprite *sprite)\n"                    # 38
    "{\n"                                                                         # 39
    "    Clear(sprite);\n"                                                        # 40
    "    return TRUE;\n"                                                          # 41
    "}\n"                                                                         # 42
    "static void (*GetGfxFunc(void))(void)\n"                                     # 43
    "{\n"                                                                         # 44
    "    Helper();\n"                                                             # 45
    "    return Draw;\n"                                                          # 46
    "}\n"                                                                         # 47
    "void Draw(void) {}\n"                                                        # 48
    "void Helper(void) {}\n"                                                      # 49
    # the callees, so the calls above have somewhere to land
    "u8 Spend(u32 c) { return 0; }\n"                                             # 50
    "u8 Vanilla(void) { return 0; }\n"                                            # 51
    "u8 Fixed(void) { return 0; }\n"                                              # 52
    "void Then(void) {}\n"                                                        # 53
    "void Fix(void) {}\n"                                                         # 54
    "void Bug(void) {}\n"                                                         # 55
    "void Clear(struct Sprite *s) {}\n"                                           # 56
    "static EWRAM_DATA s16 UNUSED sUnkVar = 0;\n"                                 # 57
    "static const struct BgTemplate UNUSED sBgTemplates[] = { 0 };\n"             # 58
    "void Reset(void) { sUnkVar = 0; Bump(sBgTemplates); }\n"                     # 59
)


def _graph(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.c").write_text(_C, encoding="utf-8")
    r = extract([src / "a.c"], root=tmp_path, cache_root=tmp_path / "out")
    nodes = {n["id"]: n for n in r["nodes"]}
    edges = {(nodes[e["source"]]["label"], e["relation"], nodes[e["target"]]["label"],
              e["source_location"])
             for e in r["edges"] if e["source"] in nodes and e["target"] in nodes}
    return nodes, edges


def test_names_one_id_apart_are_each_a_node(tmp_path: Path):
    nodes, edges = _graph(tmp_path)
    by_label = {n["label"]: nid for nid, n in nodes.items()}
    for twin in ("Footer_()", "Footer()", "_Patch()", "Patch()", "Lower()", "lower()",
                 "sTimer", "sTimer_", "Feed.monSpriteId_", "Feed.monSpriteId",
                 "Feed.Hp", "Feed.hp"):
        assert twin in by_label, twin
    # the one bare name keeps the plain id
    assert by_label["Footer()"] == "src_a_footer" and by_label["Footer_()"] != "src_a_footer"
    assert by_label["Patch()"] == "src_a_patch"
    assert by_label["sTimer"] == "src_a_stimer"
    assert by_label["Feed.monSpriteId"] == "src_a_feed_field_monspriteid"
    # and each body's edges are its own
    assert ("Footer()", "calls", "Footer_()", "L5") in edges
    assert ("Footer_()", "calls", "Draw()", "L4") in edges
    assert ("Patch()", "calls", "_Patch()", "L7") in edges
    assert ("lower()", "calls", "Lower()", "L9") in edges
    assert ("Use()", "writes", "Feed.monSpriteId_", "L10") in edges
    assert ("Use()", "writes", "Feed.hp", "L10") in edges
    assert ("Use()", "reads", "sTimer_", "L10") in edges
    assert ("Use()", "reads", "sTimer", "L10") in edges


def test_a_macro_between_type_and_name_and_a_type_after_a_keyword(tmp_path: Path):
    nodes, edges = _graph(tmp_path)
    labels = {n["label"] for n in nodes.values()}
    assert "TakePowder_()" in labels
    assert ("TakePowder_()", "calls", "Spend()", "L11") in edges
    assert {"sTemplate", "sFoo"} <= labels   # INIT_PARAM and FOO are their types
    # after EWRAM_DATA, and after a struct's tag
    assert ("Reset()", "writes", "sUnkVar", "L59") in edges
    assert ("Reset()", "reads", "sBgTemplates", "L59") in edges


def test_a_rejected_body_keeps_the_vanilla_branch_of_its_if_chains(tmp_path: Path):
    _, edges = _graph(tmp_path)
    split = {(r, t) for s, r, t, _l in edges if s == "Split()"}
    assert {("calls", "Vanilla()"), ("calls", "Then()"), ("calls", "Bug()")} <= split
    assert not {("calls", "Fixed()"), ("calls", "Fix()")} & split
    # a body that parsed keeps both branches, as before; `u8 UNUSED spare`
    # is no error that would make it one that did not
    whole = {(r, t) for s, r, t, _l in edges if s == "Whole()"}
    assert {("calls", "Fix()"), ("calls", "Bug()")} <= whole


def test_a_file_level_macro_line_does_not_swallow_the_next_function(tmp_path: Path):
    nodes, edges = _graph(tmp_path)
    labels = {n["label"] for n in nodes.values()}
    assert "MovementType_Wander_Step0()" in labels
    assert "movement_type_def()" not in labels
    assert ("MovementType_Wander_Step0()", "calls", "Clear()", "L40") in edges


def test_a_function_returning_a_function_pointer(tmp_path: Path):
    _, edges = _graph(tmp_path)
    assert ("GetGfxFunc()", "calls", "Helper()", "L45") in edges
    assert ("GetGfxFunc()", "indirect_call", "Draw()", "L46") in edges
