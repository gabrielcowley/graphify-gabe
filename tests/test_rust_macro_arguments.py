"""The arguments of Rust's std expression macros are walked as code.

tree-sitter leaves a macro's arguments a flat token tree, so `matches!(op,
Op::A | Op::B)`, `assert_eq!(x, LIMIT)` and `format!("{}", f(x))` added no edge.
extract_rust now parses each argument list in a blanked copy of the file, at its
own offsets, as an expression (or, for `matches!`, an expression and a match
arm), and walks it like any other code. Anything that does not parse adds
nothing, and a macro the file defines itself is left alone.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

_LIB = (
    "pub const LIMIT: u32 = 3;\n"                                                       # 1
    "pub enum Op { DisplayDexInfo, TryGiveCaughtMonNick { at: u32 }, Other, Spare }\n"  # 2
    "pub struct Battle { pub turns: u32 }\n"                                            # 3
    "pub fn helper(x: u32) -> u32 { x }\n"                                              # 4
    "pub fn one() -> u32 { 1 }\n"                                                       # 5
    "pub fn two() -> u32 { 2 }\n"                                                       # 6
    "pub fn waits(op: &Op) -> bool {\n"                                                 # 7
    "    matches!(op, Op::DisplayDexInfo | Op::TryGiveCaughtMonNick { .. })\n"          # 8
    "}\n"                                                                               # 9
    "pub fn guard(op: &Op) -> bool {\n"                                                 # 10
    "    matches!(op, Op::TryGiveCaughtMonNick { at } if *at > LIMIT)\n"                # 11
    "}\n"                                                                               # 12
    "pub fn compares(b: Battle) {\n"                                                    # 13
    "    assert_eq!(b.turns, LIMIT);\n"                                                 # 14
    "    assert!(\n"                                                                    # 15
    "        helper(b.turns) > 0,\n"                                                    # 16
    "        \"msg {}\",\n"                                                             # 17
    "        format!(\"{}\", helper(LIMIT))\n"                                          # 18
    "    );\n"                                                                          # 19
    "    let v = vec![Op::Other; 2];\n"                                                 # 20
    "    println!(\"{}\", v.len());\n"                                                  # 21
    "}\n"                                                                               # 22
    "pub fn packed(){assert!(one()>0);assert!(two()>0);}\n"                             # 23
    "macro_rules! check { ($e:expr) => { $e }; }\n"                                     # 24
    "pub fn own_macro() { check!(Op::Spare); }\n"                                       # 25
)


def _edges(tmp_path: Path) -> set[tuple[str, str, str, str]]:
    p = tmp_path / "src" / "lib.rs"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_LIB, encoding="utf-8")
    r = extract([p], root=tmp_path, cache_root=tmp_path / "graphify-out")
    nodes = {n["id"]: n for n in r["nodes"]}
    return {
        (nodes[e["source"]]["label"], e["relation"], nodes[e["target"]]["label"], e["source_location"])
        for e in r["edges"] if e["source"] in nodes and e["target"] in nodes
    }


def test_a_matches_pattern_references_its_variants(tmp_path: Path):
    edges = _edges(tmp_path)
    assert ("waits()", "references", "DisplayDexInfo", "L8") in edges
    assert ("waits()", "references", "TryGiveCaughtMonNick", "L8") in edges
    # and a guard is an expression: the constant it compares
    assert ("guard()", "references", "LIMIT", "L11") in edges


def test_assertion_and_format_arguments_are_code_at_their_own_lines(tmp_path: Path):
    edges = _edges(tmp_path)
    assert ("compares()", "references", "LIMIT", "L14") in edges
    assert ("compares()", "reads", "Battle.turns", "L14") in edges
    assert ("compares()", "calls", "helper()", "L16") in edges   # inside a multi-line assert!
    assert ("compares()", "references", "Other", "L20") in edges  # vec![x; n]


def test_macros_packed_on_one_line_are_each_walked(tmp_path: Path):
    edges = _edges(tmp_path)
    assert ("packed()", "calls", "one()", "L23") in edges
    assert ("packed()", "calls", "two()", "L23") in edges


def test_a_macro_the_file_defines_is_not_reparsed(tmp_path: Path):
    edges = _edges(tmp_path)
    assert not any(src == "own_macro()" and tgt == "Spare" for src, _r, tgt, _l in edges)
    assert ("own_macro()", "calls", "check!", "L25") in edges
