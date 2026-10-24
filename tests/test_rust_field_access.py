"""Rust struct fields as nodes, with the functions that read and write them.

Much of a Rust program's behaviour is state a field carries, and "what sets this
flag, and what reads it" had no answer in the graph: fields were not nodes. Now
a named field of a struct, union or struct-style variant is a `Type.field` node,
and `x.field` (a read, or a write as an assignment's left side) and a
`T { field: .. }` initialiser (a write) bind to it through the receiver's type,
found as `x.m()`'s is (tests/test_rust_typed_member_calls.py). A receiver the
pass cannot type adds no edge.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

_BATTLE = (
    "pub struct Battle {\n"
    "    pub finish_turn: bool,\n"
    "    pub turns: u32,\n"
    "}\n"
    "impl Battle {\n"
    "    pub fn new() -> Battle {\n"
    "        Battle { finish_turn: false, turns: 0 }\n"
    "    }\n"
    "    pub fn finish_turn(&self) -> bool { self.finish_turn }\n"
    "    pub fn tick(&mut self) {\n"
    "        self.turns += 1;\n"
    "    }\n"
    "}\n"
    "pub enum Refusal {\n"
    "    UnmodelledWrite { what: u32 },\n"
    "}\n"
)


def _extract(tmp_path: Path, files: dict[str, str]) -> dict:
    paths = []
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        paths.append(p)
    return extract(paths, root=tmp_path, cache_root=tmp_path / "graphify-out")


def _uses(r: dict, relation: str) -> set[tuple[str, str, str]]:
    """(function, field, line) for each `relation` edge."""
    nodes = {n["id"]: n for n in r["nodes"]}
    return {
        (nodes[e["source"]]["label"], nodes[e["target"]]["label"], e["source_location"])
        for e in r["edges"]
        if e["relation"] == relation and e["source"] in nodes and e["target"] in nodes
    }


def test_fields_are_nodes_contained_by_their_type(tmp_path: Path):
    r = _extract(tmp_path, {"src/lib.rs": "mod battle;\n", "src/battle.rs": _BATTLE})
    nodes = {n["id"]: n for n in r["nodes"]}
    labels = {n["label"] for n in r["nodes"]}
    assert {"Battle.finish_turn", "Battle.turns", "UnmodelledWrite.what"} <= labels
    # A field and a method of one name are two nodes.
    assert ".finish_turn()" in labels
    contains = {(nodes[e["source"]]["label"], nodes[e["target"]]["label"])
                for e in r["edges"] if e["relation"] == "contains"
                and e["source"] in nodes and e["target"] in nodes}
    assert ("Battle", "Battle.finish_turn") in contains
    assert ("UnmodelledWrite", "UnmodelledWrite.what") in contains


def test_reads_writes_and_initialisers_bind_through_the_receivers_type(tmp_path: Path):
    r = _extract(tmp_path, {
        "src/lib.rs": "mod battle;\nmod world;\n",
        "src/battle.rs": _BATTLE,
        "src/world.rs": (
            "use crate::battle::Battle;\n"
            "pub struct World { battle: Battle }\n"
            "impl World {\n"
            "    pub fn finish_turn(&mut self) {\n"
            "        self.battle.finish_turn = true;\n"
            "    }\n"
            "    pub fn turns(&self) -> u32 {\n"
            "        self.battle\n"
            "            .turns\n"
            "    }\n"
            "}\n"
            "pub fn by_param(b: &Battle) -> bool { b.finish_turn }\n"
        ),
    })
    writes, reads = _uses(r, "writes"), _uses(r, "reads")
    assert (".finish_turn()", "Battle.finish_turn", "L5") in writes   # through a field
    assert (".tick()", "Battle.turns", "L11") in writes                # `+=`
    assert (".new()", "Battle.finish_turn", "L7") in writes            # an initialiser
    assert ("by_param()", "Battle.finish_turn", "L12") in reads        # a typed parameter
    assert (".finish_turn()", "Battle.finish_turn", "L9") in reads     # `self.` in its impl
    # A chain's field is on its own line, not the receiver's.
    assert (".turns()", "Battle.turns", "L9") in reads
    assert (".turns()", "World.battle", "L8") in reads


def test_if_let_some_binds_the_held_type_and_filter_keeps_it(tmp_path: Path):
    r = _extract(tmp_path, {
        "src/lib.rs": "mod battle;\nmod session;\n",
        "src/battle.rs": _BATTLE,
        "src/session.rs": (
            "use crate::battle::Battle;\n"
            "pub struct Session { stage: Option<Battle> }\n"
            "impl Session {\n"
            "    pub fn battle_mut(&mut self) -> Option<&mut Battle> { self.stage.as_mut() }\n"
            "    pub fn step(&mut self) {\n"
            "        if let Some(battle) = self.battle_mut().filter(|b| b.finish_turn) {\n"
            "            battle.finish_turn = false;\n"
            "        }\n"
            "    }\n"
            "}\n"
        ),
    })
    assert (".step()", "Battle.finish_turn", "L7") in _uses(r, "writes")
    # The closure's `b` is untyped: no edge, rather than a guess.
    assert not any(line == "L6" for _f, _t, line in _uses(r, "reads"))


def test_a_type_name_two_modules_declare_narrows_to_the_one_the_file_uses(tmp_path: Path):
    # Two `World`s in one crate: neither the caller's file nor its crate picks
    # one, but `use super::world::World;` does, for fields and methods alike.
    r = _extract(tmp_path, {
        "src/lib.rs": "mod battle;\nmod field;\nmod script;\n",
        "src/battle.rs": _BATTLE,
        "src/field.rs": "pub struct World { pub map: u32 }\n",
        "src/script/mod.rs": "pub mod world;\npub mod catch;\n",
        "src/script/world.rs": (
            "use crate::battle::Battle;\n"
            "pub struct World { pub battle: Battle }\n"
            "impl World { pub fn tick_all(&mut self) {} }\n"
        ),
        "src/script/catch.rs": (
            "use super::world::World;\n"
            "impl World {\n"
            "    pub fn finish_turn(&mut self) {\n"
            "        self.battle.finish_turn = true;\n"
            "    }\n"
            "}\n"
            "pub fn run(w: &mut World) { w.tick_all(); }\n"
        ),
    })
    assert (".finish_turn()", "Battle.finish_turn", "L4") in _uses(r, "writes")
    nodes = {n["id"]: n for n in r["nodes"]}
    calls = {(nodes[e["source"]]["label"], nodes[e["target"]]["label"])
             for e in r["edges"] if e["relation"] == "calls"
             and e["source"] in nodes and e["target"] in nodes}
    assert ("run()", ".tick_all()") in calls


def test_an_untyped_receiver_adds_no_edge(tmp_path: Path):
    r = _extract(tmp_path, {
        "src/lib.rs": "mod battle;\nmod use_it;\n",
        "src/battle.rs": _BATTLE,
        "src/use_it.rs": (
            "pub fn any(items: Vec<u32>) -> usize {\n"
            "    items.iter().map(|x| x.turns).count()\n"
            "}\n"
        ),
    })
    assert not any(f == "any()" for f, _t, _l in _uses(r, "reads"))
