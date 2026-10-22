"""A Rust `use` lands on what its path names, through the crate's module tree.

extract_rust pointed each `use` at the bare last segment of its path, and #4439
reconnected that to a declaration only when the name was unique in the corpus.
A helper that eight test modules also define (`draws_between`) stayed a
dangling stub, and a bare call after the `use` had no import evidence, so it got
no edge. graphify/rust_paths.py now resolves the written path through the
workspace: crates from their `Cargo.toml`, modules from `mod x;` (and
`#[path]`), items from the nodes, and `pub use` re-exports. A path that names
exactly one item takes the edge; anything else keeps its old target.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify.extract import extract
from graphify.rust_paths import RustPathIndex

_WORKSPACE = {
    "Cargo.toml": '[workspace]\nmembers = ["a", "b"]\n',
    "a/Cargo.toml": '[package]\nname = "a"\nversion = "0.1.0"\nedition = "2021"\n',
    "b/Cargo.toml": (
        '[package]\nname = "b"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[dependencies]\na = { path = "../a" }\n'
    ),
    "a/src/lib.rs": (
        "pub mod expect;\npub mod fixture;\npub mod floor;\npub mod fs;\npub mod order;\npub mod other;\n"
        "pub mod rng;\npub mod session;\npub mod values;\n#[path = \"renamed.rs\"]\npub mod moved;\n"
        "pub use fixture::is_fixture_file;\n"
    ),
    "a/src/rng.rs": (
        "pub fn draws_between(from: u32, to: u32) -> u32 {\n    to - from\n}\n\n"
        "#[cfg(test)]\nmod tests {\n    fn draws_between(from: u32, to: u32) -> u32 {\n"
        "        super::draws_between(from, to)\n    }\n}\n"
    ),
    # The same name in another file's test module: by name alone, ambiguous.
    "a/src/order.rs": (
        "pub fn first() -> u32 {\n    crate::rng::draws_between(0, 2)\n}\n\n#[cfg(test)]\nmod tests {\n"
        "    fn draws_between(from: u32, to: u32) -> u32 {\n        to - from\n    }\n}\n"
    ),
    "a/src/fixture.rs": "pub fn is_fixture_file(name: &str) -> bool {\n    name.ends_with(\".toml\")\n}\n",
    "a/src/expect.rs": (
        "pub const LIMIT: u32 = 4;\n\npub enum ExpectError {\n    BattleNotStated,\n    Over(u32),\n}\n\n"
        "impl ExpectError {\n    pub const CAP: u32 = 9;\n\n    pub fn exceeded(n: u32) -> Self {\n        Self::Over(n)\n    }\n\n"
        "    pub fn kind(&self) -> &str {\n        match self {\n            Self::BattleNotStated => \"battle\",\n"
        "            Self::Over(_) => \"over\",\n        }\n    }\n}\n"
    ),
    "a/src/fs.rs": "pub fn read() {}\n",
    # An associated const in an impl block away from the type's declaration.
    "a/src/floor.rs": "use crate::expect::ExpectError;\n\nimpl ExpectError {\n    pub const FLOOR: u32 = 1;\n}\n",
    "a/src/renamed.rs": "pub fn relocated() {}\n",
    "a/src/other.rs": (
        "use crate::expect::{ExpectError, LIMIT};\nuse crate::moved::relocated;\n"
        "use crate::rng::draws_between;\nuse std::fs;\nuse std::fs::read;\nuse serde::Serialize;\n\n"
        "pub fn check() -> u32 {\n    relocated();\n    draws_between(0, LIMIT)\n}\n\n"
        "pub fn paths(name: &str) -> bool {\n    let _ = std::fs::read(name);\n    let _ = fs::read(name);\n    let _ = Vec::<u8>::new();\n"
        "    let _ = util::helper();\n    let _ = ExpectError::exceeded(1);\n    let _ = crate::expect::ExpectError::Over(2);\n"
        "    crate::fixture::is_fixture_file(name) && super::fixture::is_fixture_file(name)\n}\n"
    ),
    "a/src/values.rs": (
        "use crate::expect::{ExpectError, LIMIT};\nuse std::cmp::Ordering;\n\n"
        "pub fn errors(e: &ExpectError) -> Result<u32, ExpectError> {\n    match e {\n"
        "        ExpectError::BattleNotStated => Err(ExpectError::BattleNotStated.into()),\n"
        "        ExpectError::Over(n) => Ok(*n + crate::expect::LIMIT + ExpectError::CAP + ExpectError::FLOOR),\n"
        "    }\n}\n\n"
        "pub fn elsewhere(e: &ExpectError) -> u32 {\n    assert!(matches!(e, ExpectError::BattleNotStated));\n"
        "    let _ = (u32::MAX, std::u32::MAX, Ordering::Less, UNDEFINED_NAME);\n    LIMIT\n}\n\n"
        # A function's own `const` shadows the module's: neither its name nor its use is LIMIT's.
        "pub fn shadowed() -> u32 {\n    const LIMIT: u32 = 2;\n    LIMIT\n}\n"
    ),
    # A same-named Python module beside the crate is never a Rust path's target.
    "util.py": "def helper():\n    pass\n",
    "a/src/session/mod.rs": (
        "pub fn step() {}\n\npub mod inner {\n    pub fn deep() {}\n}\n\n#[cfg(test)]\nmod tests;\n"
    ),
    # `use super::*;` brings the parent's names into scope: module evidence.
    "a/src/session/tests.rs": "use super::*;\n\nfn steps() {\n    step();\n    inner::deep();\n}\n",
    "b/src/lib.rs": (
        "use a::is_fixture_file;\nuse a::rng::draws_between;\n\n"
        "pub fn run() -> bool {\n    is_fixture_file(\"x.toml\") && draws_between(1, 2) == 1\n}\n\n"
        "pub fn step() {}\n\npub fn by_path() -> bool {\n    a::fixture::is_fixture_file(\"y.toml\")\n}\n"
    ),
    "b/tests/smoke.rs": "use b::run;\n\n#[test]\nfn runs() {\n    run();\n}\n",
}


def _write(root: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body, encoding="utf-8")


def _extract(root: Path, files: dict[str, str] = _WORKSPACE) -> dict:
    _write(root, files)
    return extract(sorted(root / name for name in files), root=root, cache_root=root / "graphify-out",
                   parallel=False)


def _nid(result: dict, label: str, source_file: str) -> str:
    return next(n["id"] for n in result["nodes"]
                if n["label"] == label and n.get("source_file", "").replace("\\", "/") == source_file)


def _edges_from(result: dict, source_file: str, relation: str) -> list[tuple[str, str]]:
    return sorted((e["target"], e.get("confidence", "")) for e in result["edges"]
                  if e["relation"] == relation and e.get("source_file", "").replace("\\", "/") == source_file)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> dict:
    return _extract(tmp_path_factory.mktemp("workspace"))


def test_a_use_lands_on_the_item_its_path_names(workspace):
    draws = _nid(workspace, "draws_between()", "a/src/rng.rs")
    imports = {target for target, _ in _edges_from(workspace, "a/src/other.rs", "imports_from")}
    assert draws in imports
    assert _nid(workspace, "relocated()", "a/src/renamed.rs") in imports  # through `#[path]`
    assert {_nid(workspace, "ExpectError", "a/src/expect.rs"), _nid(workspace, "LIMIT", "a/src/expect.rs")} <= imports
    # Another crate's paths name nothing here, local `fs.rs` or not.
    assert {"read", "serialize"} <= imports
    assert _nid(workspace, "read()", "a/src/fs.rs") not in imports


def test_a_use_through_a_workspace_crate_and_a_re_export(workspace):
    imports = {target for target, _ in _edges_from(workspace, "b/src/lib.rs", "imports_from")}
    assert imports == {_nid(workspace, "draws_between()", "a/src/rng.rs"),
                       _nid(workspace, "is_fixture_file()", "a/src/fixture.rs")}
    # An integration test names its own crate's library.
    assert _edges_from(workspace, "b/tests/smoke.rs", "imports_from") == [
        (_nid(workspace, "run()", "b/src/lib.rs"), "EXTRACTED")]


def test_a_bare_call_after_a_use_binds_to_what_it_imports(workspace):
    draws = _nid(workspace, "draws_between()", "a/src/rng.rs")
    assert (draws, "EXTRACTED") in _edges_from(workspace, "a/src/other.rs", "calls")
    calls = _edges_from(workspace, "b/src/lib.rs", "calls")
    assert (draws, "EXTRACTED") in calls
    assert (_nid(workspace, "is_fixture_file()", "a/src/fixture.rs"), "EXTRACTED") in calls
    # Never the same-named helper in another file's test module.
    helper = _nid(workspace, "draws_between()", "a/src/order.rs")
    assert helper not in {e["target"] for e in workspace["edges"]
                          if not e.get("source_file", "").replace("\\", "/").startswith("a/src/order.rs")}


def test_a_glob_import_is_evidence_for_the_names_it_brings_in(workspace):
    # Two `step()`s; `use super::*;` names session/mod.rs's, and brings in its `inner`.
    assert _edges_from(workspace, "a/src/session/tests.rs", "calls") == sorted([
        (_nid(workspace, "step()", "a/src/session/mod.rs"), "EXTRACTED"),
        (_nid(workspace, "deep()", "a/src/session/mod.rs"), "EXTRACTED")])


def test_a_plain_module_import_is_no_evidence_for_a_bare_call(tmp_path: Path):
    # `use crate::util;` does not bring `helper` into scope: a bare `helper()` cannot
    # be util's, so the module import must not promote it.
    result = _extract(tmp_path, {
        "Cargo.toml": '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n',
        "src/lib.rs": "pub mod other;\npub mod user;\npub mod util;\n",
        "src/util.rs": "pub fn helper() {}\n", "src/other.rs": "pub fn helper() {}\n",
        "src/user.rs": "use crate::util;\n\npub fn run() {\n    helper();\n}\n"})
    assert _edges_from(result, "src/user.rs", "imports_from") == [(_nid(result, "util.rs", "src/util.rs"), "EXTRACTED")]
    assert all(confidence != "EXTRACTED" for _, confidence in _edges_from(result, "src/user.rs", "calls"))


def test_a_named_import_shadows_a_glob_even_when_it_names_nothing(tmp_path: Path):
    files = {
        "Cargo.toml": '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n',
        "src/lib.rs": "pub mod util;\npub struct Cursor;\npub struct Other;\n",
        "src/util.rs": "use super::*;\nuse std::io::Cursor;\n",
    }
    result = _extract(tmp_path, files)
    paths = RustPathIndex(result["nodes"], tmp_path)
    assert paths.resolve("src/util.rs", "", "Cursor", tmp_path) is None  # std's, not lib.rs's
    assert paths.resolve("src/util.rs", "", "Other", tmp_path) == ("item", _nid(result, "Other", "src/lib.rs"))


def test_a_glob_of_its_own_file_is_dropped(tmp_path: Path):
    # `use super::*;` in a file's inline test module imports that file: no edge, and
    # no `super` stub left holding it.
    result = _extract(tmp_path, {
        "Cargo.toml": '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n',
        "src/lib.rs": "pub mod util;\n",
        "src/util.rs": "pub fn helper() {}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n"
                       "    fn uses() {\n        helper();\n    }\n}\n"})
    assert _edges_from(result, "src/util.rs", "imports_from") == []
    assert "super" not in {n["id"] for n in result["nodes"]}
    assert (_nid(result, "helper()", "src/util.rs"), "EXTRACTED") in _edges_from(result, "src/util.rs", "calls")


# --- path calls ---

def _calls_from(result: dict, caller: str, source_file: str) -> list[tuple[str, str, str]]:
    caller_nid = _nid(result, caller, source_file)
    return sorted((e["target"], e["relation"], e.get("confidence", "")) for e in result["edges"]
                  if e["source"] == caller_nid and e.get("context") in ("call", "constructor"))


def test_a_path_call_binds_to_the_item_its_path_names(workspace):
    fixture = _nid(workspace, "is_fixture_file()", "a/src/fixture.rs")
    assert (fixture, "calls", "EXTRACTED") in _calls_from(workspace, "paths()", "a/src/other.rs")
    assert (fixture, "calls", "EXTRACTED") in _calls_from(workspace, "by_path()", "b/src/lib.rs")


def test_a_path_call_never_binds_to_a_same_named_item_in_its_own_file(workspace):
    # `crate::rng::draws_between()` in order.rs used to bind to order.rs's own
    # test helper, EXTRACTED.
    assert _calls_from(workspace, "first()", "a/src/order.rs") == [
        (_nid(workspace, "draws_between()", "a/src/rng.rs"), "calls", "EXTRACTED")]


def test_a_type_qualified_call_binds_to_the_types_method_or_variant(workspace):
    calls = _calls_from(workspace, "paths()", "a/src/other.rs")
    assert (_nid(workspace, ".exceeded()", "a/src/expect.rs"), "calls", "EXTRACTED") in calls
    assert (_nid(workspace, "Over", "a/src/expect.rs"), "references", "EXTRACTED") in calls


def test_a_turbofish_call_binds_like_the_plain_one(tmp_path: Path):
    result = _extract(tmp_path, {
        "Cargo.toml": '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n',
        "src/lib.rs": "pub mod parse;\npub mod user;\n",
        "src/parse.rs": "pub fn parse_num<T: Default>() -> T {\n    T::default()\n}\n",
        "src/user.rs": (
            "use crate::parse::parse_num;\n\npub fn by_use() -> u32 {\n    parse_num::<u32>()\n}\n\n"
            "pub fn by_path() -> u8 {\n    crate::parse::parse_num::<u8>()\n}\n\n"
            "fn local<T>() {}\n\npub fn in_file() {\n    local::<u8>();\n}\n"
        )})
    read = _nid(result, "parse_num()", "src/parse.rs")
    assert _calls_from(result, "by_use()", "src/user.rs") == [(read, "calls", "EXTRACTED")]
    assert _calls_from(result, "by_path()", "src/user.rs") == [(read, "calls", "EXTRACTED")]
    assert _calls_from(result, "in_file()", "src/user.rs") == [(_nid(result, "local()", "src/user.rs"), "calls", "EXTRACTED")]


def test_a_path_into_another_crate_or_language_binds_nothing(workspace):
    targets = {target for target, _, _ in _calls_from(workspace, "paths()", "a/src/other.rs")}
    # `std::fs::read()`, and `fs::read()` after `use std::fs;`, beside the crate's own fs.rs.
    assert _nid(workspace, "read()", "a/src/fs.rs") not in targets
    assert _nid(workspace, "helper()", "util.py") not in targets
    assert len(targets) == 3  # is_fixture_file, .exceeded() and Over


# --- variants and constants as values ---

def _references_from(result: dict, caller: str, source_file: str) -> list[tuple[str, str, str]]:
    caller_nid = _nid(result, caller, source_file)
    return sorted((e["target"], e.get("context", ""), e.get("confidence", "")) for e in result["edges"]
                  if e["source"] == caller_nid and e["relation"] == "references"
                  and e.get("context") in ("variant", "const", "match", "constructor"))


def test_a_variant_or_constant_named_as_a_value_is_a_reference(workspace):
    variant = _nid(workspace, "BattleNotStated", "a/src/expect.rs")
    refs = _references_from(workspace, "errors()", "a/src/values.rs")
    assert (variant, "match", "EXTRACTED") in refs  # `ExpectError::BattleNotStated =>`
    assert (variant, "variant", "EXTRACTED") not in refs  # one edge per pair, the first context
    assert (_nid(workspace, "Over", "a/src/expect.rs"), "match", "EXTRACTED") in refs
    assert (_nid(workspace, "LIMIT", "a/src/expect.rs"), "const", "EXTRACTED") in refs  # a path to a module's const
    assert (_nid(workspace, ".CAP", "a/src/expect.rs"), "const", "EXTRACTED") in refs  # an associated const
    assert (_nid(workspace, ".FLOOR", "a/src/floor.rs"), "const", "EXTRACTED") in refs  # in another file's impl


def test_a_self_variant_in_its_own_impl_is_a_reference(workspace):
    refs = _references_from(workspace, ".kind()", "a/src/expect.rs")
    assert (_nid(workspace, "BattleNotStated", "a/src/expect.rs"), "match", "EXTRACTED") in refs
    assert (_nid(workspace, "Over", "a/src/expect.rs"), "match", "EXTRACTED") in refs


def test_a_bare_constant_resolves_through_its_use(workspace):
    limit = _nid(workspace, "LIMIT", "a/src/expect.rs")
    assert _references_from(workspace, "elsewhere()", "a/src/values.rs") == [(limit, "const", "EXTRACTED")]
    assert (limit, "const", "EXTRACTED") in _references_from(workspace, "check()", "a/src/other.rs")
    assert _references_from(workspace, "shadowed()", "a/src/values.rs") == []
    # Not `matches!(…)`'s variant (a macro's arguments are not parsed), nor std's or
    # an undefined name's constants.

def test_a_struct_style_variant_is_a_reference(tmp_path: Path):
    # `E::V { .. }` spells its path as a type, not a value: as a literal it constructs
    # the variant, as a pattern it matches it. A struct by its path is the same.
    result = _extract(tmp_path, {
        "Cargo.toml": '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n',
        "src/lib.rs": "pub mod err;\npub mod user;\n",
        "src/err.rs": "pub enum RouteError {\n    NoPath { from: u32 },\n}\n\npub struct Cfg {\n    pub a: u32,\n}\n",
        "src/user.rs": (
            "use crate::err::RouteError;\n\npub fn fail() -> RouteError {\n    RouteError::NoPath { from: 1 }\n}\n\n"
            "pub fn from(e: RouteError) -> u32 {\n    match e {\n        RouteError::NoPath { from } => from,\n    }\n}\n\n"
            "pub fn cfg() -> u32 {\n    crate::err::Cfg { a: 1 }.a\n}\n"
        )})
    no_path = _nid(result, "NoPath", "src/err.rs")
    assert (no_path, "constructor", "EXTRACTED") in _references_from(result, "fail()", "src/user.rs")
    assert (no_path, "match", "EXTRACTED") in _references_from(result, "from()", "src/user.rs")
    assert _references_from(result, "cfg()", "src/user.rs") == [(_nid(result, "Cfg", "src/err.rs"), "constructor", "EXTRACTED")]


# --- the index on its own: what resolves to nothing ---

_INDEX = {
    "Cargo.toml": '[package]\nname = "demo-crate"\nversion = "0.1.0"\nedition = "2021"\n',
    "src/lib.rs": (
        "pub mod a;\npub mod both;\npub fn both() {}\nmod inner {\n    pub mod deep;\n    pub fn here() {}\n}\n"
        "pub use a::Thing;\npub use a::*;\n"
    ),
    "src/a.rs": "pub struct Thing;\npub enum Kind {\n    One,\n}\npub fn globbed() {}\n",
    "src/both.rs": "",
    "src/inner/deep.rs": "pub fn down() {}\nuse super::here;\n",
    "src/hops.rs": "",
}


@pytest.fixture(scope="module")
def index(tmp_path_factory) -> tuple[RustPathIndex, dict, Path]:
    root = tmp_path_factory.mktemp("index")
    result = _extract(root, _INDEX)
    return RustPathIndex(result["nodes"], root), result, root


@pytest.mark.parametrize("path, expected", [
    ("crate::a::Thing", ("Thing", "src/a.rs")),
    ("crate::Thing", ("Thing", "src/a.rs")),  # a `pub use` re-export
    ("demo_crate::a::Kind::One", ("One", "src/a.rs")),  # the crate's name, `-` read as `_`
    ("::demo_crate::a::Kind", ("Kind", "src/a.rs")),
    ("crate::inner::deep::down", ("down()", "src/inner/deep.rs")),  # `mod` inside an inline `mod`
    ("crate::inner::here", ("here()", "src/lib.rs")),
    ("crate::inner::deep::here", ("here()", "src/lib.rs")),  # a private `use` seen from below
    ("crate::globbed", ("globbed()", "src/a.rs")),  # through `pub use a::*;`
])
def test_a_path_resolves_to_exactly_one_item(index, path, expected):
    paths, result, root = index
    target = paths.resolve("src/lib.rs", "", path, root)
    assert target == ("item", _nid(result, *expected))


@pytest.mark.parametrize("path", [
    "std::fs::read", "serde::Serialize", "a::Thing::new",
    "crate::both",  # a module and a fn: two things
    "crate::hops",  # a file no `mod` declares
    "crate::missing", "super::a", "crate::a::Kind::Two",
])
def test_a_path_that_names_no_single_item_resolves_to_nothing(index, path):
    paths, _result, root = index
    target = paths.resolve("src/lib.rs", "", path, root)
    assert target is None or path == "a::Thing::new" and target[0] != "item"


def test_super_and_self_resolve_from_the_module_the_use_is_in(index):
    paths, result, root = index
    here = ("item", _nid(result, "here()", "src/lib.rs"))
    assert paths.resolve("src/inner/deep.rs", "", "super::here", root) == here
    assert paths.resolve("src/lib.rs", "inner", "self::here", root) == here
    assert paths.resolve("src/lib.rs", "inner", "here", root) == here


def test_a_file_reached_from_no_root_resolves_nothing(index):
    paths, _result, root = index
    assert paths.resolve("src/hops.rs", "", "crate::a::Thing", root) is None


# --- incremental builds ---

def _rebuild(root: Path, changed: list[Path] | None = None) -> dict:
    from graphify.watch import _rebuild_code

    assert _rebuild_code(root, changed_paths=changed, no_cluster=True, acquire_lock=False) is True
    return json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))


def _view(graph: dict) -> tuple:
    return (sorted(json.dumps(e, sort_keys=True) for e in graph.get("links", graph.get("edges", []))),
            sorted(json.dumps(n, sort_keys=True) for n in graph["nodes"] if not n.get("external")))


@pytest.mark.parametrize("changed, text", [
    ("a/src/other.rs", _WORKSPACE["a/src/other.rs"] + "\npub fn again() -> u32 {\n    draws_between(1, 2)\n}\n"),
    ("b/src/lib.rs", _WORKSPACE["b/src/lib.rs"].replace("use a::is_fixture_file;\n", "")),
    ("a/src/session/tests.rs", "use super::step;\n\nfn steps() {\n    step();\n}\n"),
    ("a/src/order.rs", "pub fn first() -> bool {\n    crate::fixture::is_fixture_file(\"z\")\n}\n"),
    ("a/src/expect.rs", _WORKSPACE["a/src/expect.rs"].replace("exceeded", "above")),
    ("a/src/floor.rs", _WORKSPACE["a/src/floor.rs"].replace("FLOOR", "BASE")),
    ("a/src/values.rs", _WORKSPACE["a/src/values.rs"] + "\npub fn more() -> u32 {\n    ExpectError::CAP\n}\n"),
])
def test_an_incremental_build_resolves_what_a_full_one_does(tmp_path: Path, changed: str, text: str):
    root, fresh = tmp_path / "incremental", tmp_path / "fresh"
    _write(root, _WORKSPACE)
    _rebuild(root)
    (root / changed).write_text(text, encoding="utf-8")
    incremental = _rebuild(root, [root / changed])
    _write(fresh, {**_WORKSPACE, changed: text})
    assert _view(incremental) == _view(_rebuild(fresh))
