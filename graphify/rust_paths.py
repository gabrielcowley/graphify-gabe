"""Which item or module a written Rust path names, from the extracted nodes alone.

extractors/rust.py leaves the facts on the nodes it emits, so the index can be
rebuilt from graph.json on an incremental build (the markers are forwarded with
the other resolution-context markers in watch.py and cli.py):

- a file node carries `_rust_mods` (`[scope, name, #[path]]` for each `mod x;`),
  `_rust_inline_mods` (each inline `mod x { … }`, as a path inside the file) and
  `_rust_imports` (`[scope, bound name, path, is pub]` for each `use` binding);
- an item node (fn, struct, enum, trait, const, static, enum variant) carries
  `_rust_paths`, the paths it is declared at inside its file (`tests::helper`,
  `Error::Missing`).

Crate names come from `Cargo.toml` package nodes (manifest_ingest), with `-` read
as `_`. A crate's roots follow Cargo's default layout: `src/lib.rs`,
`src/main.rs`, `src/bin/*.rs`, and `tests/`, `benches/`, `examples/` and
`build.rs` beside a crate's `src/`. The module tree is walked from each root
through `mod x;` declarations to `x.rs` or `x/mod.rs` (a `#[path]` wins), so a
file reached from no root has no module and resolves nothing.

Resolution fails closed: a path resolves only when it names exactly one item or
module. `std::`, `serde::`, any crate outside the corpus, and a `[lib]` or
dependency rename resolve to nothing. `use` and `pub use` chains, globs
included (a name a module declares or imports by name shadows a glob's), are
followed at most `_MAX_HOPS` deep.
"""
from __future__ import annotations

import functools
import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

_MAX_HOPS = 4
_MAX_MODULE_DEPTH = 64
_ROOT_DIRS = ("tests", "benches", "examples")

ModKey = tuple[str, tuple[str, ...]]  # (crate root file, module path from that root)
Target = tuple[str, object]  # ("module", ModKey) | ("item", node id)


@functools.lru_cache(maxsize=65536)
def _norm(source_file: str, root: Path) -> str:
    """``source_file`` as an absolute, normalized posix path (many nodes share one)."""
    path = source_file if os.path.isabs(source_file) else os.path.join(root, source_file)
    return os.path.normpath(path).replace("\\", "/")


def _split(scope: str) -> tuple[str, ...]:
    return tuple(scope.split("::")) if scope else ()


class RustPathIndex:
    """The crates, module tree and items of every Rust file in ``nodes``."""

    def __init__(self, nodes: Iterable[dict], root: Path) -> None:
        self._files: dict[str, dict] = {}
        self._items: dict[tuple[str, str], set[str]] = {}
        self._item_paths: dict[str, list[tuple[str, str]]] = {}
        manifests: dict[str, set[str]] = {}
        for node in nodes:
            source_file = str(node.get("source_file") or "")
            if not source_file:
                continue
            name = PurePosixPath(source_file.replace("\\", "/")).name
            if node.get("type") == "package" and name.lower() == "cargo.toml" and node.get("label"):
                crate_dir = str(PurePosixPath(_norm(source_file, root)).parent)
                manifests.setdefault(str(node["label"]).replace("-", "_"), set()).add(crate_dir)
                continue
            if not source_file.endswith(".rs"):
                continue
            file = _norm(source_file, root)
            # A file node is labelled with its basename, or (two files sharing one, in
            # a stored graph) with as much of its path as tells them apart.
            label = str(node.get("label") or "").replace("\\", "/")
            if label.endswith(name) and source_file.replace("\\", "/").endswith(label):
                self._files.setdefault(file, node)
            for item_path in node.get("_rust_paths") or ():
                self._items.setdefault((file, item_path), set()).add(node["id"])
                self._item_paths.setdefault(node["id"], []).append((file, item_path))

        crate_dirs = {d for dirs in manifests.values() for d in dirs}
        for file in self._files:
            parts = PurePosixPath(file)
            if parts.parent.name == "src" and parts.name in ("lib.rs", "main.rs"):
                crate_dirs.add(str(parts.parent.parent))
        # A name two crates share names neither.
        self._crate_libs: dict[str, str] = {}
        for crate_name, dirs in manifests.items():
            libs = {f"{d}/src/lib.rs" for d in dirs} & self._files.keys()
            if len(dirs) == 1 and len(libs) == 1:
                self._crate_libs[crate_name] = libs.pop()

        # Each module's `use` bindings, by the name they bind, and its globs.
        self._imports: dict[tuple[str, tuple[str, ...]], tuple[dict[str, list[tuple[str, ...]]], list[tuple[str, ...]]]] = {}
        for file, info in self._files.items():
            for scope, bound, use_path, _is_pub in info.get("_rust_imports") or ():
                by_name, globs = self._imports.setdefault((file, _split(scope)), ({}, []))
                if use_path.endswith("*"):
                    if _split(use_path)[:-1]:
                        globs.append(_split(use_path)[:-1])
                else:
                    by_name.setdefault(bound, []).append(_split(use_path))

        self._modules: dict[ModKey, tuple[str, tuple[str, ...]]] = {}
        self._file_modules: dict[str, set[ModKey]] = {}
        for file in sorted(self._files):
            if self._is_root(file, crate_dirs):
                self._visit(file, (), file, mod_rs=True, depth=0)

    @staticmethod
    def _is_root(file: str, crate_dirs: set[str]) -> bool:
        path = PurePosixPath(file)
        parent = path.parent
        if parent.name == "src" and path.name in ("lib.rs", "main.rs"):
            return True
        if parent.name == "bin" and parent.parent.name == "src":
            return True
        if path.name == "main.rs" and parent.parent.name == "bin" and parent.parent.parent.name == "src":
            return True
        if parent.name in _ROOT_DIRS and str(parent.parent) in crate_dirs:
            return True
        if path.name == "main.rs" and parent.parent.name in _ROOT_DIRS and str(parent.parent.parent) in crate_dirs:
            return True
        return path.name == "build.rs" and str(parent) in crate_dirs

    def _visit(self, root_file: str, module: tuple[str, ...], file: str, *, mod_rs: bool, depth: int) -> None:
        key = (root_file, module)
        if key in self._modules or depth > _MAX_MODULE_DEPTH:
            return
        self._modules[key] = (file, ())
        self._file_modules.setdefault(file, set()).add(key)
        info = self._files[file]
        for inline in info.get("_rust_inline_mods") or ():
            self._modules.setdefault((root_file, module + _split(inline)), (file, _split(inline)))
        path = PurePosixPath(file)
        # A root, a `mod.rs` and a `#[path]` file own their directory; any other
        # `x.rs` owns `x/`.
        mod_dir = path.parent if mod_rs or path.name == "mod.rs" else path.parent / path.stem
        for scope, name, attribute in info.get("_rust_mods") or ():
            inline = _split(scope)
            base = mod_dir.joinpath(*inline)
            if attribute:
                found = [f for f in [_norm(str((path.parent if not inline else base) / attribute), Path("/"))]
                         if f in self._files]
            else:
                found = [f for f in (f"{base}/{name}.rs", f"{base}/{name}/mod.rs") if f in self._files]
            if len(found) == 1:
                self._visit(root_file, module + inline + (name,), found[0],
                            mod_rs=bool(attribute), depth=depth + 1)

    # --- resolution ---------------------------------------------------------

    def resolve(self, source_file: str, scope: str, path: str, root: Path) -> Target | None:
        """What ``path``, written in ``scope`` of ``source_file``, names: exactly one
        target, or None."""
        segments = _split(path)
        file = _norm(source_file, root)
        found: set[Target] = set()
        for root_file, module in self._file_modules.get(file, ()):
            start = (root_file, module + _split(scope))
            if start in self._modules:
                found |= self._walk(start, segments, _MAX_HOPS)
        return next(iter(found)) if len(found) == 1 else None

    def file_node(self, target: Target) -> str | None:
        """The file node of a module target that is a whole file, else None."""
        if target[0] != "module":
            return None
        file, inline = self._modules[target[1]]  # type: ignore[index]
        return self._files[file]["id"] if not inline else None

    def _walk(self, start: ModKey, segments: tuple[str, ...], hops: int) -> set[Target]:
        if not segments:
            return set()
        first, rest = segments[0], segments[1:]
        current: set[Target]
        if first == "crate":
            current = {("module", (start[0], ()))}
        elif first in ("self", "super"):
            current = self._member(("module", start), first, hops)
        elif first == "":
            # `::name` always names a crate.
            if not rest or rest[0] not in self._crate_libs:
                return set()
            current = {("module", (self._crate_libs[rest[0]], ()))}
            rest = rest[1:]
        else:
            # A path's first segment is a name in scope or, failing that, a crate.
            current = self._member(("module", start), first, hops)
            if not current and first in self._crate_libs:
                current = {("module", (self._crate_libs[first], ()))}
        for segment in rest:
            current = {found for target in current for found in self._member(target, segment, hops)}
            if not current:
                break
        return current

    def _member(self, target: Target, name: str, hops: int) -> set[Target]:
        kind, value = target
        if kind == "item":
            # An enum's variant.
            return {("item", nid) for file, item_path in self._item_paths.get(value, ())  # type: ignore[arg-type]
                    for nid in self._items.get((file, f"{item_path}::{name}"), ())}
        root_file, module = value  # type: ignore[misc]
        if name == "self":
            return {target}
        if name == "super":
            return {("module", (root_file, module[:-1]))} if module else set()
        found: set[Target] = set()
        child = (root_file, module + (name,))
        if child in self._modules:
            found.add(("module", child))
        file, inline = self._modules[(root_file, module)]
        found |= {("item", nid) for nid in self._items.get((file, "::".join(inline + (name,))), ())}
        if hops > 0:
            # A `use` binds the name in its module; `pub use` re-exports it.
            by_name, globs = self._imports.get((file, inline), ({}, []))
            for segments in by_name.get(name, ()):
                found |= self._walk((root_file, module), segments, hops - 1)
            # A glob (`use super::*;`) brings in every name of what it imports, but
            # anything the module declares or imports by name shadows it.
            if not found:
                for segments in globs:
                    for source in self._walk((root_file, module), segments, hops - 1):
                        found |= self._member(source, name, hops - 1)
        return found


def retarget_use_edges(nodes: Iterable[dict], edges: list[dict], root: Path) -> None:
    """Point each Rust `use` edge at what its path names, when that is exactly one thing.

    An item takes the edge, and marks it `_rust_use_evidence = "symbol"`: the
    shared call pass counts it as an import of that symbol. A module that is a
    whole file gives the edge that file's node; only a glob of it
    (`use crate::util::*;`) brings names into scope, so only a glob is marked
    `"module"` evidence, and any other module import `"none"`. A resolved edge
    drops `_rust_use_name`, so #4439's by-name rewire only sees the rest. A `use`
    of the file it is written in (`use super::*;` in the file's inline
    `mod tests`) says nothing, and is dropped rather than left on a `super` stub.
    """
    pending = [e for e in edges if e.get("_rust_use_path")]
    if not pending:
        return
    index = RustPathIndex(nodes, root)
    own_file: set[int] = set()
    for edge in pending:
        path = str(edge["_rust_use_path"])
        glob = path.endswith("*")
        target = index.resolve(str(edge.get("source_file", "")), str(edge.get("_rust_use_scope", "")),
                               path[:-1].rstrip(":") if glob else path, root)
        if target is None:
            continue
        if target[0] == "item" and not glob:
            new_target, evidence = str(target[1]), "symbol"
        else:
            file_node = index.file_node(target)
            if file_node is None:
                continue
            new_target, evidence = file_node, "module" if glob else "none"
        if new_target == edge.get("source"):
            own_file.add(id(edge))
            continue
        edge["target"] = new_target
        edge["_rust_use_evidence"] = evidence
        edge.pop("_rust_use_name", None)
    if own_file:
        edges[:] = [e for e in edges if id(e) not in own_file]


def resolve_path_calls(
    raw_calls: Iterable[dict], nodes: list[dict], edges: list[dict], root: Path,
) -> list[dict]:
    """`calls` edges for the Rust path calls (`rust_path` raw calls) the path names.

    The whole path is tried first: `crate::fixture::is_fixture_file`,
    `super::f`, `a::rng::draws_between`, or a variant constructor
    (`ExpectError::Missing(x)`, a `references` edge). Failing that, all but the
    last segment must name a type and the last one of its methods: the type's
    impl blocks are pooled across files the way `_resolve_rust_self_member_calls`
    pools them, and only when one declaration has that name; with several, only
    the impl blocks sharing the declaration's node count. Exactly one target or
    no edge, and every edge is EXTRACTED: the path says which item it is.
    """
    pending = [rc for rc in raw_calls if rc.get("rust_path") and rc.get("caller_nid")]
    if not pending:
        return []
    index = RustPathIndex(nodes, root)
    node_by_id = {n.get("id"): n for n in nodes}
    methods: dict[tuple[str, str], set[str]] = {}
    for e in edges:
        if e.get("relation") == "method" and (target := node_by_id.get(e.get("target"))) is not None:
            methods.setdefault((str(e.get("source")), str(target.get("label", "")).strip("()").lstrip(".")),
                               set()).add(e["target"])
    owners_by_label: dict[str, list[str]] = {}
    declarations: dict[str, int] = {}
    for n in nodes:
        if not str(n.get("source_file") or "").endswith(".rs"):
            continue
        label = str(n.get("label", ""))
        impl_key = n.get("_rust_impl_key")
        owner = impl_key.rsplit("/", 1)[0] if isinstance(impl_key, str) and "/" in impl_key else label
        owners_by_label.setdefault(owner, []).append(n["id"])
        if isinstance(count := n.get("_rust_declaration_count"), int) and not isinstance(count, bool):
            declarations[label] = declarations.get(label, 0) + count

    def method_of(type_nid: str, name: str) -> str | None:
        label = str(node_by_id.get(type_nid, {}).get("label", ""))
        owners = owners_by_label.get(label, []) if declarations.get(label) == 1 else [type_nid]
        found = {m for owner in owners for m in methods.get((owner, name), ())}
        return next(iter(found)) if len(found) == 1 else None

    existing = {(e.get("source"), e.get("target")) for e in edges if e.get("relation") in ("calls", "references")}
    new_edges: list[dict] = []
    for rc in pending:
        source_file, scope, path = str(rc.get("source_file", "")), str(rc.get("rust_scope", "")), str(rc["rust_path"])
        target = index.resolve(source_file, scope, path, root)
        tgt = str(target[1]) if target is not None and target[0] == "item" else None
        if tgt is None and "::" in path:
            owner = index.resolve(source_file, scope, path.rsplit("::", 1)[0], root)
            if owner is not None and owner[0] == "item":
                tgt = method_of(str(owner[1]), path.rsplit("::", 1)[1])
        caller = rc["caller_nid"]
        if tgt is None or tgt == caller or (caller, tgt) in existing:
            continue
        existing.add((caller, tgt))
        constructor = not str(node_by_id.get(tgt, {}).get("label", "")).endswith(")")
        new_edges.append({
            "source": caller,
            "target": tgt,
            "relation": "references" if constructor else "calls",
            "context": "constructor" if constructor else "call",
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "source_file": source_file,
            "source_location": rc.get("source_location"),
            "weight": 1.0,
        })
    return new_edges
