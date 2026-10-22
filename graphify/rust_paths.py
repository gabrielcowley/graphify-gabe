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
module. `std::`, `serde::`, any crate outside the corpus, a glob, a `[lib]` or
dependency rename, and a name reached only through a glob re-export resolve to
nothing; `pub use` chains are followed at most `_MAX_HOPS` deep.
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
            info = self._files[file]
            for scope, bound, use_path, _is_pub in info.get("_rust_imports") or ():
                if bound == name and _split(scope) == inline and not use_path.endswith("*"):
                    found |= self._walk((root_file, module), _split(use_path), hops - 1)
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
