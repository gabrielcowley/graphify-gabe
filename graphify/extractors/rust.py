"""Rust extractor. Moved verbatim from graphify/extract.py."""
from __future__ import annotations


import hashlib
import re
from pathlib import Path
from graphify.extractors.base import (
    _LANGUAGE_BUILTIN_GLOBALS,
    _file_stem,
    _make_id,
    _read_source_bytes,
    _read_text,
)


# Rust prelude / std types. Language-local (like _GO_PREDECLARED_FUNCS and
# _RUST_TRAIT_METHOD_BLOCKLIST below) rather than added to the shared
# _LANGUAGE_BUILTIN_GLOBALS: the shared set is consulted with no language gate,
# and `Path`, `Result`, `Error`, `From`, `Into`, `Iterator`, `Default` and
# friends are ordinary user type names in the other languages that read it.
#
# Without this set a Rust codebase resolves every `Option`/`Vec`/`String`/
# `Result` annotation to a few canonical nodes, so the language's own primitives
# become the top god nodes (a ~1,200-file workspace had `String` at #1 with
# ~1,700 edges). Filtering is definition-aware: see _rust_collect_type_refs.
_RUST_BUILTIN_TYPES: frozenset[str] = frozenset({
    # prelude types and aliases
    "String", "str", "Option", "Result", "Vec", "VecDeque", "Box", "Rc",
    "Arc", "Weak", "RefCell", "Cell", "Cow", "Pin",
    # collections
    "HashMap", "HashSet", "BTreeMap", "BTreeSet", "BinaryHeap",
    # paths / OS strings
    "Path", "PathBuf", "OsStr", "OsString", "CStr", "CString",
    # time / ranges / markers
    "Duration", "Instant", "SystemTime", "Ordering", "Range", "RangeInclusive",
    "RangeFrom", "RangeTo", "RangeFull", "PhantomData", "ManuallyDrop",
    "NonZeroU8", "NonZeroU16", "NonZeroU32", "NonZeroU64", "NonZeroUsize",
    # variants and core traits
    "Some", "None", "Ok", "Err", "Self",
    "Default", "Clone", "Copy", "Debug", "Display", "Error", "From", "Into",
    "TryFrom", "TryInto", "AsRef", "AsMut", "Iterator", "IntoIterator",
    "Extend", "PartialEq", "Eq", "PartialOrd", "Ord", "Hash", "Send", "Sync",
    "Sized", "Drop", "Deref", "DerefMut", "Future", "Fn", "FnMut", "FnOnce",
})


def _rust_collect_type_refs(
    node,
    source: bytes,
    generic: bool,
    out: list[tuple[str, str, str]],
    local_types: frozenset[str] = frozenset(),
) -> None:
    """Walk a Rust type expression; append (name, role, path) tuples: the path is
    the type as written (`crate::a::Flow`), the name its last segment.

    A name in ``_RUST_BUILTIN_TYPES`` is skipped unless ``local_types`` holds it:
    a file that defines its own ``struct Result<T>`` shadows the prelude, and its
    references must survive. Filtering the bare name alone would erase a
    legitimate user type that happens to share a std name.
    """

    def _keep(text: str) -> bool:
        return bool(text) and (text not in _RUST_BUILTIN_TYPES or text in local_types)

    if node is None:
        return
    t = node.type
    if t == "primitive_type":
        return
    if t == "type_identifier":
        text = _read_text(node, source)
        if _keep(text):
            out.append((text, "generic_arg" if generic else "type", text))
        return
    if t == "scoped_type_identifier":
        path = "".join(_read_text(node, source).split())
        text = path.rsplit("::", 1)[-1]
        if _keep(text):
            out.append((text, "generic_arg" if generic else "type", path))
        return
    if t == "generic_type":
        name_node = node.child_by_field_name("type")
        if name_node is None:
            for c in node.children:
                if c.type in ("type_identifier", "scoped_type_identifier"):
                    name_node = c
                    break
        if name_node is not None:
            path = "".join(_read_text(name_node, source).split())
            text = path.rsplit("::", 1)[-1]
            if _keep(text):
                out.append((text, "generic_arg" if generic else "type", path))
        for c in node.children:
            if c.type == "type_arguments":
                for arg in c.children:
                    if arg.is_named:
                        _rust_collect_type_refs(arg, source, True, out, local_types)
        return
    if t in ("reference_type", "pointer_type", "array_type", "tuple_type", "slice_type"):
        for c in node.children:
            if c.is_named:
                _rust_collect_type_refs(c, source, generic, out, local_types)
        return
    if node.is_named:
        for c in node.children:
            if c.is_named:
                _rust_collect_type_refs(c, source, generic, out, local_types)


def _rust_simple_generic_impl_key(node, source: bytes) -> str | None:
    """Return a stable owner/arity key for a deliberately narrow impl shape."""
    if node.child_by_field_name("trait") is not None:
        return None
    parameters = node.child_by_field_name("type_parameters")
    owner_type = node.child_by_field_name("type")
    if parameters is None or owner_type is None or owner_type.type != "generic_type":
        return None
    if any(child.type == "where_clause" for child in node.named_children):
        return None

    parameter_names: list[str] = []
    for parameter in parameters.named_children:
        if parameter.type != "type_parameter":
            return None
        named = parameter.named_children
        if len(named) != 1 or named[0].type != "type_identifier":
            return None
        name = _read_text(named[0], source)
        if not name or name in parameter_names:
            return None
        parameter_names.append(name)
    if not parameter_names:
        return None

    owner = owner_type.child_by_field_name("type")
    if owner is None or owner.type != "type_identifier":
        return None
    arguments = next(
        (child for child in owner_type.named_children if child.type == "type_arguments"),
        None,
    )
    if arguments is None:
        return None
    argument_names = [
        _read_text(argument, source)
        for argument in arguments.named_children
        if argument.type == "type_identifier"
    ]
    if len(argument_names) != len(arguments.named_children):
        return None
    if argument_names != parameter_names:
        return None

    owner_name = _read_text(owner, source)
    return f"{owner_name}/{len(parameter_names)}" if owner_name else None


def _rust_use_module(argument: str) -> str:
    """The name a plain `use` imports from: the last path segment of its argument, as
    written (`a::b` -> `b`, `a::b as c` -> `b as c`, `a::*` -> `a`)."""
    clean = argument.split("{")[0].rstrip(":").rstrip("*").rstrip(":")
    return clean.split("::")[-1].strip()


def _rust_path_segments(text: str) -> tuple[str, ...]:
    """A path as written, as its segments: comments and spacing dropped, `r#type` read
    as `type`, and a leading `::` (always a crate name) kept as an empty first segment.
    `crate :: a::/* x */b` gives `("crate", "a", "b")`; a glob ends in `"*"`."""
    text = re.sub(r"\s+", "", re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)).rstrip("{").rstrip(":")
    lead: tuple[str, ...] = ()
    if text.startswith("::"):
        lead, text = ("",), text[2:]
    return lead + tuple(s[2:] if s.startswith("r#") else s for s in text.split("::") if s) if text else lead


def _rust_use_bindings(node, source: bytes, prefix: str = "", self_path: str = "") -> list[tuple[str, int, tuple[str, ...], str]]:
    """Each binding of a `use` tree, as the argument of a plain `use` that binds the same
    name, with its line: `a::{b, c as d, e::{self}, f::*}` gives `b`, `c as d`, `a::e`
    and `f::*` (as `use a::b;` and so on would: only the last segment counts). Paths
    keep their text as written. Each also carries its whole path as segments
    (`_rust_path_segments`) and the name it binds: `d` for `c as d`, `*` for a glob."""
    t, line = node.type, node.start_point[0] + 1
    if t == "use_list":
        return [binding for child in node.named_children for binding in _rust_use_bindings(child, source, prefix, self_path)]
    if t == "scoped_use_list":
        path, group = node.child_by_field_name("path"), node.child_by_field_name("list")
        lead = prefix + source[node.start_byte:group.start_byte].decode("utf-8", errors="replace") if group is not None else ""
        owner = prefix + _read_text(path, source) if path is not None else ""
        return _rust_use_bindings(group, source, lead, owner) if group is not None else []
    alias, bound = "", None
    if t == "use_as_clause":
        clause, node = node, node.child_by_field_name("path")
        alias = source[node.end_byte:clause.end_byte].decode("utf-8", errors="replace") if node is not None else ""
        alias_node = clause.child_by_field_name("alias")
        bound = _read_text(alias_node, source) if alias_node is not None else None
    if node is not None and node.type == "self" and self_path:  # `a::{self}` binds `a`
        segments = _rust_path_segments(self_path)
        return [(self_path + alias, line, segments, bound or (segments[-1] if segments else ""))]
    if node is not None and node.type in ("use_wildcard", "identifier", "scoped_identifier", "crate", "self", "super"):
        segments = _rust_path_segments(prefix + _read_text(node, source))
        return [(prefix + _read_text(node, source) + alias, line, segments, bound or (segments[-1] if segments else ""))]
    return []


def _rust_path_attribute(mod_node, source: bytes) -> str:
    """The file a `#[path = "…"]` above a `mod x;` names, or "" when it has none. Other
    attributes and comments between them are skipped; `cfg_attr(…, path = …)` is not read."""
    sibling = mod_node.prev_named_sibling
    while sibling is not None and sibling.type in ("attribute_item", "line_comment", "block_comment"):
        if sibling.type == "attribute_item":
            match = re.fullmatch(r'#\[\s*path\s*=\s*"([^"\\]*)"\s*\]', _read_text(sibling, source).strip())
            if match:
                return match.group(1)
        sibling = sibling.prev_named_sibling
    return ""


_RUST_TRAIT_METHOD_BLOCKLIST: frozenset[str] = frozenset({
    "new", "default", "parse", "from_str", "now", "clone", "into", "from",
    "to_string", "to_owned", "len", "is_empty", "iter", "next", "build",
    "start", "run", "init", "app", "get", "set", "push", "pop", "insert",
    "remove", "contains", "collect", "map", "filter", "unwrap", "expect",
    "ok", "err", "some", "none", "send", "recv", "lock", "read", "write",
})

_RUST_PRIMITIVE_TYPES: frozenset[str] = frozenset({
    "bool", "char", "str", "f32", "f64",
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
})


def _rust_qualifier_type(path_node, source: bytes) -> str | None:
    """The type a `Type::f()` call is qualified by, or None for a module path.

    `Foo`, `crate::a::Foo`, `Foo::<T>`, `Vec::<u8>` and `i32` give their last
    segment without generic arguments; `Self` is returned as is for the caller to
    map to the enclosing impl type. Any other lowercase last segment is a module
    (`fs::read`, `self::helper`), and `<T as Trait>` has no segment of its own:
    both give None.
    """
    if path_node is None:
        return None
    last = _rust_type_last_segment(_read_text(path_node, source))
    return last if last[:1].isupper() or last in _RUST_PRIMITIVE_TYPES else None


def _rust_call_path(func_node, source: bytes) -> tuple[str, ...] | None:
    """A path callee's segments without generic arguments (`Vec::<u8>::new` gives
    `("Vec", "new")`), or None for a qualified-self path (`<T as Tr>::f`), which
    names no item by itself."""
    text = _read_text(func_node, source).strip()
    if text.startswith("<"):
        return None
    depth, kept = 0, []
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif depth == 0:
            kept.append(ch)
    segments = _rust_path_segments("".join(kept))
    return segments if len(segments) >= 2 else None


def _rust_path_may_be_local(segments: tuple[str, ...], local_types: set[str]) -> bool:
    """Whether a call path can name a corpus item at all. `Self::f()` is the impl's
    own method, and `Vec::new()` / `i32::from()` a std one unless the file defines
    that type itself."""
    first = segments[0]
    if first == "Self":
        return False
    builtin = first in _RUST_BUILTIN_TYPES or first in _RUST_PRIMITIVE_TYPES
    return not (len(segments) == 2 and builtin and first not in local_types)


# A constant's name, as Rust spells one: `LIMIT`, `MAX_LEN`, `E2`. One letter is
# more often a const generic or a type parameter than a constant.
_RUST_CONST_NAME = re.compile(r"[A-Z][A-Z0-9_]*[A-Z0-9][A-Z0-9_]*")

# Where an identifier or a path is not a reference of its own: part of a longer
# path, a macro's name or token tree, a generic call's function, a binding, or
# a declaration in a function body (a local enum's variant, a const generic).
_RUST_NOT_A_REFERENCE_PARENTS = frozenset({
    "scoped_identifier", "scoped_type_identifier", "macro_invocation", "token_tree",
    "generic_function", "let_declaration", "parameter", "closure_parameters",
    "shorthand_field_initializer", "enum_variant", "const_parameter",
})

_RUST_PATTERN_NODES = frozenset({
    "tuple_struct_pattern", "tuple_pattern", "or_pattern", "slice_pattern", "reference_pattern",
    "ref_pattern", "captured_pattern", "struct_pattern", "mut_pattern", "range_pattern",
})


def _rust_local_consts(body, source: bytes) -> set[str]:
    """The `const` and `static` names a function body declares, nested blocks
    included and nested functions not: none of them is a graph node."""
    names: set[str] = set()
    stack = [body]
    while stack:
        node = stack.pop()
        if node.type in ("const_item", "static_item") and (name := node.child_by_field_name("name")) is not None:
            names.add(_read_text(name, source))
        stack.extend(child for child in node.children if child.type != "function_item")
    return names


def _rust_in_pattern(node) -> bool:
    """Whether ``node`` is (inside) a `match` arm's or an `if let`'s pattern."""
    child, parent = node, node.parent
    while parent is not None and parent.type in _RUST_PATTERN_NODES:
        child, parent = parent, parent.parent
    if parent is None:
        return False
    if parent.type == "match_pattern":
        return True
    pattern = parent.child_by_field_name("pattern") if parent.type in ("let_condition", "let_chain") else None
    return pattern is not None and pattern.start_byte == child.start_byte and pattern.end_byte == child.end_byte


def _rust_type_last_segment(text: str) -> str:
    """`crate::a::Foo<T>` -> `Foo`: the last path segment, generic arguments dropped.
    A leading `&`, `&'a mut` or `dyn` (`impl Tr for &'a Foo`) is not part of it."""
    text = re.sub(r"^(?:&\s*(?:'\w+\s+)?(?:mut\s+)?|dyn\s+)", "", text.strip())
    depth, kept = 0, []
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif depth == 0:
            kept.append(ch)
    segments = [s.strip() for s in "".join(kept).split("::") if s.strip()]
    return segments[-1] if segments else ""


def extract_rust(path: Path) -> dict:
    """Extract functions, structs, enums, traits, impl methods, statics/consts, and use declarations from a .rs file."""
    try:
        import tree_sitter_rust as tsrust
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree-sitter-rust not installed"}

    try:
        language = Language(tsrust.language())
        parser = Parser(language)
        source = _read_source_bytes(path)
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, object, str | None, str | None, tuple[str, ...]]] = []
    impl_keys: dict[str, str | None] = {}

    # Names this file defines as a type (struct/enum/trait/union/alias) before it
    # starts referring to them. A local definition shadows the prelude, so its
    # references must not be dropped by the builtin-name filter further down.
    local_types: set[str] = set()

    def _scan_local_types(scan_node) -> None:
        if scan_node.type in ("struct_item", "enum_item", "trait_item", "union_item", "type_item"):
            name_node = scan_node.child_by_field_name("name")
            if name_node is not None:
                name = _read_text(name_node, source)
                if name:
                    local_types.add(name)
        for child in scan_node.children:
            _scan_local_types(child)

    _scan_local_types(root)
    local_type_names = frozenset(local_types)

    # `macro_rules!` macros defined in this file, keyed by bare name. A macro and a
    # function can share a name in Rust (`vec!` vs `vec`), and a macro is invoked
    # as `name!(...)` (a `macro_invocation`, never a `call_expression`), so macros
    # are resolved through their own registry rather than the function/type
    # label_to_nid to avoid cross-binding.
    macro_nids_by_name: dict[str, str] = {}

    # (bare impl type, method name) -> the method nodes this file's `impl` blocks
    # define, so an in-file `Type::f()` / `self.f()` binds to a method of that type.
    impl_methods: dict[tuple[str, str], set[str]] = {}

    # What graphify/rust_paths.py needs to resolve a written path to this file's
    # items: the inline `mod` blocks walk() is inside, each `mod x;` it declares
    # (with its `#[path]`), each `use` binding, and each item's path in the file.
    mod_scope: list[str] = []
    declared_mods: list[list[str]] = []
    inline_mods: list[str] = []
    use_bindings: list[list] = []
    # Nodes by id, so a declaration's node is found without scanning every node
    # (once per declaration, that scan was quadratic in the file's size).
    node_by_id: dict[str, dict] = {}

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            node_by_id[nid] = {
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": str_path,
                "source_location": f"L{line}",
            }
            nodes.append(node_by_id[nid])

    def mark_item_path(nid: str, *names: str) -> None:
        # Two items can share a node (`fn f` beside `mod tests { fn f }`), so a node
        # keeps every path it is declared at.
        node = node_by_id.get(nid)
        if node is not None:
            item_path = "::".join([*mod_scope, *names])
            paths = node.setdefault("_rust_paths", [])
            if item_path not in paths:
                paths.append(item_path)

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0,
                 context: str | None = None) -> None:
        edge = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": weight,
        }
        if context:
            edge["context"] = context
        edges.append(edge)

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    # Pre-scan the struct/enum/trait names declared in this file so a type used
    # above its declaration resolves exactly as a use below it already does,
    # rather than to a sourceless stub that the corpus-level rewire only folds
    # back when the name is unique across the corpus (#3782). Only these three
    # items are pre-registered: walk() never makes a node of a `type` alias or a
    # `union`, so registering one would leave its references pointing at no node.
    # The scan never descends further than walk() does, so every id registered
    # here ends up as a node. A type is matched by its exact name, since a `type
    # HANDLE` alias or an imported `HANDLE` folds to the id of a local `struct
    # Handle`. Distinct names that fold to one id get distinct ids (see
    # file_item_nid below), so a declared type is pre-registered unless a
    # non-type item has its exact name (`struct Foo {}` beside `fn Foo()`, which
    # still share one node); then the forward reference keeps the sourceless stub
    # v8's extractor gives it. An `impl` block counts as the type it names,
    # because walk() gives it the node of its type text (`impl Tr for HANDLE`
    # folds to `struct Handle`'s id, `impl Tr for &Handle<'_>` too); methods and
    # impl-scoped items have impl-qualified ids and are not counted.
    items_by_id: dict[str, set[tuple[bool, str]]] = {}
    declared_type_names: dict[str, set[str]] = {}

    def _scan_type_items(node) -> None:
        t = node.type
        if t in ("struct_item", "enum_item", "trait_item", "union_item", "type_item",
                 "function_item", "function_signature_item", "static_item", "const_item"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = _read_text(name_node, source)
                if t in ("static_item", "const_item") and name == "_":
                    return
                nid = _make_id(stem, name)
                is_type = t in ("struct_item", "enum_item", "trait_item", "union_item", "type_item")
                items_by_id.setdefault(nid, set()).add((is_type, name))
                if is_type:
                    declared_type_names.setdefault(nid, set()).add(name)
            return
        if t == "impl_item":
            type_node = node.child_by_field_name("type")
            if type_node is not None:
                type_name = _read_text(type_node, source).strip()
                refs: list[tuple[str, str, str]] = []
                _rust_collect_type_refs(type_node, source, False, refs)
                bare = next((ref for ref, role, _path in refs if role == "type"), type_name)
                items_by_id.setdefault(_make_id(stem, type_name), set()).add((True, bare))
                impl_bare_names[node.start_byte] = bare
            return
        if t == "use_declaration":
            return
        for child in node.children:
            _scan_type_items(child)

    impl_bare_names: dict[int, str] = {}
    _scan_type_items(root)
    local_type_names = frozenset(
        name
        for nid, names in declared_type_names.items()
        for name in names
        if all(is_type for is_type, item_name in items_by_id[nid] if item_name == name)
    )

    # The same scan is the collision census for file-level items (#3302's rule in
    # Rust's shape). Distinct names that fold to one id (`fn ignore` / `struct
    # Ignore`, `fn e` / `fn E`) used to keep whichever walk() met first and drop
    # the other, with the dropped item's calls credited to the survivor. Now the
    # one declared type (struct/enum/trait) in such a group keeps the plain id —
    # type positions and `impl` blocks name it, so its id and its methods' ids do
    # not move — and every other member is salted with a hash of its own name.
    # With no single declared type in the group, every member is salted, so the
    # outcome never depends on declaration order. Repeats of one name (`#[cfg]`
    # twins, an `impl` of a local type) are one member and still share a node.
    def file_item_nid(plain_nid: str, name: str) -> str:
        names = {item_name for _is_type, item_name in items_by_id.get(plain_nid, ())}
        if len(names) < 2 or name not in names:
            return plain_nid
        if declared_type_names.get(plain_nid) == {name}:
            return plain_nid
        salt = hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest()[:6]
        return _make_id(plain_nid, salt)

    def ensure_named_node(name: str, line: int) -> str:
        nid = file_item_nid(_make_id(stem, name), name)
        if nid in seen_ids or name in local_type_names:
            return nid
        nid = _make_id(name)
        if nid not in seen_ids:
            # The name isn't defined in this file, so this is a cross-file reference
            # (e.g. a `Thing` type annotation imported from another module). Emit a
            # SOURCELESS stub — like the inheritance-base path below — so the
            # corpus-level rewire can collapse it onto the real definition. A sourced
            # stub here makes _disambiguate_colliding_node_ids bake the referencing
            # file's path (with extension) into the id and blocks the rewire, which is
            # the phantom-duplicate-node bug (#1402).
            seen_ids.add(nid)
            node_by_id[nid] = {
                "id": nid,
                "label": name,
                "file_type": "code",
                "source_file": "",
                "source_location": "",
                "origin_file": str_path,
            }
            nodes.append(node_by_id[nid])
        return nid

    def add_type_ref(src_nid: str, ref: tuple[str, str, str], line: int, relation: str = "references",
                     context: str | None = None) -> None:
        """An edge from ``src_nid`` to the type ``ref`` names. A type this file does
        not declare gets a sourceless stub, and the edge its path and module
        scope, for rust_paths.retarget_path_edges to resolve."""
        ref_name, _role, ref_path = ref
        tgt = ensure_named_node(ref_name, line)
        if tgt == src_nid:
            return
        add_edge(src_nid, tgt, relation, line, context=context)
        if tgt == _make_id(ref_name):
            edges[-1]["_rust_type_path"] = ref_path
            if mod_scope:
                edges[-1]["_rust_use_scope"] = "::".join(mod_scope)

    def emit_param_return_refs(func_node, func_nid: str, line: int) -> None:
        params = func_node.child_by_field_name("parameters")
        if params is not None:
            for p in params.children:
                if p.type != "parameter":
                    continue
                type_node = p.child_by_field_name("type")
                refs: list[tuple[str, str, str]] = []
                _rust_collect_type_refs(type_node, source, False, refs, local_type_names)
                for ref in refs:
                    add_type_ref(func_nid, ref, line,
                                 context="generic_arg" if ref[1] == "generic_arg" else "parameter_type")
        return_type = func_node.child_by_field_name("return_type")
        if return_type is not None:
            refs = []
            _rust_collect_type_refs(return_type, source, False, refs, local_type_names)
            for ref in refs:
                add_type_ref(func_nid, ref, line, context="generic_arg" if ref[1] == "generic_arg" else "return_type")

    def walk(
        node,
        parent_impl_nid: str | None = None,
        parent_impl_type: str | None = None,
        parent_impl_key: str | None = None,
    ) -> None:
        t = node.type

        if t == "function_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                if parent_impl_nid:
                    func_nid = _make_id(parent_impl_nid, func_name)
                    add_node(func_nid, f".{func_name}()", line)
                    add_edge(parent_impl_nid, func_nid, "method", line)
                    if parent_impl_type:
                        owner = _rust_type_last_segment(parent_impl_type)
                        impl_methods.setdefault((owner, func_name), set()).add(func_nid)
                else:
                    func_nid = file_item_nid(_make_id(stem, func_name), func_name)
                    add_node(func_nid, f"{func_name}()", line)
                    add_edge(file_nid, func_nid, "contains", line)
                    mark_item_path(func_nid, func_name)
                emit_param_return_refs(node, func_nid, line)
                body = node.child_by_field_name("body")
                if body:
                    function_bodies.append((
                        func_nid,
                        body,
                        parent_impl_type,
                        parent_impl_key,
                        tuple(mod_scope),
                    ))
            return

        if t == "function_signature_item":
            # `fn greet(&self) -> String;` — a trait method with no body. This node type
            # had no branch at all, so a trait's required methods were unreachable: the
            # only method nodes in a graph came from impl blocks, and a trait with no
            # implementor in the corpus contributed none. Mirrors function_item, minus
            # the body walk (there is no body).
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                if parent_impl_nid:
                    func_nid = _make_id(parent_impl_nid, func_name)
                    add_node(func_nid, f".{func_name}()", line)
                    add_edge(parent_impl_nid, func_nid, "method", line)
                else:
                    func_nid = file_item_nid(_make_id(stem, func_name), func_name)
                    add_node(func_nid, f"{func_name}()", line)
                    add_edge(file_nid, func_nid, "contains", line)
                    mark_item_path(func_nid, func_name)
                emit_param_return_refs(node, func_nid, line)
            return

        if t == "macro_definition":
            # `macro_rules! name { ... }` defines a named, invocable item. This node
            # type had no branch, so the macro was dropped entirely: it was never a
            # node, and a `name!(...)` invocation of it had nothing to resolve to.
            # The id is macro-qualified so it never collides with a same-named fn or
            # type (`vec!` vs `vec`); the `!` suffix in the label marks it a macro.
            name_node = node.child_by_field_name("name")
            if name_node:
                macro_name = _read_text(name_node, source)
                if macro_name:
                    line = node.start_point[0] + 1
                    macro_nid = _make_id(stem, "macro", macro_name)
                    add_node(macro_nid, f"{macro_name}!", line)
                    add_edge(file_nid, macro_nid, "contains", line)
                    macro_nids_by_name.setdefault(macro_name, macro_nid)
            return

        # An associated type in an `impl` body (`type Output = Meters;`) is a
        # type_item too, but not a file-level item: it would mint one colliding
        # `Output` node per `impl Add`. Only an alias at module level is one.
        if t == "type_item" and node.parent is not None and node.parent.parent is not None \
                and node.parent.parent.type == "impl_item":
            return
        if t in ("struct_item", "enum_item", "trait_item", "union_item", "type_item"):
            name_node = node.child_by_field_name("name")
            if name_node:
                item_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                item_nid = file_item_nid(_make_id(stem, item_name), item_name)
                add_node(item_nid, item_name, line)
                mark_item_path(item_nid, item_name)
                declaration_node = node_by_id[item_nid]
                declaration_node["_rust_declaration_count"] = (
                    declaration_node.get("_rust_declaration_count", 0) + 1
                )
                add_edge(file_nid, item_nid, "contains", line)
                if t == "trait_item":
                    for c in node.children:
                        if c.type != "trait_bounds":
                            continue
                        for sub in c.children:
                            if not sub.is_named:
                                continue
                            refs: list[tuple[str, str, str]] = []
                            _rust_collect_type_refs(sub, source, False, refs, local_type_names)
                            for idx, ref in enumerate(refs):
                                if idx == 0:
                                    add_type_ref(item_nid, ref, line, "inherits")
                                else:
                                    add_type_ref(item_nid, ref, line, context="generic_arg")
                if t in ("struct_item", "union_item"):
                    for c in node.children:
                        if c.type != "field_declaration_list":
                            continue
                        for field in c.children:
                            if field.type != "field_declaration":
                                continue
                            type_node = field.child_by_field_name("type")
                            if type_node is None:
                                for fc in field.children:
                                    if fc.type in ("type_identifier", "generic_type",
                                                    "scoped_type_identifier",
                                                    "reference_type", "primitive_type"):
                                        type_node = fc
                                        break
                            refs = []
                            _rust_collect_type_refs(type_node, source, False, refs, local_type_names)
                            for ref in refs:
                                add_type_ref(item_nid, ref, field.start_point[0] + 1,
                                             context="generic_arg" if ref[1] == "generic_arg" else "field")
                    # Tuple structs (`struct Wrapper(pub Logger, Config);`) nest their
                    # positional field types directly under ordered_field_declaration_list
                    # with no field_declaration wrapper -- the same shape handled for tuple
                    # enum variants below. Without this branch these field type references
                    # are silently dropped.
                    for c in node.children:
                        if c.type != "ordered_field_declaration_list":
                            continue
                        fline = c.start_point[0] + 1
                        for tc in c.children:
                            if tc.type not in ("type_identifier", "generic_type",
                                               "scoped_type_identifier", "reference_type",
                                               "primitive_type", "tuple_type", "array_type"):
                                continue
                            refs = []
                            _rust_collect_type_refs(tc, source, False, refs, local_type_names)
                            for ref in refs:
                                add_type_ref(item_nid, ref, fline,
                                             context="generic_arg" if ref[1] == "generic_arg" else "field")
                if t == "enum_item":
                    # Variant payload types nest under enum_variant_list ->
                    # enum_variant -> ordered_field_declaration_list (tuple variant,
                    # `Click(Logger)`) | field_declaration_list (struct variant,
                    # `Resize { size: Dim }`). Neither was traversed, so every
                    # enum-variant type reference was silently dropped.
                    _TYPE_NODES = ("type_identifier", "generic_type",
                                   "scoped_type_identifier", "reference_type",
                                   "primitive_type", "tuple_type", "array_type")

                    def _emit_enum_type(type_node, at_line):
                        if type_node is None:
                            return
                        refs2: list[tuple[str, str, str]] = []
                        _rust_collect_type_refs(type_node, source, False, refs2, local_type_names)
                        for ref in refs2:
                            add_type_ref(item_nid, ref, at_line,
                                         context="generic_arg" if ref[1] == "generic_arg" else "field")

                    for c in node.children:
                        if c.type != "enum_variant_list":
                            continue
                        for variant in c.children:
                            if variant.type != "enum_variant":
                                continue
                            vline = variant.start_point[0] + 1
                            # Emit a node per variant with a case_of edge back to
                            # the enum. Only the variants' payload types were
                            # walked before, so the variants themselves (`Circle`,
                            # `Square`, `Empty`) never became nodes and the enum
                            # was left a memberless leaf. This is the Rust parity
                            # of Java #1719 / Kotlin #1738 / Swift / Scala enums.
                            # The variant name is the enum_variant's `identifier`.
                            vname_node = next(
                                (vc for vc in variant.children
                                 if vc.type == "identifier"),
                                None,
                            )
                            if vname_node is not None:
                                vname = _read_text(vname_node, source)
                                if vname:
                                    variant_nid = _make_id(item_nid, vname)
                                    add_node(variant_nid, vname, vline)
                                    add_edge(item_nid, variant_nid, "case_of", vline)
                                    mark_item_path(variant_nid, item_name, vname)
                            for vc in variant.children:
                                if vc.type == "ordered_field_declaration_list":
                                    for tc in vc.children:
                                        if tc.type in _TYPE_NODES:
                                            _emit_enum_type(tc, vline)
                                elif vc.type == "field_declaration_list":
                                    for field in vc.children:
                                        if field.type != "field_declaration":
                                            continue
                                        type_node = field.child_by_field_name("type")
                                        _emit_enum_type(type_node, field.start_point[0] + 1)
                if t == "trait_item":
                    # The methods a trait declares are its contract, and they were not
                    # reached: this branch returns below, so nothing walked the body.
                    # Descend the same way impl_item does, attributing each method to
                    # the trait node — so `explain <Trait>` can list what an
                    # implementor must provide.
                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            walk(child, parent_impl_nid=item_nid)
            return

        if t in ("static_item", "const_item"):
            # `static NAME: T = …;` / `const NAME: T = …;` at module level, or an
            # associated const inside an impl. Neither node type had a branch, so a
            # constant reached the graph only through files that referenced it,
            # never from the Rust that defines it (#3471).
            name_node = node.child_by_field_name("name")
            if name_node:
                item_name = _read_text(name_node, source)
                if item_name == "_":
                    return
                line = node.start_point[0] + 1
                if parent_impl_nid:
                    item_nid = _make_id(parent_impl_nid, item_name)
                    add_node(item_nid, f".{item_name}", line)
                    add_edge(parent_impl_nid, item_nid, "contains", line)
                    if parent_impl_type:  # `Type::CONST` names an associated const
                        mark_item_path(item_nid, _rust_type_last_segment(parent_impl_type), item_name)
                else:
                    item_nid = file_item_nid(_make_id(stem, item_name), item_name)
                    add_node(item_nid, item_name, line)
                    add_edge(file_nid, item_nid, "contains", line)
                    mark_item_path(item_nid, item_name)
                type_node = node.child_by_field_name("type")
                if type_node is not None:
                    refs: list[tuple[str, str, str]] = []
                    _rust_collect_type_refs(type_node, source, False, refs, local_type_names)
                    for ref in refs:
                        add_type_ref(item_nid, ref, line, context="generic_arg" if ref[1] == "generic_arg" else "field")
            return

        if t == "impl_item":
            type_node = node.child_by_field_name("type")
            trait_node = node.child_by_field_name("trait")
            impl_nid: str | None = None
            impl_type_bare: str | None = None
            impl_key: str | None = None
            if type_node:
                type_name = _read_text(type_node, source).strip()
                impl_nid = file_item_nid(
                    _make_id(stem, type_name), impl_bare_names.get(node.start_byte, type_name)
                )
                add_node(impl_nid, type_name, node.start_point[0] + 1)
                impl_key = _rust_simple_generic_impl_key(node, source)
                # Bare name (generics stripped) for typing a `self.` receiver
                # inside this block's methods (#2234) — `impl Foo<T>` types
                # `self` as `Foo`, not the literal `Foo<T>` text.
                impl_type_bare = type_name.split("<")[0].strip()
            if trait_node is not None and impl_nid is not None:
                refs: list[tuple[str, str, str]] = []
                _rust_collect_type_refs(trait_node, source, False, refs, local_type_names)
                for idx, ref in enumerate(refs):
                    if idx == 0:
                        add_type_ref(impl_nid, ref, node.start_point[0] + 1, "implements")
                    else:
                        add_type_ref(impl_nid, ref, node.start_point[0] + 1, context="generic_arg")
            body = node.child_by_field_name("body")
            if body:
                has_methods = any(
                    child.type in ("function_item", "function_signature_item")
                    for child in body.children
                )
                if impl_nid is not None and has_methods:
                    if impl_nid not in impl_keys:
                        impl_keys[impl_nid] = impl_key
                    elif impl_keys[impl_nid] != impl_key:
                        impl_keys[impl_nid] = None
                    impl_node = node_by_id[impl_nid]
                    if impl_keys[impl_nid]:
                        impl_node["_rust_impl_key"] = impl_keys[impl_nid]
                    else:
                        impl_node.pop("_rust_impl_key", None)
                for child in body.children:
                    walk(
                        child,
                        parent_impl_nid=impl_nid,
                        parent_impl_type=impl_type_bare,
                        parent_impl_key=impl_key,
                    )
            return

        if t == "mod_item":
            name_node = node.child_by_field_name("name")
            mod_name = _read_text(name_node, source) if name_node is not None else ""
            if mod_name and node.child_by_field_name("body") is None:
                # `mod x;` names a file; a `#[path = "…"]` above it names which.
                declared_mods.append(["::".join(mod_scope), mod_name, _rust_path_attribute(node, source)])
            elif mod_name:
                mod_scope.append(mod_name)
                inline_mods.append("::".join(mod_scope))
                try:
                    for child in node.children:
                        walk(child, parent_impl_nid=None)
                finally:
                    mod_scope.pop()
                return

        if t == "use_declaration":
            arg = node.child_by_field_name("argument")
            if arg:
                # The mods declared beside this `use`: an inline `mod` block is a scope of
                # its own, so a `use serde::..` inside `mod tests { .. }` sees none of the
                # file's top-level mods (#4564).
                scope_mods = {
                    _read_text(name_node, source)
                    for sibling in (node.parent.children if node.parent is not None else ())
                    if sibling.type == "mod_item"
                    and (name_node := sibling.child_by_field_name("name")) is not None
                }

                def name_use(written: str, path: str) -> None:
                    # The name as written, for the corpus rewire (#4439): the id folds its
                    # case. Only for a path rooted in this crate: `serde::Serialize` or
                    # `std::fs::File` is never the corpus's own `Serialize` or `File`. A
                    # first segment that is a `mod` declared in this scope is this crate's
                    # too (`mod item; pub use item::Invoice;`), unless the path starts with
                    # `::`, which always names a crate (#4564).
                    first = path.lstrip(": ").split("::")[0].strip()
                    ident = re.match(r"[A-Za-z_][A-Za-z0-9_]*", first)  # `item as items` -> `item`
                    if first in ("crate", "self", "super") or (
                        ident is not None
                        and ident.group() in scope_mods
                        and not path.lstrip().startswith("::")
                    ):
                        edges[-1]["_rust_use_name"] = written.split(" as ")[0].strip()

                def path_use(segments: tuple[str, ...]) -> None:
                    # The whole path, for graphify/rust_paths.py to resolve from this scope.
                    if segments:
                        edges[-1]["_rust_use_path"] = "::".join(segments)
                        if mod_scope:
                            edges[-1]["_rust_use_scope"] = "::".join(mod_scope)

                module_name = _rust_use_module(_read_text(arg, source))
                grouped = arg.type in ("use_list", "scoped_use_list")
                bindings = _rust_use_bindings(arg, source)
                is_pub = any(child.type == "visibility_modifier" for child in node.children)
                for _text, _line, segments, bound in bindings:
                    use_bindings.append(["::".join(mod_scope), bound, "::".join(segments), is_pub])
                if module_name:
                    tgt_nid = _make_id(module_name)
                    add_edge(file_nid, tgt_nid, "imports_from", node.start_point[0] + 1, context="import")
                    name_use(module_name, _read_text(arg, source))
                    if grouped:  # call evidence as in v8, then dropped (extract.py)
                        edges[-1]["_rust_use_group"] = True
                        group = arg.child_by_field_name("list")
                        if arg.type == "scoped_use_list" and group is not None:
                            path_use(_rust_path_segments(
                                source[arg.start_byte:group.start_byte].decode("utf-8", errors="replace")))
                    elif bindings:
                        path_use(bindings[0][2])
                # A group imports what the plain `use` of each of its bindings would.
                for binding, line, segments, _bound in bindings if grouped else []:
                    if binding_name := _rust_use_module(binding):
                        add_edge(file_nid, _make_id(binding_name), "imports_from", line, context="import")
                        name_use(binding_name, binding)
                        path_use(segments)
                        edges[-1]["_rust_use_binding"] = True  # not call evidence: v8 had no such edge
            return

        for child in node.children:
            walk(child, parent_impl_nid=None)

    walk(root)
    for marker, value in (("_rust_mods", declared_mods), ("_rust_inline_mods", inline_mods),
                          ("_rust_imports", use_bindings)):
        if value:
            node_by_id[file_nid][marker] = value

    # A bare `f()` can only reach a free function or a type (a tuple struct or
    # variant constructor), and `x.f()` only a method, so each call shape looks up
    # its own index: one shared name index let `drop(x)` bind to `impl Drop`'s
    # `drop` and `x.len()` to a free `fn len`.
    label_to_nid: dict[str, str] = {}
    method_label_to_nid: dict[str, str] = {}
    for n in nodes:
        raw = n["label"]
        normalised = raw.strip("()").lstrip(".")
        if raw.startswith("."):
            if raw.endswith(")"):
                method_label_to_nid[normalised] = n["id"]
        else:
            label_to_nid[normalised] = n["id"]

    # Nodes whose label has no `()` suffix are data definitions (structs, enums,
    # traits, statics), not callables. In Rust `Foo(x)` / `Foo { .. }` onto one
    # of them constructs a value rather than invoking a function, so the edge is
    # a `references` (context "constructor"), not a `calls`. Without this a
    # newtype used everywhere (`ClientId(id)`) reads as a top call hub.
    type_nids: set[str] = {n["id"] for n in nodes if not n["label"].endswith(")")}

    seen_call_pairs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []

    # This file's items by their path in it, for a module-qualified call that
    # stays in the file (`self::f()`, `super::f()` from an inline `mod`, `inner::f()`).
    # A path two nodes share names neither.
    in_file_items: dict[str, str | None] = {}
    for n in nodes:
        for item_path in n.get("_rust_paths") or ():
            in_file_items[item_path] = n["id"] if in_file_items.get(item_path, n["id"]) == n["id"] else None
    inline_mod_paths = frozenset(inline_mods)

    def in_file_target(segments: tuple[str, ...], scope: tuple[str, ...]) -> str | None:
        base, rest = list(scope), list(segments)
        if rest and rest[0] == "self":
            rest.pop(0)
        elif rest and rest[0] == "super":
            while rest and rest[0] == "super":
                if not base:
                    return None  # the parent module is another file
                base.pop()
                rest.pop(0)
        elif not rest or "::".join([*base, rest[0]]) not in inline_mod_paths:
            return None  # `crate::`, a `use`d name, a file module or another crate
        return in_file_items.get("::".join(base + rest)) if rest else None

    local_consts: dict[str, set[str]] = {}

    def reference(node, caller_nid: str, self_type: str | None, scope: tuple[str, ...]) -> None:
        # A variant or constant named as a value or in a pattern: `Err(E::Missing)`,
        # `E::Missing =>`, `LIMIT`, `crate::config::LIMIT`, `Type::CONST`. The
        # function of a call is a call (above), and a path's prefix is part of it.
        parent = node.parent
        if parent is None or parent.type in _RUST_NOT_A_REFERENCE_PARENTS:
            return
        # A struct-style variant or struct by its path: `E::Missing { at }`, as a
        # value or a pattern.
        struct_style = node.type == "scoped_type_identifier"
        if struct_style and parent.child_by_field_name(
                "name" if parent.type == "struct_expression" else "type") != node:
            return
        if parent.type == "call_expression" and parent.child_by_field_name("function") == node:
            return
        if parent.type in ("const_item", "static_item") and parent.child_by_field_name("name") == node:
            return  # a constant's own name, declared in a function body
        if node.type == "identifier":
            name = _read_text(node, source)
            if not _RUST_CONST_NAME.fullmatch(name) or name in local_consts.get(caller_nid, ()):
                return  # a function's own `const` shadows any outside it
            segments: tuple[str, ...] | None = (name,)
        else:
            segments = _rust_call_path(node, source)
            if not segments or not (_RUST_CONST_NAME.fullmatch(segments[-1]) or (
                    segments[-1][:1].isupper() and (struct_style or segments[-2][:1].isupper()))):
                return
            if segments[0] == "Self":
                if not self_type:
                    return
                segments = (_rust_type_last_segment(self_type), *segments[1:])
        if struct_style:
            context = "match" if parent.type == "struct_pattern" else "constructor"
        else:
            context = "match" if _rust_in_pattern(node) else (
                "const" if _RUST_CONST_NAME.fullmatch(segments[-1]) else "variant")
        if len(segments) == 1:
            tgt_nid = in_file_items.get("::".join([*scope, segments[0]]))
        else:
            tgt_nid = in_file_target(segments, scope)
            if tgt_nid is None and segments[-2][:1].isupper():
                # `Type::Variant`: the type as this module or the file's top sees it.
                tgt_nid = (in_file_items.get("::".join([*scope, *segments]))
                           or in_file_items.get("::".join(segments)))
        if tgt_nid is not None:
            pair = (caller_nid, tgt_nid)
            if tgt_nid in type_nids and tgt_nid != caller_nid and pair not in seen_call_pairs:
                seen_call_pairs.add(pair)
                add_edge(caller_nid, tgt_nid, "references", node.start_point[0] + 1, context=context)
            return
        if len(segments) == 1 or _rust_path_may_be_local(segments, local_types):
            rc_entry = {
                "caller_nid": caller_nid,
                "callee": segments[-1],
                "is_member_call": False,
                "rust_path": "::".join(segments),
                "rust_ref": context,
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
            }
            if scope:
                rc_entry["rust_scope"] = "::".join(scope)
            raw_calls.append(rc_entry)

    def walk_calls(
        node,
        caller_nid: str,
        self_type: str | None = None,
        self_impl_key: str | None = None,
        scope: tuple[str, ...] = (),
    ) -> None:
        if node.type in ("function_item", "use_declaration", "attribute_item"):
            return
        if node.type in ("scoped_identifier", "identifier") or (
                node.type == "scoped_type_identifier" and node.parent is not None
                and node.parent.type in ("struct_expression", "struct_pattern")):
            reference(node, caller_nid, self_type, scope)
        if node.type == "macro_invocation":
            # `name!(...)` invoking a macro_rules! macro defined in this file.
            # Resolve only a bare `identifier` macro against the local registry;
            # a `scoped_identifier` (`log::info!`) is cross-module/crate and stays
            # unresolved (fail-closed), matching how scoped calls are handled above.
            macro_node = node.child_by_field_name("macro")
            if macro_node is not None and macro_node.type == "identifier":
                macro_name = _read_text(macro_node, source)
                tgt_nid = macro_nids_by_name.get(macro_name)
                if tgt_nid and tgt_nid != caller_nid:
                    pair = (caller_nid, tgt_nid)
                    if pair not in seen_call_pairs:
                        seen_call_pairs.add(pair)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "calls",
                            "context": "call",
                            "confidence": "EXTRACTED",
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })
            # Fall through to the generic child recursion below rather than
            # returning, preserving the prior traversal of the invocation's
            # subtree (its arguments are a raw token tree, so this neither adds
            # nor drops any nested edges relative to before).
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node is not None and func_node.type == "generic_function":
                # `f::<T>()`, `a::f::<T>()`, `x.collect::<T>()`: the call of what is
                # named before the turbofish.
                func_node = func_node.child_by_field_name("function")
            callee_name: str | None = None
            is_member_call: bool = False
            is_scoped_call: bool = False
            is_self_call: bool = False
            qualifier_type: str | None = None
            call_path: tuple[str, ...] | None = None
            if func_node:
                if func_node.type == "identifier":
                    callee_name = _read_text(func_node, source)
                elif func_node.type == "field_expression":
                    is_member_call = True
                    field = func_node.child_by_field_name("field")
                    if field:
                        callee_name = _read_text(field, source)
                    receiver = func_node.child_by_field_name("value")
                    if receiver is not None and receiver.type == "self":
                        is_self_call = True
                elif func_node.type == "scoped_identifier":
                    # A path call never reaches the shared cross-file pass: bare
                    # last-segment lookup ignores crate boundaries and produced
                    # spurious INFERRED edges (#908). Unbound here, it goes to the
                    # path resolver (rust_paths.resolve_path_calls) with its path.
                    is_scoped_call = True
                    name = func_node.child_by_field_name("name")
                    if name:
                        callee_name = _read_text(name, source)
                    call_path = _rust_call_path(func_node, source)
                    qualifier_type = _rust_qualifier_type(func_node.child_by_field_name("path"), source)
                    if qualifier_type == "Self":
                        # A trait's default method has no impl type: "" matches
                        # no method, so `Self::f()` there binds nothing in-file.
                        qualifier_type = self_type or ""
            if (
                callee_name
                and callee_name not in _LANGUAGE_BUILTIN_GLOBALS
                and (callee_name not in _RUST_BUILTIN_TYPES or callee_name in local_types)
            ):
                if qualifier_type is not None or (is_self_call and self_type):
                    # `Type::f()` / `Self::f()` / `self.f()` name the type whose
                    # method is called: bind only to that type's method in this
                    # file. Matching the bare name instead sent `Arc::new(x)` and
                    # `B::new()` to whichever `fn new` the file happened to define.
                    owner = _rust_type_last_segment(qualifier_type or self_type or "")
                    owned = impl_methods.get((owner, callee_name), set())
                    tgt_nid = next(iter(owned)) if len(owned) == 1 else None
                    if tgt_nid is None and qualifier_type is not None:
                        # `Enum::Variant(x)` constructs a value: the variant node.
                        named = label_to_nid.get(callee_name)
                        tgt_nid = named if named in type_nids else None
                elif is_member_call:
                    tgt_nid = method_label_to_nid.get(callee_name)
                elif is_scoped_call:
                    # `inner::f()` binds to this file's `inner::f`, never to any `f`
                    # the file defines: `crate::rng::f()` beside a test module's own
                    # `f` used to bind to that one.
                    tgt_nid = in_file_target(call_path, scope) if call_path else None
                else:
                    tgt_nid = label_to_nid.get(callee_name)
                if tgt_nid and tgt_nid != caller_nid:
                    pair = (caller_nid, tgt_nid)
                    if pair not in seen_call_pairs:
                        seen_call_pairs.add(pair)
                        line = node.start_point[0] + 1
                        is_constructor = tgt_nid in type_nids
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "references" if is_constructor else "calls",
                            "context": "constructor" if is_constructor else "call",
                            "confidence": "EXTRACTED",
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })
                elif is_scoped_call:
                    if call_path and _rust_path_may_be_local(call_path, local_types):
                        rc_entry = {
                            "caller_nid": caller_nid,
                            "callee": callee_name,
                            "is_member_call": False,
                            "rust_path": "::".join(call_path),
                            "source_file": str_path,
                            "source_location": f"L{node.start_point[0] + 1}",
                        }
                        if scope:
                            rc_entry["rust_scope"] = "::".join(scope)
                        raw_calls.append(rc_entry)
                elif callee_name.lower() not in _RUST_TRAIT_METHOD_BLOCKLIST:
                    rc_entry = {
                        "caller_nid": caller_nid,
                        "callee": callee_name,
                        "is_member_call": is_member_call,
                        "source_file": str_path,
                        "source_location": f"L{node.start_point[0] + 1}",
                    }
                    if is_self_call and self_type:
                        rc_entry["rust_self_type"] = self_type
                        if self_impl_key:
                            rc_entry["rust_self_impl_key"] = self_impl_key
                    raw_calls.append(rc_entry)
        for child in node.children:
            walk_calls(child, caller_nid, self_type, self_impl_key, scope)

    for caller_nid, body_node, impl_type, impl_key, body_scope in function_bodies:
        local_consts.setdefault(caller_nid, set()).update(_rust_local_consts(body_node, source))
    for caller_nid, body_node, impl_type, impl_key, body_scope in function_bodies:
        walk_calls(body_node, caller_nid, impl_type, impl_key, body_scope)

    valid_ids = seen_ids
    clean_edges = []
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in valid_ids and (tgt in valid_ids or edge["relation"] in ("imports", "imports_from")):
            clean_edges.append(edge)

    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}
