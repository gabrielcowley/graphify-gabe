"""`explain "Owner::member"` and `"Owner.member"` name a method, variant or field
by the type it belongs to.

A method is labelled `.name()` and a variant by its bare name, so the spelling a
Rust or C++ reader types (`SidesApart::observe`, `Field::PlayerX`) matched no
label, and the miss suggested the type and its fields but not its methods. Twin
methods in one file could then be told apart only by node id.
"""
from __future__ import annotations

import json

import pytest

import graphify.__main__ as mainmod
from graphify.cli import EXIT_AMBIGUOUS, EXIT_NOT_FOUND


_NODES = [
    ("src_x", "x.rs", "src/x.rs", "L1"),
    ("src_x_alpha", "Alpha", "src/x.rs", "L3"),
    ("src_x_alpha_field_key", "Alpha.key", "src/x.rs", "L4"),
    ("src_x_alpha_nonce", ".nonce()", "src/x.rs", "L6"),
    ("src_x_alpha_reset", ".reset()", "src/x.rs", "L8"),
    ("src_x_beta", "Beta", "src/x.rs", "L10"),
    ("src_x_beta_nonce", ".nonce()", "src/x.rs", "L12"),
    ("src_x_holder", "Holder", "src/x.rs", "L14"),
    ("src_x_holder_field_inner", "Holder.inner", "src/x.rs", "L15"),
    ("src_x_holder_t", "Holder<T>", "src/x.rs", "L17"),
    ("src_x_holder_t_get", ".get()", "src/x.rs", "L18"),
    ("src_y", "y.rs", "src/y.rs", "L1"),
    ("src_y_color", "Color", "src/y.rs", "L2"),
    ("src_y_color_red", "Red", "src/y.rs", "L3"),
    ("src_z", "z.rs", "src/z.rs", "L1"),
    ("src_z_color", "Color", "src/z.rs", "L2"),
    ("src_z_color_blue", "Blue", "src/z.rs", "L3"),
]
_LINKS = [
    ("src_x", "src_x_alpha", "contains"),
    ("src_x_alpha", "src_x_alpha_field_key", "contains"),
    ("src_x_alpha", "src_x_alpha_nonce", "method"),
    ("src_x_alpha", "src_x_alpha_reset", "method"),
    ("src_x", "src_x_beta", "contains"),
    ("src_x_beta", "src_x_beta_nonce", "method"),
    ("src_x", "src_x_holder", "contains"),
    ("src_x_holder", "src_x_holder_field_inner", "contains"),
    ("src_x_holder_t", "src_x_holder_t_get", "method"),
    ("src_y", "src_y_color", "contains"),
    ("src_y_color", "src_y_color_red", "case_of"),
    ("src_z", "src_z_color", "contains"),
    ("src_z_color", "src_z_color_blue", "case_of"),
]


def _write(tmp_path):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps({
        "directed": False, "multigraph": False, "graph": {},
        "nodes": [{"id": i, "label": lbl, "source_file": sf, "source_location": loc}
                  for i, lbl, sf, loc in _NODES],
        "links": [{"source": a, "target": b, "relation": rel, "confidence": "EXTRACTED"}
                  for a, b, rel in _LINKS],
    }))
    return p


def _explain(monkeypatch, capsys, graph_path, label):
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv",
        ["graphify", "explain", label, "--graph", str(graph_path)])
    try:
        mainmod.main()
    except SystemExit as exc:
        return capsys.readouterr().out, exc.code
    return capsys.readouterr().out, None


@pytest.mark.parametrize("label,node", [
    ("Alpha::nonce", "src_x_alpha_nonce"),
    ("Alpha.nonce", "src_x_alpha_nonce"),
    ("Alpha::nonce()", "src_x_alpha_nonce"),
    ("Alpha::.nonce()", "src_x_alpha_nonce"),
    ("Beta::nonce", "src_x_beta_nonce"),
    ("Alpha::key", "src_x_alpha_field_key"),
    # The variant's owner name is shared by two files; only one has a `Red`.
    ("Color::Red", "src_y_color_red"),
    ("Color::Blue", "src_z_color_blue"),
    # A generic type's methods hang off its `impl` node, `Holder<T>`.
    ("Holder::get", "src_x_holder_t_get"),
    ("Holder::inner", "src_x_holder_field_inner"),
])
def test_an_owner_names_its_member(monkeypatch, tmp_path, capsys, label, node):
    out, code = _explain(monkeypatch, capsys, _write(tmp_path), label)
    assert f"ID:        {node}" in out, out
    assert code in (None, 0)


def test_an_exact_field_label_is_unchanged(monkeypatch, tmp_path, capsys):
    out, _ = _explain(monkeypatch, capsys, _write(tmp_path), "Alpha.key")
    assert "ID:        src_x_alpha_field_key" in out


def test_a_missing_member_lists_the_owners_members(monkeypatch, tmp_path, capsys):
    out, code = _explain(monkeypatch, capsys, _write(tmp_path), "Alpha::nothing")
    assert code == EXIT_NOT_FOUND
    assert out.startswith("No node matching 'Alpha::nothing' found.")
    assert "Alpha has no member 'nothing'" in out
    for member in (".nonce()", ".reset()", "Alpha.key"):
        assert member in out
    assert "Beta" not in out
    assert "Retry with Owner::member (e.g. Alpha::" in out


def test_a_member_sharing_a_word_is_listed_first(monkeypatch, tmp_path, capsys):
    out, _ = _explain(monkeypatch, capsys, _write(tmp_path), "Alpha::reset_all")
    listed = [line.split()[0] for line in out.splitlines() if line.startswith("  ")]
    assert listed[0] == ".reset()", listed


def test_a_generic_owners_two_nodes_are_named_once(monkeypatch, tmp_path, capsys):
    out, _ = _explain(monkeypatch, capsys, _write(tmp_path), "Holder::nothing")
    assert "Holder has no member 'nothing'" in out
    assert ".get()" in out and "Holder.inner" in out


def test_twins_in_one_file_are_listed_by_owner(monkeypatch, tmp_path, capsys):
    out, code = _explain(monkeypatch, capsys, _write(tmp_path), "src/x.rs::.nonce()")
    assert code == EXIT_AMBIGUOUS
    assert "Alpha::nonce" in out and "Beta::nonce" in out
    assert "Retry with the Owner::member form shown" in out


def test_an_unknown_owner_is_an_ordinary_miss(monkeypatch, tmp_path, capsys):
    out, code = _explain(monkeypatch, capsys, _write(tmp_path), "Gamma::nonce")
    assert code == EXIT_NOT_FOUND
    assert "has no member" not in out


def _run(monkeypatch, capsys, argv):
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", *argv])
    try:
        mainmod.main()
    except SystemExit as exc:
        return capsys.readouterr().out, exc.code
    return capsys.readouterr().out, None


def test_path_takes_the_member_form(monkeypatch, tmp_path, capsys):
    out, code = _run(monkeypatch, capsys,
                     ["path", "Alpha", "Alpha::nonce", "--graph", str(_write(tmp_path))])
    assert "Shortest path (1 hops)" in out, out
    assert code in (None, 0)


def test_affected_takes_the_member_form(monkeypatch, tmp_path, capsys):
    out, _ = _run(monkeypatch, capsys,
                  ["affected", "Beta::nonce", "--graph", str(_write(tmp_path))])
    assert "No unique node match" not in out, out
