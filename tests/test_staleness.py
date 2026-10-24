"""The CLI says when a graph may not describe the checkout it is queried in (#04).

A lookup made mid-rebase, or after a commit no rebuild has caught up with, read
a missing name as an absent one. A fresh graph, or one stamped by another
checkout, must print nothing.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

import graphify.__main__ as mainmod
from graphify.staleness import staleness_note


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True).stdout.strip()


def _age(path, seconds=100):
    t = path.stat().st_mtime - seconds
    os.utime(path, (t, t))


@pytest.fixture
def repo(tmp_path):
    """A repo with src/a.py committed and a graph of it built at that commit."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def run():\n    pass\n")
    (tmp_path / "README.md").write_text("# r\n")
    _git(tmp_path, "add", ".")
    commit = _git(tmp_path, "commit", "-qm", "one")
    _age(tmp_path / "src" / "a.py")
    _age(tmp_path / "README.md")
    out = tmp_path / "graphify-out"
    out.mkdir()
    gp = out / "graph.json"
    gp.write_text(json.dumps({
        "directed": False, "multigraph": False, "graph": {}, "built_at_commit": commit,
        "nodes": [{"id": "src_a_run", "label": "run()", "source_file": "src/a.py",
                   "source_location": "L1"}],
        "links": [],
    }))
    return tmp_path, gp, commit


def _note(gp, commit, **kw):
    return staleness_note(gp, commit, ["src/a.py"], **kw)


def test_a_fresh_graph_says_nothing(repo):
    _root, gp, commit = repo
    assert _note(gp, commit, found_files=["src/a.py"]) is None


def test_a_commit_the_graph_has_not_caught_up_with_is_named(repo):
    root, gp, commit = repo
    (root / "src" / "a.py").write_text("def run():\n    return 1\n")
    head = _git(root, "commit", "-qam", "two")
    note = _note(gp, commit, missed=True)
    assert note.startswith(f"[graphify] 1 file the graph covers changed since it was built at {commit[:7]} (HEAD is {head[:7]})")
    assert "the name may be new rather than absent" in note
    assert f"graphify update {root}" in note


def test_a_commit_of_files_the_graph_does_not_cover_says_nothing(repo):
    root, gp, commit = repo
    (root / "README.md").write_text("# changed\n")
    _git(root, "commit", "-qam", "docs")
    assert _note(gp, commit) is None


def test_a_commit_the_graph_was_refreshed_after_says_nothing(repo):
    # The working-tree refresh ran after the edit and before the commit: the
    # graph holds the content although its stamp is older than HEAD.
    root, gp, commit = repo
    (root / "src" / "a.py").write_text("def run():\n    return 1\n")
    _age(root / "src" / "a.py")
    _git(root, "commit", "-qam", "two")
    assert _note(gp, commit) is None


def test_a_rebase_in_progress_is_named(repo):
    root, gp, commit = repo
    (root / ".git" / "rebase-merge").mkdir()
    note = _note(gp, commit, missed=True)
    assert note.startswith(f"[graphify] a rebase is in progress in {root}")
    assert "the name may be new rather than absent" in note


def test_the_found_nodes_file_written_after_the_graph_is_named(repo):
    root, gp, commit = repo
    (root / "src" / "a.py").write_text("def run():\n    return 2\n")
    assert _note(gp, commit, found_files=["src/a.py"]) == (
        f"[graphify] src/a.py changed after the graph was built at {commit[:7]}, "
        "so its lines and connections may be out of date.")


def test_a_graph_stamped_by_another_checkout_says_nothing(repo):
    # A graph kept in one repo that describes another (a decomp beside the engine).
    root, gp, _commit = repo
    (root / "src" / "a.py").write_text("x = 1\n")
    _git(root, "commit", "-qam", "two")
    assert _note(gp, "0" * 40, found_files=["src/a.py"]) is None


def test_outside_git_says_nothing(tmp_path):
    gp = tmp_path / "graph.json"
    gp.write_text("{}")
    assert staleness_note(gp, "abc", []) is None


def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", *argv])
    try:
        mainmod.main()
    except SystemExit:
        pass
    return capsys.readouterr()


@pytest.mark.parametrize("cmd", [("explain", "run()"), ("explain", "missing_thing"),
                                 ("affected", "run()"), ("query", "run")])
def test_the_cli_warns_on_stderr_and_leaves_stdout_alone(monkeypatch, capsys, repo, cmd):
    root, gp, commit = repo
    fresh = _run(monkeypatch, capsys, *cmd, "--graph", str(gp))
    assert fresh.err == ""
    (root / ".git" / "rebase-merge").mkdir()
    stale = _run(monkeypatch, capsys, *cmd, "--graph", str(gp))
    assert stale.out == fresh.out
    assert stale.err.startswith("[graphify] a rebase is in progress")
