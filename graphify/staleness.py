"""Whether a graph may no longer describe the checkout a CLI lookup runs against.

The MCP server compares the build commit with HEAD (`serve._build_commit_line`),
but the CLI said nothing, so a lookup made mid-rebase, or after a commit no
rebuild has caught up with, read a missing name as an absent one. This is the
CLI's check. It stays cheap: one `git rev-parse`, a few stats, and a `git diff`
or `git cat-file` only when HEAD has moved past the build commit.

Every check fails silent: no git, no stamp, or a stamp from another checkout
(a graph kept in one repo that describes another) prints nothing, rather than
a warning that would be wrong.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

_IN_PROGRESS = (
    ("rebase-merge", "a rebase"),
    ("rebase-apply", "a rebase"),
    ("MERGE_HEAD", "a merge"),
    ("CHERRY_PICK_HEAD", "a cherry-pick"),
)


def _git(cwd: Path, *args: str) -> list[str] | None:
    try:
        r = subprocess.run(
            ["git", *args], cwd=str(cwd), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=3,
        )
    except Exception:
        return None
    return r.stdout.splitlines() if r.returncode == 0 else None


def _newer_or_gone(path: Path, in_graph: bool, graph_mtime: float) -> bool:
    try:
        return path.stat().st_mtime > graph_mtime
    except OSError:
        return in_graph


def staleness_note(
    graph_path: Path,
    built_at_commit: str | None,
    source_files: Iterable[str],
    *,
    found_files: Iterable[str] = (),
    missed: bool = False,
) -> str | None:
    """One line saying why the graph may be stale, or None when nothing says so.

    *source_files* are every node's `source_file` (which kinds of file the graph
    covers); *found_files* are those of the node(s) the lookup answered with, whose
    mtimes are checked against the graph's. *missed* words the note for a lookup
    that matched nothing: the name may be new rather than absent.
    """
    graph_path = Path(graph_path)
    out = _git(graph_path.parent, "rev-parse", "--show-toplevel", "--absolute-git-dir", "HEAD")
    if not out or len(out) < 3:
        return None
    root, git_dir, head = Path(out[0]), Path(out[1]), out[2]
    commit = (built_at_commit or "").strip() or None
    if commit and commit != head and _git(root, "cat-file", "-e", f"{commit}^{{commit}}") is None:
        return None  # stamped by another checkout: this repo cannot judge it
    consequence = (
        "so the name may be new rather than absent." if missed
        else "so its lines and connections may be out of date."
    )
    where = f"built at {commit[:7]}" if commit else "built"

    for marker, what in _IN_PROGRESS:
        if (git_dir / marker).exists():
            return (f"[graphify] {what} is in progress in {root}: the graph was {where} "
                    f"and does not describe the working tree, {consequence}")

    try:
        graph_mtime = graph_path.stat().st_mtime
    except OSError:
        return None

    if commit and commit != head:
        # A commit moves HEAD without changing what a graph refreshed from the
        # working tree already holds, so a file counts only when it was written
        # after the graph, or deleted while the graph still has it.
        changed = _git(root, "diff", "--name-only", commit, head) or []
        files = {f for f in source_files if f}
        kinds = {Path(f).suffix for f in files if Path(f).suffix}
        behind = [f for f in changed
                  if (f in files or Path(f).suffix in kinds) and _newer_or_gone(root / f, f in files, graph_mtime)]
        if behind:
            noun = "file" if len(behind) == 1 else "files"
            rebuild = (f" Rebuild with `graphify update {root}`."
                       if graph_path == root / "graphify-out" / "graph.json" else "")
            return (f"[graphify] {len(behind)} {noun} the graph covers changed since it was "
                    f"{where} (HEAD is {head[:7]}), {consequence}{rebuild}")

    for f in dict.fromkeys(found_files):
        try:
            if f and (root / f).stat().st_mtime > graph_mtime:
                return (f"[graphify] {f} changed after the graph was {where}, "
                        f"so its lines and connections may be out of date.")
        except OSError:
            continue
    return None
