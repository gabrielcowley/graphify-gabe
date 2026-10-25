"""A C function named as a value is an `indirect_call` of what names it.

C reaches most callbacks through a pointer: `gTasks[id].func = Task_X`,
`CreateTask(Task_X, 0)`, a command table indexed by opcode. None of these is a
call expression, so such a function had no caller in the graph. Now an
identifier naming a function on an assignment's right side, in an initializer,
as a call argument, in a `return` or as an `==`/`!=` operand is an INFERRED
`indirect_call`. A parameter or local of that name shadows it. A file-level
table is a node, as every file-scope variable is, and holds those edges: from
the file node they would replace the file's `contains` edges, since the graph
keeps one edge per pair. And a direct call between the same two functions, either way
round, is kept over the indirect edge.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

_A = (
    "typedef void (*TaskFunc)(unsigned char);\n"                            # 1
    "static void Task_B(unsigned char t);\n"                                # 2
    "static const TaskFunc sTasks[] = { Task_B, [1] = Task_A };\n"          # 3
    "static void (*const sCmds[])(void) = { Cmd_go };\n"                    # 4
    "void Run(TaskFunc func)\n"                                             # 5
    "{\n"                                                                   # 6
    "    CreateTask(func, 0);\n"                                            # 7
    "    gTasks[0].func = Task_A;\n"                                        # 8
    "    gMain.cb = &CB2_X;\n"                                              # 9
    "    if (gMain.cb == CB2_Y) {}\n"                                       # 10
    "    CreateTask(Task_B, 0);\n"                                          # 11
    "    Task_B(3);\n"                                                      # 12
    "}\n"                                                                   # 13
    "TaskFunc Pick(int x) { return x ? Task_A : (TaskFunc)Task_B; }\n"     # 14
    "static void Task_B(unsigned char t) { Helper(); }\n"                   # 15
    "void Task_A(unsigned char t) {}\n"                                     # 16
    "void Cmd_go(void) {}\n"                                                # 17
)
_B = (
    "void CB2_X(void) {}\n"                                                 # 1
    "void CB2_Y(void) {}\n"                                                 # 2
    "void Helper(void) { CreateTask(Task_B, 1); }\n"                        # 3
    "static const void *sOther[] = { Task_A };\n"                           # 4
    "void Other(void) { CreateTask(Task_A, 1); }\n"                         # 5
    "void Step(void) {}\n"                                                  # 6
    "void Use(TaskFunc Step) { TaskFunc CB2_X = Step; CreateTask(CB2_X, 0); }\n"  # 7
)


def _edges(tmp_path: Path) -> tuple[set, set]:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.c").write_text(_A, encoding="utf-8")
    (src / "b.c").write_text(_B, encoding="utf-8")
    r = extract([src / "a.c", src / "b.c"], root=tmp_path, cache_root=tmp_path / "out")
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    edges = {
        (labels[e["source"]], e["relation"], labels[e["target"]], e.get("context"),
         e["source_location"])
        for e in r["edges"] if e["source"] in labels and e["target"] in labels
    }
    return edges, set(labels.values())


def test_a_function_named_as_a_value_is_an_indirect_call(tmp_path: Path):
    edges, _ = _edges(tmp_path)
    assert ("Run()", "indirect_call", "Task_A()", "assignment", "L8") in edges
    assert ("Run()", "indirect_call", "CB2_X()", "assignment", "L9") in edges   # through &
    assert ("Run()", "indirect_call", "CB2_Y()", "comparison", "L10") in edges
    # both arms of ?:, one through a cast
    assert ("Pick()", "indirect_call", "Task_A()", "return", "L14") in edges
    assert ("Pick()", "indirect_call", "Task_B()", "return", "L14") in edges
    # across files: Task_A is a.c's, named in b.c
    assert ("Other()", "indirect_call", "Task_A()", "argument", "L5") in edges


def test_a_parameter_or_local_of_the_name_is_not_the_function(tmp_path: Path):
    edges, _ = _edges(tmp_path)
    # Use's parameter `Step` and local `CB2_X` shadow the functions of those names
    assert not any(s == "Use()" and r == "indirect_call" for s, r, _t, _c, _l in edges)


def test_a_table_naming_its_files_functions_is_a_node(tmp_path: Path):
    edges, labels = _edges(tmp_path)
    assert "sTasks" in labels and "sCmds" in labels   # `(*const sCmds[])(void)` too
    assert ("sTasks", "indirect_call", "Task_B()", "collection", "L3") in edges
    assert ("sTasks", "indirect_call", "Task_A()", "collection", "L3") in edges   # [1] = Task_A
    assert ("sCmds", "indirect_call", "Cmd_go()", "collection", "L4") in edges
    # so the file keeps containing the functions the tables name
    for fn in ("Task_A()", "Task_B()", "Cmd_go()"):
        assert ("a.c", "contains", fn, None, f"L{ {'Task_A()': 16, 'Task_B()': 15, 'Cmd_go()': 17}[fn] }") \
            in {(s, r, t, None, loc) for s, r, t, _c, loc in edges}


def test_a_table_of_another_files_functions_is_a_node_too(tmp_path: Path):
    # every file-scope variable is a node (test_c_globals.py)
    edges, labels = _edges(tmp_path)
    assert "sOther" in labels
    assert ("sOther", "indirect_call", "Task_A()", "collection", "L4") in edges


def test_a_direct_call_is_kept_over_an_indirect_one(tmp_path: Path):
    edges, _ = _edges(tmp_path)
    # Run names Task_B (L11) before calling it (L12)
    run_b = {(r, loc) for s, r, t, _c, loc in edges if s == "Run()" and t == "Task_B()"}
    assert run_b == {("calls", "L12")}
    # Task_B calls Helper, which names Task_B: the pair keeps the call
    pair = {(s, r) for s, r, t, _c, _l in edges if {s, t} == {"Task_B()", "Helper()"}}
    assert pair == {("Task_B()", "calls")}
