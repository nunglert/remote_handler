"""Guard: the literal-included docs examples stay API-current.

The examples in ``docs/source/examples/`` cannot be *run* here -- they drive
live clusters / GPUs / VASP (the ``remote`` tier). But the failure mode that
actually rots documentation is API drift: a renamed class or a removed keyword
argument silently making an example wrong.

This module catches exactly that, without contacting any cluster:

* :func:`test_example_imports_resolve` loads each example module, so every
  ``import`` in it must resolve (catches moved/renamed symbols).
* :func:`test_example_kwargs_match_signatures` statically walks each example's
  AST and checks that every constructor / classmethod call on an imported
  symbol uses only keyword arguments the real signature accepts.

No example body runs (the work lives under ``if __name__ == "__main__":``), so
nothing connects anywhere. Each example is also confirmed to be wired into
``examples.rst``.
"""

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = _REPO_ROOT / "docs" / "source" / "examples"
EXAMPLES_RST = _REPO_ROOT / "docs" / "source" / "examples.rst"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("example_*.py"))


def _load(path: Path):
    """Import an example by path; runs only its top-level imports."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve(node, namespace):
    """Resolve an ``ast`` call target (Name / Attribute) to an object."""
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _resolve(node.value, namespace)
        if parent is None:
            return None
        return getattr(parent, node.attr, None)
    return None


def test_examples_exist():
    assert EXAMPLE_FILES, f"no example_*.py found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
def test_example_imports_resolve(path):
    """Every symbol an example imports must still exist."""
    _load(path)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
def test_example_kwargs_match_signatures(path):
    """Keyword args used in the example must be accepted by the real API."""
    module = _load(path)
    namespace = vars(module)
    tree = ast.parse(path.read_text())

    problems = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        target = _resolve(call.func, namespace)
        if target is None or not callable(target):
            # Local variable, builtin, or instance method we cannot resolve
            # without executing the example -- skip (can't be validated here).
            continue
        try:
            params = inspect.signature(target).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            continue  # accepts **kwargs -> any name is valid
        name = getattr(target, "__qualname__", repr(target))
        for keyword in call.keywords:
            if keyword.arg is None:
                continue  # ``**mapping`` splat
            if keyword.arg not in params:
                problems.append(f"{name}: unexpected keyword '{keyword.arg}'")

    assert not problems, "stale example keyword args:\n" + "\n".join(problems)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
def test_example_is_referenced_in_rst(path):
    """Each example script must be literal-included by examples.rst."""
    rst = EXAMPLES_RST.read_text()
    assert (
        path.name in rst
    ), f"{path.name} is not literalincluded in examples.rst"
