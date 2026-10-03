"""Load selected definitions from the original scripts in ``legacy/`` for golden tests.

The original training scripts run training at import time, so they cannot be imported.
This loader parses a script, executes only its top-level imports (skipping those whose
packages are not installed, e.g. torchvision) and the requested top-level function,
class and constant definitions, and returns them in a namespace.
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

LEGACY_DIR = Path(__file__).resolve().parents[1] / "legacy"


def load_legacy(filename, names, extra_globals=None):
    path = LEGACY_DIR / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    namespace = {"__name__": f"legacy_{path.stem}", "__file__": str(path)}
    if extra_globals:
        namespace.update(extra_globals)
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ast.Module(body=[node], type_ignores=[])
            try:
                exec(compile(module, str(path), "exec"), namespace)
            except ImportError:
                pass

    wanted = set(names)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted for t in node.targets
        ):
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    missing = wanted - set(namespace)
    if missing:
        raise KeyError(f"{sorted(missing)} not found in {filename}")
    return SimpleNamespace(**{name: namespace[name] for name in names})
