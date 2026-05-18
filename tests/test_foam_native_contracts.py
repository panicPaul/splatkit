from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FOAM_PACKAGE_ROOTS = [
    REPO_ROOT / "packages" / "ember-native-powerfoam" / "src",
    REPO_ROOT / "packages" / "ember-native-radfoam" / "src",
]


def _python_files() -> list[Path]:
    return [
        path
        for package_root in FOAM_PACKAGE_ROOTS
        for path in package_root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def test_foam_init_files_are_facades_only() -> None:
    for path in _python_files():
        if path.name != "__init__.py":
            continue
        tree = _tree(path)
        body = list(tree.body)
        if body and isinstance(body[0], ast.Expr):
            assert isinstance(body[0].value, ast.Constant)
            assert isinstance(body[0].value.value, str)
            body = body[1:]
        for node in body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                continue
            if isinstance(node, ast.Assign):
                targets = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
                if targets == ["__all__"]:
                    continue
            raise AssertionError(f"{path} contains implementation code")


def test_foam_runtime_does_not_import_upstream_packages() -> None:
    for path in _python_files():
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                upstream_names = {
                    alias.name.split(".", maxsplit=1)[0]
                    for alias in node.names
                } & {"powerfoam", "radfoam"}
                assert not upstream_names, f"{path} imports {upstream_names}"
            if isinstance(node, ast.ImportFrom) and node.module:
                upstream_name = node.module.split(".", maxsplit=1)[0]
                assert upstream_name not in {"powerfoam", "radfoam"}, (
                    f"{path} imports upstream {node.module}"
                )


def test_foam_adapter_code_does_not_call_autograd_function_apply() -> None:
    for path in _python_files():
        relative_parts = path.relative_to(REPO_ROOT).parts
        if "native" in relative_parts and "warp" in relative_parts:
            continue
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "apply", (
                    f"{path} calls .apply instead of a custom op"
                )
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == "Function":
                        raise AssertionError(
                            f"{path} defines torch.autograd.Function subclass"
                        )
