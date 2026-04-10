"""Architecture boundary regressions."""

from __future__ import annotations

import ast
from pathlib import Path

import perceptimg
from perceptimg import api


def _collect_forbidden_imports(
    path: Path,
    *,
    forbidden_prefixes: tuple[str, ...],
    allow_relative_modules: tuple[str, ...] = (),
) -> list[tuple[int, str]]:
    forbidden_roots = {
        prefix.strip(".").rsplit(".", maxsplit=1)[-1] for prefix in forbidden_prefixes
    }
    allow_relative = set(allow_relative_modules)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(forbidden_prefixes):
                    violations.append((node.lineno, f"from absolute module '{alias.name}'"))

        elif isinstance(node, ast.ImportFrom):
            # Absolute imports.
            if node.level == 0 and node.module:
                if node.module.startswith(forbidden_prefixes):
                    violations.append((node.lineno, f"from '{node.module}' import ..."))
                continue

            # Relative imports with restricted roots.
            if node.level >= 1 and node.module is not None:
                module_head = node.module.split(".")[0]
                if module_head not in allow_relative and module_head in forbidden_roots:
                    violations.append((node.lineno, f"from ..{node.module} import ..."))
                continue

            # Relative imports like `from .. import core`.
            if node.level >= 1 and node.module is None:
                imported_modules = {alias.name.split(".")[0] for alias in node.names}
                for module_name in imported_modules:
                    if module_name in forbidden_roots and module_name not in allow_relative:
                        violations.append(
                            (
                                node.lineno,
                                f"from .. import {', '.join(sorted(imported_modules))}",
                            )
                        )
                        break
    return violations


def test_core_does_not_import_adapters_or_application() -> None:
    """Core package must not import application/adapters directly."""
    root = Path(__file__).resolve().parents[1]
    core_dir = root / "perceptimg" / "core"
    allowed_exceptions = {"perceptimg/core/batch/__init__.py"}

    violations: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(core_dir.rglob("*.py")):
        if path.name == "__main__.py":
            continue
        if str(path) in {str(root / exception) for exception in allowed_exceptions}:
            continue
        file_violations = _collect_forbidden_imports(
            path,
            forbidden_prefixes=("perceptimg.application.", "perceptimg.adapters."),
        )
        if file_violations:
            violations[str(path)] = file_violations

    assert not violations, f"Architecture boundary violations in core: {violations}"


def test_top_level_modules_do_not_cross_layer_boundaries() -> None:
    """Top-level public modules should keep boundary imports explicit and minimal."""
    root = Path(__file__).resolve().parents[1]
    module_rules: tuple[tuple[Path, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            root / "perceptimg" / "cli.py",
            ("perceptimg.core", "perceptimg.application", "perceptimg.adapters"),
            ("api",),
        ),
        (
            root / "perceptimg" / "__init__.py",
            ("perceptimg.core", "perceptimg.application", "perceptimg.adapters"),
            (),
        ),
        (
            root / "perceptimg" / "api.py",
            ("perceptimg.adapters",),
            (),
        ),
    )
    for module_path, forbidden, allow_relative in module_rules:
        module_violations = _collect_forbidden_imports(
            module_path,
            forbidden_prefixes=forbidden,
            allow_relative_modules=allow_relative,
        )
        assert (
            not module_violations
        ), f"Boundary imports violated in {module_path}: {module_violations}"


def test_cli_does_not_import_internal_core_or_adapters() -> None:
    """CLI should not import core/adapters directly."""
    root = Path(__file__).resolve().parents[1]
    cli_module = root / "perceptimg" / "cli.py"
    violations = _collect_forbidden_imports(
        cli_module,
        forbidden_prefixes=(
            "perceptimg.core",
            "perceptimg.application",
            "perceptimg.adapters",
        ),
        allow_relative_modules=("api",),
    )
    assert not violations, f"Forbidden CLI imports detected: {violations}"


def test_public_facades_do_not_import_bootstrap() -> None:
    """Public facades should depend on composition, not the bootstrap shim."""
    root = Path(__file__).resolve().parents[1]
    modules = (
        root / "perceptimg" / "__init__.py",
        root / "perceptimg" / "api.py",
        root / "perceptimg" / "application" / "batch.py",
    )
    for module_path in modules:
        violations = _collect_forbidden_imports(
            module_path,
            forbidden_prefixes=("perceptimg.bootstrap",),
        )
        assert not violations, f"Bootstrap imports detected in {module_path}: {violations}"


def test_application_package_does_not_import_bootstrap() -> None:
    """Application package should not depend on the bootstrap shim."""
    root = Path(__file__).resolve().parents[1]
    application_dir = root / "perceptimg" / "application"
    violations: dict[str, list[tuple[int, str]]] = {}

    for path in sorted(application_dir.rglob("*.py")):
        file_violations = _collect_forbidden_imports(
            path,
            forbidden_prefixes=("perceptimg.bootstrap",),
        )
        if file_violations:
            violations[str(path)] = file_violations

    assert not violations, f"Bootstrap imports detected in application package: {violations}"


def test_application_package_exports_are_explicit() -> None:
    """Application package init should expose batch exports explicitly."""
    root = Path(__file__).resolve().parents[1]
    init_module = root / "perceptimg" / "application" / "__init__.py"
    tree = ast.parse(init_module.read_text(encoding="utf-8"), filename=str(init_module))

    has_getattr = any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body
    )
    eager_batch_imports: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "batch":
            eager_batch_imports.append((node.lineno, "from .batch import ..."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "perceptimg.application.batch":
                    eager_batch_imports.append((node.lineno, "import perceptimg.application.batch"))

    assert not has_getattr, "application/__init__.py should not use lazy __getattr__ exports"
    assert eager_batch_imports, "application/__init__.py should import batch exports explicitly"


def test_batch_module_does_not_import_private_composition() -> None:
    """Batch orchestration should use application-local runtime hooks, not composition."""
    root = Path(__file__).resolve().parents[1]
    batch_module = root / "perceptimg" / "application" / "batch.py"
    violations = _collect_forbidden_imports(
        batch_module,
        forbidden_prefixes=("perceptimg._composition",),
    )
    assert not violations, f"Private composition imports detected in batch module: {violations}"


def test_module_entrypoint_is_pure_orchestration() -> None:
    """The python -m entrypoint should only delegate to the CLI facade."""
    root = Path(__file__).resolve().parents[1]
    entrypoint_module = root / "perceptimg" / "__main__.py"
    violations = _collect_forbidden_imports(
        entrypoint_module,
        forbidden_prefixes=(
            "perceptimg.core",
            "perceptimg.application",
            "perceptimg.adapters",
        ),
    )
    assert not violations, f"Forbidden entrypoint imports detected: {violations}"


def test_presentation_layer_does_not_import_core() -> None:
    """Presentation helpers should avoid direct core imports."""
    root = Path(__file__).resolve().parents[1]
    presentation_module = root / "perceptimg" / "application" / "presentation.py"
    violations = _collect_forbidden_imports(
        presentation_module,
        forbidden_prefixes=("perceptimg.core",),
    )
    assert not violations, f"Forbidden presentation imports detected: {violations}"


def test_public_api_is_curated() -> None:
    """Top-level API should expose only the curated facade surface."""

    assert set(perceptimg.__all__) == set(api.PUBLIC_API)
    assert set(perceptimg.__all__).isdisjoint(set(api._LEGACY_PUBLIC_API))


def test_public_api_does_not_expose_internal_symbols() -> None:
    """Top-level package should not expose internal symbols by accident."""
    forbidden_symbols = (
        "CheckpointManager",
        "CoreBatchProcessorAdapter",
        "CoreCheckpointAdapter",
        "CoreRateLimiterAdapter",
        "CoreRetryAdapter",
        "MetricCalculator",
        "MultiRateLimiter",
        "RetryPolicy",
        "RateLimiter",
        "StrategyGenerator",
        "_ALLOWED_FORMATS",
    )
    for symbol in forbidden_symbols:
        assert not hasattr(perceptimg, symbol), f"forbidden public symbol leaked: {symbol}"


def test_top_level_package_import_is_pure() -> None:
    """Top-level package import should not bootstrap default wiring."""
    root = Path(__file__).resolve().parents[1]
    init_module = root / "perceptimg" / "__init__.py"
    tree = ast.parse(init_module.read_text(encoding="utf-8"), filename=str(init_module))

    forbidden_calls: list[tuple[int, str]] = []
    forbidden_imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "_composition" or node.module == "perceptimg._composition":
                forbidden_imports.append((node.lineno, f"from {node.module} import ..."))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "ensure_default_wiring":
                forbidden_calls.append((node.lineno, "ensure_default_wiring()"))

    assert (
        not forbidden_imports
    ), f"Top-level package imports composition directly: {forbidden_imports}"
    assert not forbidden_calls, f"Top-level package bootstraps on import: {forbidden_calls}"
