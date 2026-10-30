from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "town_model"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _files(directory: str) -> list[Path]:
    return sorted((ROOT / directory).rglob("*.py"))


def test_models_cannot_depend_on_presentation_experiment_or_training_layers():
    forbidden = ("town_model.dashboard", "town_model.experiments", "town_model.training", "town_model.cli", "town_model.inspection", "streamlit", "matplotlib")
    violations = []
    for path in _files("models"):
        for imported in _imports(path):
            if imported.startswith(forbidden):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
    assert not violations, "forbidden model dependencies:\n" + "\n".join(violations)


def test_vision_adapter_cannot_depend_on_ontology_training_or_presentation():
    forbidden = ("town_model.models", "town_model.training", "town_model.experiments", "town_model.dashboard", "town_model.cli", "streamlit", "matplotlib")
    violations = []
    for path in _files("vision"):
        for imported in _imports(path):
            if imported.startswith(forbidden):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
    assert not violations, "forbidden vision dependencies:\n" + "\n".join(violations)


def test_training_and_dataset_preparation_do_not_import_presentation_or_torchvision():
    presentation = ("town_model.dashboard", "town_model.cli", "streamlit", "matplotlib")
    violations = []
    for path in _files("training"):
        for imported in _imports(path):
            if imported.startswith(presentation):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
    for relative in ("dataset/preparation.py", "dataset/generator.py", "dataset/generator_worker.py"):
        path = ROOT / relative
        for imported in _imports(path):
            if imported == "torchvision" or imported.startswith("torchvision."):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
    assert not violations, "forbidden dependency directions:\n" + "\n".join(violations)


def test_new_layer_import_graph_has_no_cycles():
    source_files = [
        path for path in ROOT.rglob("*.py")
        if any(part in {"dataset", "vision", "models", "training", "experiments", "inspection", "dashboard"} for part in path.relative_to(ROOT).parts)
    ]
    module_for_path: dict[Path, str] = {}
    for path in source_files:
        relative = path.relative_to(ROOT).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        module_for_path[path] = "town_model" + ("." + ".".join(parts) if parts else "")
    known = set(module_for_path.values())
    graph: dict[str, set[str]] = {module: set() for module in known}
    for path, module in module_for_path.items():
        package = module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base_parts = package.split(".")
                    base_parts = base_parts[: len(base_parts) - node.level + 1]
                    if node.module:
                        base_parts.extend(node.module.split("."))
                    candidates = {".".join(base_parts)}
                elif node.module:
                    candidates = {node.module}
                else:
                    candidates = set()
            else:
                continue
            graph[module].update(candidate for candidate in candidates if candidate in known and candidate != module)
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            cycle = visiting[visiting.index(module) :] + [module]
            raise AssertionError("circular dependency: " + " -> ".join(cycle))
        if module in visited:
            return
        visiting.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)
