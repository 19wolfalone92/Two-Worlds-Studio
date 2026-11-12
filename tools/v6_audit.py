"""Repository-wide, offline/zero-GPU source inventory for TWO WORLDS STUDIO.

This does NOT perform destructive automatic pruning: even unimported CLI tools,
workflow-dispatched utilities and asset exporters may be intentional.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def inspect(root: Path = ROOT) -> dict:
    studio = root / "studio"
    source = sorted(p for p in studio.glob("*.py") if p.name != "__init__.py")
    all_py = sorted(root.rglob("*.py"))
    all_yml = sorted((root / ".github" / "workflows").glob("*.yml"))
    index = {"studio." + p.stem: p for p in source}
    imported: dict[str, set[str]] = defaultdict(set)
    parse_errors = []
    for p in all_py:
        # Never scan private media or bundled third party environments.
        if any(part in {".git", ".venv", "venv", "site-packages"} for part in p.parts):
            continue
        try:
            ast_tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append({"file": str(p.relative_to(root)), "error": type(exc).__name__})
            continue
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.module == "studio":
                    names += ["studio." + alias.name for alias in node.names]
            else:
                continue
            for name in names:
                for module in index:
                    if name == module or name.startswith(module + "."):
                        imported[module].add(p.relative_to(root).as_posix())
    text_mentions: dict[str, set[str]] = defaultdict(set)
    text_files = sorted(
        list((root / "docs").glob("*.md")) + list((root / ".github" / "workflows").glob("*.yml"))
        + list((root / "tools").glob("*.py")) + [root / "README.md"]
    )
    for path in text_files:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for module in index:
            if module in content:
                text_mentions[module].add(path.relative_to(root).as_posix())
    orphan_candidates = []
    for name in sorted(index):
        # Candidates are suggestions, never proof a file is dead.
        referenced = sorted((imported.get(name, set()) | text_mentions.get(name, set()))
                            - {index[name].relative_to(root).as_posix()})
        if not referenced:
            orphan_candidates.append(name)
    workflows = []
    for p in all_yml:
        data = p.read_text(encoding="utf-8", errors="replace")
        external = any(term in data for term in (
            "kaggle kernels push", "python -m studio.kaggle_api poll",
            "publish", "upload", "metricool", "tiktok",
        ))
        periodic = bool(re.search(r"(?m)^\s*schedule:\s*$", data))
        gpu = any(term in data for term in (
            "kaggle kernels push", "NvidiaTeslaT4", "CUDA_VISIBLE_DEVICES",
        ))
        workflows.append({"path": p.relative_to(root).as_posix(),
                          "schedule": periodic, "mentions_external_or_publishing": external,
                          "gpu_or_compute_trigger": gpu})
    return {
        "status": "CODE_INVENTORY_ONLY_NOT_MEDIA_QA",
        "source_modules": len(source), "python_files": len(all_py),
        "workflows": len(all_yml),
        "orphan_candidates_not_safe_to_autodelete": orphan_candidates,
        "orphan_count": len(orphan_candidates), "parse_errors": parse_errors,
        "workflows_inventory": workflows,
        "new_default": "studio.v6_production",
        "safe_removal_policy": "Only after caller/import + workflow + tests + backup check",
        "gpu_jobs_launched": 0, "media_published": False,
    }


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    a=p.parse_args(argv)
    report=inspect()
    pretty=json.dumps(report, ensure_ascii=False, indent=2)
    print(pretty)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(pretty + "\n", encoding="utf-8")


if __name__=="__main__":
    main()
