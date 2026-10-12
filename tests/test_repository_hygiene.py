from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_contains_code_and_instructions_only() -> None:
    forbidden_suffixes = {
        ".bib", ".csv", ".jsonl", ".log", ".pdf", ".pt", ".pth",
        ".safetensors", ".tex", ".zip",
    }
    forbidden_parts = {"__pycache__", ".pytest_cache", "outputs", "results"}
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and not forbidden_parts.intersection(path.parts)
    ]
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert files
    assert not any(path.suffix.lower() in forbidden_suffixes for path in files)
    assert all(f"{name}/" in gitignore for name in forbidden_parts)
