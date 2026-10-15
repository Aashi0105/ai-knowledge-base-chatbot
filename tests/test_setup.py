"""Automated tests for Phase 1 project setup, structure, and configuration validity."""

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_required_directories_exist():
    """Verify that all required Phase 1 directories exist."""
    required_dirs = [
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "chroma_db",
    ]
    for directory in required_dirs:
        assert directory.is_dir(), f"Expected directory missing: {directory}"


def test_gitkeep_files_exist():
    """Verify that .gitkeep files exist in data and chroma_db directories."""
    assert (PROJECT_ROOT / "data" / ".gitkeep").is_file(), "data/.gitkeep missing"
    assert (PROJECT_ROOT / "chroma_db" / ".gitkeep").is_file(), "chroma_db/.gitkeep missing"


def test_required_root_files_exist():
    """Verify that all root configuration and documentation files exist."""
    required_files = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / ".gitignore",
    ]
    for file_path in required_files:
        assert file_path.is_file(), f"Expected root file missing: {file_path}"


def test_src_module_placeholders_exist_and_syntax_valid():
    """Verify all placeholder modules exist in src/ and contain valid Python syntax."""
    src_modules = [
        "__init__.py",
        "document_processor.py",
        "chunker.py",
        "embeddings.py",
        "vector_store.py",
        "retriever.py",
        "prompt_builder.py",
        "llm_service.py",
        "chatbot.py",
    ]
    for module_name in src_modules:
        module_path = PROJECT_ROOT / "src" / module_name
        assert module_path.is_file(), f"Module missing in src/: {module_name}"
        with open(module_path, "r", encoding="utf-8") as f:
            code = f.read()
        # Verify valid Python syntax
        ast.parse(code, filename=str(module_path))


def test_system_prompt_configuration():
    """Verify config/system_prompt.json is valid JSON with all required keys."""
    config_file = PROJECT_ROOT / "config" / "system_prompt.json"
    assert config_file.is_file(), f"Config file missing: {config_file}"
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = ["role", "tone", "answer_format", "permitted_topics", "system_prompt"]
    for key in required_keys:
        assert key in data, f"Key '{key}' missing from system_prompt.json"
        assert data[key], f"Key '{key}' should not be empty"

    assert isinstance(data["permitted_topics"], list)
    assert len(data["permitted_topics"]) > 0


def test_env_files_variables():
    """Verify .env and .env.example contain required variable names."""
    expected_vars = {"GEMINI_API_KEY", "CHROMA_DB_PATH", "EMBEDDING_MODEL"}

    for filename in [".env", ".env.example"]:
        file_path = PROJECT_ROOT / filename
        assert file_path.is_file(), f"{filename} missing"
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        keys = set()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                keys.add(line.split("=", 1)[0].strip())
        missing = expected_vars - keys
        assert not missing, f"Missing variables {missing} in {filename}"


def test_gitignore_content():
    """Verify that .gitignore contains crucial exclusions."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore missing"
    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    crucial_patterns = [".venv", ".env", "__pycache__", "chroma_db/*", "!chroma_db/.gitkeep", "data/*", "!data/.gitkeep"]
    for pattern in crucial_patterns:
        assert pattern in content, f"Pattern '{pattern}' missing from .gitignore"


def test_app_syntax():
    """Verify that app.py has valid Python syntax."""
    app_file = PROJECT_ROOT / "app.py"
    assert app_file.is_file(), "app.py is missing"
    with open(app_file, "r", encoding="utf-8") as f:
        code = f.read()
    ast.parse(code, filename=str(app_file))
