"""Verification script for Phase 1 environment setup and configuration."""

import importlib
import json
import os
import sys
from pathlib import Path


def print_status(check_name: str, passed: bool, message: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    symbol = "[OK]" if passed else "[FAIL]"
    details = f" - {message}" if message else ""
    print(f"[{status}] {symbol} {check_name}{details}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 65)
    print("Phase 1: Environment & Setup Verification")
    print("=" * 65)

    all_passed = True
    project_root = Path(__file__).resolve().parent.parent

    # 1. Python Version
    py_version = sys.version.replace("\n", " ")
    print(f"Active Python Executable: {sys.executable}")
    print(f"Active Python Version: {py_version}")
    version_ok = sys.version_info >= (3, 10)
    print_status(
        "Python Version Check",
        version_ok,
        f"Version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    if not version_ok:
        all_passed = False

    # 2. Virtual Environment Check
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    venv_in_project = ".venv" in sys.executable or ".venv" in sys.prefix
    is_project_venv = in_venv and venv_in_project
    print_status(
        "Virtual Environment Active",
        is_project_venv,
        f"Prefix: {sys.prefix}",
    )
    if not is_project_venv:
        all_passed = False

    # 3. Dependencies Import Verification
    required_modules = [
        ("streamlit", "streamlit"),
        ("google.genai", "google-genai"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("fitz", "PyMuPDF"),
        ("dotenv", "python-dotenv"),
        ("pytest", "pytest"),
    ]

    print("\n--- Verifying Dependencies ---")
    for mod_name, pkg_name in required_modules:
        try:
            importlib.import_module(mod_name)
            print_status(f"Import: {pkg_name} ({mod_name})", True)
        except Exception as e:
            print_status(f"Import: {pkg_name} ({mod_name})", False, str(e))
            all_passed = False

    # 4. JSON Configuration Verification
    print("\n--- Verifying Configuration ---")
    config_path = project_root / "config" / "system_prompt.json"
    if not config_path.exists():
        print_status("System Prompt Config File Exists", False, f"Missing: {config_path}")
        all_passed = False
    else:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            expected_keys = {"role", "tone", "answer_format", "permitted_topics", "system_prompt"}
            missing_keys = expected_keys - set(config_data.keys())
            if missing_keys:
                print_status(
                    "System Prompt Config Format",
                    False,
                    f"Missing keys: {missing_keys}",
                )
                all_passed = False
            else:
                print_status("System Prompt Config Format", True, "Valid JSON with all required keys")
        except Exception as e:
            print_status("System Prompt Config Format", False, f"Invalid JSON: {e}")
            all_passed = False

    # 5. Required Project Directories
    print("\n--- Verifying Project Structure ---")
    required_dirs = ["config", "src", "tests", "scripts", "data", "chroma_db"]
    for d in required_dirs:
        dir_path = project_root / d
        exists = dir_path.is_dir()
        print_status(f"Directory: {d}/", exists)
        if not exists:
            all_passed = False

    # 6. Environment File Verification (.env)
    print("\n--- Verifying Environment Variables File (.env) ---")
    env_path = project_root / ".env"
    if not env_path.exists():
        print_status(".env File Exists", False, f"Missing: {env_path}")
        all_passed = False
    else:
        with open(env_path, "r", encoding="utf-8") as f:
            env_lines = f.readlines()
        env_keys = set()
        for line in env_lines:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key = line.split("=", 1)[0].strip()
                env_keys.add(key)

        expected_vars = {"GEMINI_API_KEY", "CHROMA_DB_PATH", "EMBEDDING_MODEL"}
        missing_vars = expected_vars - env_keys
        if missing_vars:
            print_status(
                ".env Variable Presence",
                False,
                f"Missing variables: {missing_vars} (Secrets are masked)",
            )
            all_passed = False
        else:
            print_status(
                ".env Variable Presence",
                True,
                f"All required keys found: {sorted(list(expected_vars))} (Values masked)",
            )

    print("=" * 65)
    if all_passed:
        print("OVERALL RESULT: ALL CHECKS PASSED ✅")
        print("=" * 65)
        return 0
    else:
        print("OVERALL RESULT: VERIFICATION FAILED ❌")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    sys.exit(main())
