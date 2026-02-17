"""Packaging profile tests for OSS SDK core vs web extras."""

from pathlib import Path
import tomllib


def test_web_stack_is_optional_and_not_in_core_dependencies():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    core_deps = list(data["project"]["dependencies"])
    web_extra = list(data["project"]["optional-dependencies"]["web"])

    assert all("fastapi" not in dep for dep in core_deps)
    assert all("uvicorn" not in dep for dep in core_deps)
    assert all("jinja2" not in dep for dep in core_deps)

    assert any("fastapi" in dep for dep in web_extra)
    assert any("uvicorn" in dep for dep in web_extra)
    assert any("jinja2" in dep for dep in web_extra)
