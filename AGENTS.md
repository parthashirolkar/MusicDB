# AGENTS.md

**MusicDB:** Music similarity search system using MERT-v1-95M embeddings and ChromaDB vector store.

This file contains technical guidelines for agentic coding working in this repository.

## Commands

### Dependency Management
- `uv sync` - Install/sync dependencies from pyproject.toml
- `uv run python <script>.py` - Run a Python script in the uv environment

### Linting & Formatting
- `uv run ruff check` - Check all files for lint errors
- `uv run ruff check <path>` - Check specific file or directory
- `uv run ruff check --fix` - Auto-fix lint issues
- `uv run ruff check --fix --unsafe-fixes` - Apply unsafe fixes

### Testing
- No test framework is currently configured

## Code Style

### Type Annotations
- Use modern `|` union syntax (e.g., `str | None`, `Path | str`) NOT `Optional[str]`
- Use `list[str]`, `dict[str, int]` instead of `List[str]`, `Dict[str, int]`
- Always type function parameters and return values in services
- Use `| None` for optional types, not `Optional[T]`
