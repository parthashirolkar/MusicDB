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
- To add tests: Set up pytest in pyproject.toml with test discovery

## Code Style

### Type Annotations
- Use modern `|` union syntax (e.g., `str | None`, `Path | str`) NOT `Optional[str]`
- Use `list[str]`, `dict[str, int]` instead of `List[str]`, `Dict[str, int]`
- Always type function parameters and return values in services
- Use `| None` for optional types, not `Optional[T]`

### Imports
- Order: stdlib → third-party → local imports (separated by blank lines)
- Local imports use absolute paths from project root (e.g., `from core.config import get_settings`)
- Add `sys.path.insert(0, str(Path(__file__).parent))` in CLI scripts to fix imports
- Example:
  ```python
  from pathlib import Path
  import chromadb
  
  from core.config import get_settings
  from core.exceptions import DatabaseError
  ```

### Classes & Architecture
- Service layer pattern: All business logic in `services/` directory
- Configuration: Use `get_settings()` from `core.config` for all settings
- Custom exceptions: Raise from `core.exceptions`, never raise generic `Exception`
- Lazy initialization: Use `_attr: Type | None = None` pattern with private getters
- Example:
  ```python
  class Service:
      def __init__(self):
          self._client: chromadb.Client | None = None
      
      def _get_client(self) -> chromadb.Client:
          if self._client is None:
              self._client = chromadb.PersistentClient(path=str(self.db_path))
          return self._client
  ```

### Error Handling
- Always catch and re-raise as custom exceptions from `core.exceptions`
- Use descriptive error messages with context
- Log errors before raising
- Example:
  ```python
  try:
      result = operation()
  except Exception as e:
      logger.error(f"Operation failed: {e}")
      raise DatabaseError(f"Failed to process: {e}")
  ```

### Logging
- Use `loguru` logger (not `print` statements)
- Import: `from loguru import logger`
- Levels: `debug()` for details, `info()` for normal operations, `warning()` for recoverable issues, `error()` for failures

### Docstrings
- Use Google-style docstrings for all public methods and classes
- Include Args, Returns, Raises sections
- Keep descriptions concise
- Example:
  ```python
  def process_song(self, song_id: str) -> dict:
      """Process a song and generate embedding.
      
      Args:
          song_id: Unique identifier for the song
          
      Returns:
          Dictionary with status and embedding
          
      Raises:
          AudioProcessingError: If audio processing fails
      """
  ```

### Async Patterns
- Use `asyncio.run()` in CLI commands to run async functions
- Use `async/await` for I/O operations (downloads, file processing)
- Use `asyncio.Semaphore` to limit concurrent operations
- Use `loop.run_in_executor()` for blocking calls in async context

### Pydantic Settings
- Use `pydantic-settings` `BaseSettings` for configuration
- Use `model_config = SettingsConfigDict(env_prefix="PREFIX_")` for env var mapping
- Use `@field_validator` for validation
- Use `Field(default=...)` for defaults
- Call `get_settings()` to access settings (not direct instantiation)

### Path Handling
- Use `pathlib.Path` for all file paths, never string paths
- Convert user input: `Path(file_path)` immediately
- Ensure directories exist: `path.mkdir(parents=True, exist_ok=True)`

### Click CLI Patterns
- Use `@click.group()` for main command
- Use `@click.pass_context` to pass context
- Define async functions inside command handlers, then `asyncio.run()`

### Type Aliases & Forward References
- Use string quotes for forward references when needed
- Import `TYPE_CHECKING` for type-check-only imports
