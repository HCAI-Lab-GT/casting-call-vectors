# Code Style and Conventions

## General Principles
- Lightweight, performant, clean architectural code
- Strict separation of concerns across modules
- Synchronous, deterministic operations preferred
- Files should not exceed 150 lines (250 for website components)
- Avoid over-engineering; only make directly requested changes

## Python Style
- **Version**: Python 3.12+
- **Formatting**: Black-compatible via `ruff format`
- **Line length**: 100 characters (configured in pyproject.toml)
- **Imports**: Sorted by ruff (stdlib, third party, local)
- **Type hints**: Use native Python types (`list[str]`, not `List[str]`)
- **Naming**: 
  - `snake_case` for variables and functions
  - `PascalCase` for classes

## Code Organization
- `src/` layout with `src/pvx/` as main package
- Modular design with single responsibility per component
- Reuse existing functions; no redundant code
- Prefer `pathlib.Path` over `os.path`

## Documentation
- Avoid inline comments; code should be self-explanatory
- Use Google-style docstrings when documentation is required
- Docstrings present in utility classes (see RIASECHelpers as example)

## Error Handling
- Robust error handling without over-engineering
- Specific exceptions with context messages
- Use `logging` module instead of `print`

## Testing
- Framework: pytest with fixtures
- Table-driven tests with parameterization
- Use `unittest.mock` for external dependencies
- Test directories exist at `tests/unit/`, `tests/integration/`, `tests/mocks/`

## Ruff Configuration (from pyproject.toml)
```toml
[tool.ruff]
line-length = 100
extend-select = ["B", "I", "C4", "RET"]
```

## Type Checking
- ty type checker configured with relaxed rules for third-party libraries
- Strict rules for internal code: `unresolved-import`, `unresolved-reference`, `invalid-assignment`
