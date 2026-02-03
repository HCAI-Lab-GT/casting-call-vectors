# Task Completion Checklist

When completing a development task on this project, perform the following steps:

## 1. Code Quality Checks
```bash
# Run linter
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

## 2. Testing
```bash
# Run tests
pytest

# With coverage if relevant
python -m pytest --cov=src/pvx tests/
```

## 3. Verification
- Ensure no new ruff warnings/errors
- Verify code follows project conventions (see code_style.md)
- Check that files don't exceed 150 lines
- Confirm type hints are present for new functions

## 4. Documentation
- Update README.md if adding new CLI commands or features
- Update configs/*.yaml if adding new presets
- Log changes in LOGBOOK.md with timestamp and context

## 5. Git Workflow
After completing the task, ask the user if they want to:
1. Commit changes (requires manual staging first)
2. Commit and push
3. Neither

## Notes
- Never stage files automatically (git add)
- Use jj instead of git for operations
- Follow existing commit message style from repo history
- Anonymity is important: never mention names in commits
