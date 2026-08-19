# Pre-Commit Verification Skill

## Purpose
Run this checklist at the end of every session before any `git commit` or `git push` to ensure the codebase is clean, all tests pass, and deployment assets are in sync.

## Required Checks (run in this order)

### 1. Run the full test suite
```bash
uv run pytest tests/ -q --tb=short -m "not integration"
```
**Expected**: all tests pass. If any test fails, fix before proceeding.

### 2. Run execution-specific tests
```bash
uv run pytest tests/test_execution_*.py -v --tb=short
```
**Expected**: all execution tests pass (currently ~64 as of 2026-07-28).

### 3. Run the linter
```bash
uv run ruff check src/ tests/
```
**Expected**: `All checks passed!` (may have warnings about `F401` in `__init__.py` — that's expected).

### 4. Run the formatter
```bash
uv run ruff format --check src/ tests/
```
**Expected**: all files already formatted. If any files need reformatting, run `uv run ruff format src/ tests/`.

### 5. Check for secrets in tracked files
```bash
grep -rn 'GMAIL_USER\|GMAIL_APP_PASSWORD\|RECIPIENT_EMAIL\|APCA_API_KEY\|APCA_API_SECRET\|UNUSUAL_WHALES' src/ --include='*.py' 2>/dev/null
```
**Expected**: No output. Secrets should only live in `.env` (gitignored) and be referenced via `${VAR}` in `config.yaml`.

### 6. Verify execution dependencies are declared
```bash
grep -c 'alpaca-py\|tenacity' pyproject.toml
```
**Expected**: at least 2 matches.

### 7. Verify execution config section is valid
```bash
uv run python -c "from src.execution import ExecutionConfig; c = ExecutionConfig(); print('OK')"
```
**Expected**: prints `OK`.

### 8. Review the diff
```bash
git diff --stat
git diff
```
Look for:
- No `.env` or secrets in the diff
- No large binary files
- No unintended changes to config files
- No commented-out code

### 9. Read the commit message
Ensure it is a single concise line matching repo style (no periods, no emojis, imperative mood).

## Quick one-liner
If all checks above pass individually, run this combined smoke check:
```bash
uv run pytest tests/test_execution_*.py tests/test_config.py tests/test_decision.py tests/test_models.py tests/test_options_strategy.py tests/test_parser.py tests/test_risk.py tests/test_status.py tests/test_strategies.py tests/test_mcp_client.py -q --tb=short && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```
