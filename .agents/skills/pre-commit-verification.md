# Pre-Commit Verification Skill

## Purpose
Run this checklist at the end of every session before any `git commit` or `git push` to ensure the codebase is clean, all tests pass, and deployment assets are in sync.

## Required Checks (run in this order)

### 1. Run the full test suite
```bash
uv run pytest tests/ -q --tb=short
```
**Expected**: `127 passed` (or the current count). If any test fails, fix before proceeding.

### 2. Run the linter
```bash
uv run ruff check src/ tests/
```
**Expected**: `All checks passed!`

### 3. Run the formatter
```bash
uv run ruff format --check src/ tests/
```
**Expected**: `45 files already formatted` (or similar count). If any files need reformatting, run `uv run ruff format src/ tests/`.

### 4. Check for secrets in tracked files
```bash
grep -rn 'GMAIL_USER\|GMAIL_APP_PASSWORD\|RECIPIENT_EMAIL\|APCA_API_KEY\|UNUSUAL_WHALES' src/ --include='*.py' 2>/dev/null
```
**Expected**: No output. Secrets should only live in `.env` (gitignored) and be referenced via `${VAR}` in `config.yaml`.

### 5. Verify site is up-to-date
The site at `site/index.html` should reflect the current state of the system:
- Check that the example JSON matches the actual LLM output format
- Check that the email preview table matches the actual rendered HTML
- Check that the feature descriptions are accurate
- Run a quick dry-run to get fresh output:
  ```bash
  uv run python -m src.main --dry-run --correlation-id precommit-check
  ```

### 6. Verify GitHub Actions are consistent
Check that `.github/workflows/ci.yml` and `.github/workflows/deploy.yml` reference the correct:
- Python version (3.12)
- Test count (no hardcoded numbers that drift)
- Ruff commands

### 7. Review the diff
```bash
git diff --stat
git diff
```
Look for:
- No `.env` or secrets in the diff
- No large binary files
- No unintended changes to config files

### 8. Read the commit message
Ensure it is a single concise line matching repo style (no periods, no emojis, imperative mood).

## Quick one-liner
If all checks above pass individually, run this combined smoke check:
```bash
uv run pytest tests/ -q --tb=short && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```
