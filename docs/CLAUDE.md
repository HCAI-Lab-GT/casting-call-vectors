# Peresonality Vectors
- @AGENTS.md

## [STARTUP]
- Activate Serena project "persona-vectors" before any work.

## Running Commands

**Ad-hoc scripts:** Only `/tmp/claude-execution-allowed/persona-vectors/` is approved for ad-hoc scripts. Write scripts there and run with `uv run /tmp/claude-execution-allowed/persona-vectors/<script-name>`. Scripts can be Python or bash depending on the task. For bash scripts, make them executable first with `chmod +x`.

**Bash operations:**

Complex bash syntax is hard for Claude Code to permission correctly. Keep commands simple.

Simple operations are fine: `|`, `||`, `&&`, `>` redirects.

For bulk operations on multiple files, use xargs:
- Plain: `ls *.md | xargs wc -l`
- With placeholder: `ls *.md | xargs -I{} head -1 {}`

For string interpolation (`$()`, backticks, `${}`), heredocs, loops, or advanced xargs flags, write a script in `/tmp/claude-execution-allowed/persona-vectors/` instead.

**Patterns:**
- File creation: Write tool, not `cat << 'EOF' > file`
- Env vars: `export VAR=val && command`, not `VAR=val command` or `env VAR=val command`
- Bulk operations: `ls *.md | xargs wc -l`, not `for f in *.md; do cmd "$f"; done`
- Parallel/batched xargs: script, not `xargs -P4` or `xargs -L1`
- Per-item shell logic: script, not `xargs sh -c '...'`

If a command that should be allowed is denied, or if project structure changes significantly, ask about running `/mats:permissions` to update settings.
