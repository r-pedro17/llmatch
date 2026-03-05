---
name: git
description: >
  Perform git operations for the current repository. This skill MUST be used
  for ANY git-related request — even a single word like "commit", "push",
  "status", "diff", or "log" is enough to trigger it. Do not reach for Bash
  git commands directly; always use this skill first. Trigger phrases include
  but are not limited to: "commit", "make a commit", "do a commit", "git commit",
  "save my changes", "stage and commit", "push", "git push", "git status",
  "show status", "what changed", "git diff", "show diff", "create a branch",
  "new branch", "switch branch", "git log", "show log", "show recent commits".
  Also use proactively before pushing or when changes need to be saved.
---

# Git Skill

Perform git operations for the current repository. This skill handles committing, branching, status checks, and diffs following the project's git conventions.

## Usage
Invoked via `/git [optional args]` — args describe the desired git action (e.g. `/git commit`, `/git status`, `/git branch feature-x`).

## Subagent rule

- **Run inline** when context usage is below 50% AND the current session already has context about what changed (e.g. you just edited files or the user described the changes).
- **Delegate to a `general-purpose` subagent** in either of these cases:
  - The user asks for a git operation out of the blue with no prior file edits in the session, OR
  - Context usage is at or above 50% (visible in the status line).

  Pass the full commit flow instructions below to the subagent and report its result back to the user.

## Safety rules (always enforced)
- Never force-push to main/master without explicit user confirmation
- Never use `--no-verify` unless the user explicitly requests it
- Never amend published commits; create new ones instead
- Stage specific files by name — avoid `git add -A` or `git add .`
- Never commit `.env`, credentials, or secrets

## Pre-flight check (always run first)
Before any git operation, verify a repository exists:
```bash
git status 2>/dev/null || git init
```
If `git init` runs, inform the user and proceed.

## Commit flow
1. Run `git status` + `git diff HEAD` + `git log --oneline -5` in parallel
2. Draft a concise commit message focused on **why**, not what
3. Stage relevant files by name
4. Commit with co-author trailer:
   ```
   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```
5. Run `git status` to confirm success

## Common operations
- **Status:** `git status` — show working tree state
- **Diff:** `git diff` (unstaged) or `git diff --staged` (staged)
- **Log:** `git log --oneline -10` — recent commits
- **Branch:** `git checkout -b <name>` — create and switch
- **Push:** confirm with user before pushing to remote

## Commit message format
```
<type>: <short summary under 70 chars>

<optional body — only if non-obvious>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```
Types: `feat`, `fix`, `style`, `refactor`, `docs`, `chore`

## Output
Report the git action taken and resulting status. If anything fails or is ambiguous, ask the user before proceeding.
