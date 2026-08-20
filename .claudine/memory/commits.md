# Commit Orchestration — Lessons Learned

## Verifying commits under concurrency

- Never verify "your" commit with `git log -1` / `git log -1 --stat` when multiple agents commit to the same worktree concurrently — HEAD advances between commands and you will see another agent's commit. Capture the commit hash from the commit command's own output (e.g. `[main fdd198a]`) and verify with `git show <hash> --stat`.
- Expect the working tree and index to change mid-flight: developers may stage/unstage/modify files (e.g. `.gitignore` went from unstaged-modification to staged during one orchestrated run). This is normal; never treat it as corruption and never "fix" it with resets.

## Committing staged renames with `--only`

- To commit a staged pure rename (git shows `R old -> new`), pass BOTH the old (deleted) path and the new path as pathspecs: `git commit --only -F - -- old/path new/path`. Git then records the change as a 100%-similarity rename. The old paths no longer exist in the working tree — that is fine; `--only` commits their staged deletions.

## Cheap pre-commit guards

- Before committing, run `git diff --cached -- <paths>` (add `-M100%` when renames are involved) and check it matches the expected delta — a cheap guard against committing the wrong or adjacent changes.
