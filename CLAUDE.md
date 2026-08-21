# Project rules for Claude Code

- Never open a PR or merge to `main` without the user explicitly agreeing first.
- Ask before: major architectural decisions, setting/changing any threshold or
  numeric parameter, installing or uninstalling any package.
- Log every methodological choice (thresholds, masking logic, model/library
  versions, why an approach was picked) in `METHODS.md` as it's made, not
  after the fact.
- Keep the repo properly versioned: commit logical units of work with clear
  messages as the work progresses, don't batch everything into one commit.
- `.claude/` is git-ignored — never commit Claude Code session/config files.
