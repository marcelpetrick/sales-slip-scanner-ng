# Working Agreement for Agents

## Commit Messages

- Always use **conventional commit format**: `<type>: <short imperative subject>`
  - Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `style`, `perf`
- The subject line must be short and imperative (e.g. "add image resize step", not "added" or "adding").
- The body must explain **what** and **why** — not how the code was generated.
- Include meaningful detail in the body: changed behaviour, motivation, affected components.
- **Never mention Claude, any other AI assistant, or any AI tooling** in commit messages.

## Local Pipeline

A local pipeline script exists in the repository root and **must pass successfully before every commit**.

Run it with:
```
./pipeline.sh
```

Do not commit if the pipeline exits with a non-zero status. Fix all failures first, then re-run to confirm green before committing.

## General

- Report back when a task is fully done.
- Prefer small, focused commits over large omnibus ones.
