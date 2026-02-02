# GitHub Copilot Instructions

## Tool use

When reading from a file, prefer using the file tool to access its contents and do not resort to command like `cat`, `less`, `nl`, or `head`.
If there are no file tools available for what you want to do (e.g. complex filters that the provided tools do not support), feel free to use shell commands.

## Chat behavior

### LaTeX / KaTeX formatting notes (chat only)

These rules apply **only** to math rendered in the chat UI with me (GitHub Copilot using GPT-5.1). They do **not** apply to LaTeX source files or notebooks where a full LaTeX engine is used.

- Avoid using `\\text{}` with names that contain `*` or escaped underscores when writing equations in the chat. These often fail to render in the chat UI.
- Prefer simple variable names inside math blocks in the chat, e.g. use `S_\\text{after}` instead of `\\text{*_logprobs\_sum}`.
- When equations fail to render in the chat, simplify them (remove `\\text{}` around complex identifiers and avoid exotic symbols) and re-send.
