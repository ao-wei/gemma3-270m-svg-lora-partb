# logo_detailed_prompt — (detailed prompt → SVG logo) training pairs

Supervised pairs for teaching a small model to draw an SVG logo from a **detailed
visual prompt**. Each target SVG was produced by Claude Sonnet from that prompt.

## Files

| File | Rows | Contents |
|---|---|---|
| `train.jsonl` | 138 | training pairs |
| `valid.jsonl` | 17 | validation pairs |

Each line is one chat-format example:

```json
{"messages": [
  {"role": "system",    "content": "<SVG-designer instructions>"},
  {"role": "user",      "content": "<detailed visual prompt>"},
  {"role": "assistant", "content": "<complete <svg>…</svg>>"}
]}
```

- **Input** = the detailed prompt (`user`).
- **Target** = one complete `<svg …>…</svg>` document (`assistant`), `viewBox="0 0 256 256"`.
- Train loss should be masked to the assistant (SVG) tokens only.

## Notes

- These are **detailed-prompt → Sonnet-SVG** pairs only. Raw-query augmentation rows
  and the held-out **test set are intentionally excluded** (the test set is kept
  private for grading).
