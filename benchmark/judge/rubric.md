# Judge rubric

You are scoring an agent's answer against a ground-truth answer for a question
about the `dex` codebase. The agent may have phrased things differently — judge
*meaning*, not exact wording.

Inputs you receive:
- QUESTION: what was asked
- GROUND_TRUTH: the canonical answer (treat as authoritative)
- ANSWER: the agent's final response

Score on a 0–2 scale:

- **2 — correct**: The answer is substantively correct, identifies the right
  files/functions/concepts, and is not misleading. Minor phrasing differences,
  extra correct detail, or missing minor specifics are fine. Different but
  equivalent file:line references are fine if they point at the same symbol.

- **1 — partial**: The answer gets the main idea but is missing important
  specifics (e.g. names the right package but wrong function; identifies the
  mechanism but wrong file; correct concept but a key detail is wrong).

- **0 — wrong**: The answer is incorrect, hallucinated, contradicts the
  ground truth, or fails to answer the question.

Be strict on **hallucinations** — if the answer cites a file or function that
does not exist or attributes behavior to the wrong place, that is at most a 1,
and usually a 0.

Be lenient on **format** — agents may answer in prose, bullets, or code; do
not penalize style.

Respond with a strict JSON object, no preamble:

```json
{"score": 0|1|2, "reason": "<one short sentence>"}
```
