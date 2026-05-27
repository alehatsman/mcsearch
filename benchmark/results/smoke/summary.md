# Benchmark summary

Results dir: `/home/aleh/projects/dex/benchmark/results/smoke`


```
{
  "model": "sonnet",
  "replicates": 1,
  "started": "2026-05-27T10:24:52Z",
  "git_sha": "045b8719391d3d549b69095e3b6e77e2402441c4",
  "claude_version": "2.1.152"
}
```


## Overall (per mode)

| Mode | N | Avg score | Correct (2/N) | Avg in tok | Avg out tok | Avg cache-read | Avg tool calls | Avg wall (s) | Avg $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dex | 1 | 2.0 | 1/1 | 7 | 214 | 36205 | 0 | 7 | 0.087 |
| native | 1 | 2.0 | 1/1 | 4 | 117 | 30553 | 0 | 5 | 0.031 |

## Per band × mode

| Band | Mode | N | Avg score | Avg in tok | Avg out tok | Avg tool calls | Avg wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | dex | 1 | 2.0 | 7 | 214 | 0 | 7 |
| L1 | native | 1 | 2.0 | 4 | 117 | 0 | 5 |

## Per-question (dex vs native, averaged across reps)

| QID | Band | dex score | nat score | dex tok | nat tok | dex calls | nat calls |
|---|---|---:|---:|---:|---:|---:|---:|
| L1-01 | L1 | 2.0 | 2.0 | 221 | 121 | 0.0 | 0.0 |
