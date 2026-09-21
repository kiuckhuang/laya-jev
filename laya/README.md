# Laya sidecar — a local, Jev-dialect decision endpoint

This directory turns [Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0,
322–421M parameters) into a **drop-in replacement for the TypeSafe Jev API**, so
`jev-ultrafast` runs its whole decision loop without any cloud call.

Everything here was verified live on 2026-09-21:

| Host | Checkpoint | Latency / decision | Notes |
| --- | --- | --- | --- |
| WSL laptop (12-core CPU) | multilingual 322M | 29–40 ms | uncalibrated — misses cancellation threats |
| `hhnode-185` (Quadro RTX 6000) | **English 421M** | **26–35 ms**, p95 ≈ p50 | temperature-calibrated; recommended |
| TypeSafe cloud Jev | — | ~250 ms round trip | best zero-shot accuracy |

Honest limitation, reproduced in testing: zero-shot **element-selection** is weak
(flat distributions, DONE/WAIT bias — phrasing experiments did not fix it). Laya is
strong out of the box for clear-cut routing and yes/no guards; for element tables,
plan to fine-tune (upstream ships a Kaggle notebook) or cascade to cloud Jev on low
confidence. The dialect transport itself is fully verified: every answer passes the
repo's `validate_choice()`.

## What was added vs upstream

- `jev_ultrafast/model.py` — `TYPESAFE_ENDPOINT` env override (3 lines; default
  unchanged, all 31 upstream tests pass).
- `laya/laya_jev_sidecar.py` — FastAPI sidecar speaking the repo's exact
  `/v1/systemone` dialect: parses `state`/`questions`, splits large target questions
  into ≤`LAYA_CHUNK`-option chunks inside one forward pass, merges + renormalizes
  probabilities, returns `{choice, probabilities, confidence}` per question.
- `laya/dialect_test.py` — conformance test: runs the repo's `model.choose()` and
  `validate_choice()` against the sidecar on a 31-element fixture page.
- `laya/smoke_test.py` — model self-test (perception/action inversion, numbers,
  calibration, guard, multi-question latency).
- `laya/setup_remote.sh` — one-shot setup for a GPU/CPU server over SSH.

## Quickstart

```bash
# 1. Set up any server with SSH access (GPU optional, CPU works):
laya/setup_remote.sh hhnode-185          # default host alias; runs over ssh

# 2. Tunnel a local port to the sidecar (127.0.0.1:8001 on the server):
ssh -N -L 7185:127.0.0.1:8001 hhnode-185 &

# 3. Point the repo at it (.env):
#    TYPESAFE_ENDPOINT=http://127.0.0.1:7185/v1/systemone
#    TYPESAFE_API_KEY=local-laya        # value unused by the sidecar
#    TEXT_MODEL_API_KEY=...             # TYPE_TEXT still needs a text LLM (Ollama works)

# 4. Verify without any browser:
uv run python laya/dialect_test.py
```

First sidecar start downloads ~2.3 GB of weights (English root bundle) and loads in
25–55 s. The model is loaded once and shared; a `threading.Lock` serializes forward
passes.

## Tunables (env on the server)

| Var | Default | Meaning |
| --- | --- | --- |
| `LAYA_PORT` | `8001` | sidecar port (binds `127.0.0.1` only) |
| `LAYA_MODEL` | `convaiinnovations/laya` | root = English; add `subfolder` handling in code for `multilingual` / `typed-decisions` |
| `LAYA_STATE_CHARS` | `1000` | page-text character budget (English checkpoint reads 512 tokens **total**) |
| `LAYA_OPTION_CHARS` | `140` | per-option description cap (options are matched like textual entailment) |
| `LAYA_CHUNK` | `10` | options per chunk; upstream reports degradation past ~20 options |

## Remote-host quirks learned the hard way

- If `ssh` fails with `Bad owner or permissions on /etc/ssh/ssh_config.d/...`,
  bypass the system config: `ssh -F ~/.ssh/config ...`.
- To restart the sidecar over ssh, **do not** `pkill -f laya_jev_sidecar` from a
  shell whose own command line contains that pattern — it kills itself. Use the
  bracket trick: `pkill -f "laya_jev_sideca[r]"`.
- If your `~/.ssh/config` already forwards local port 7185 (common vLLM
  convention), reuse it — a second `-L 7185` fails with exit 255.
- The first call at a new batch shape compiles kernels (~1.7 s); the sidecar warms
  up at startup — keep the warm-up.

## Upstream workflow

The upstream `browser-use/jev-ultrafast` stays configured as the `upstream` remote:

```bash
git fetch upstream && git rebase upstream/main   # or merge
```

## Stopping / restarting on the server

```bash
ssh hhnode-185 'pkill -f "laya_jev_sideca[r]"'                       # stop
ssh hhnode-185 'setsid nohup ~/laya-venv/bin/python ~/laya_jev_sidecar.py \
  > ~/laya-sidecar.log 2>&1 < /dev/null & sleep 40; \
  curl -s -X POST http://127.0.0.1:8001/health -H "Content-Type: application/json" -d "{}"'
```

## Provenance

- Laya model: Nandakishor M, Convai Innovations — Apache-2.0
  ([code](https://github.com/NandhaKishorM/laya),
  [weights](https://huggingface.co/convaiinnovations/laya),
  [SalesRLAgent paper](https://arxiv.org/abs/2503.23303),
  [routing paper](https://arxiv.org/abs/2510.01237)).
- Sidecar, benchmark framing and this integration: this fork. Not affiliated with
  TypeSafe AI; "Jev" is named only to describe the dialect.
