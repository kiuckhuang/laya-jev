"""Laya sidecar: a drop-in, Jev-dialect /v1/systemone endpoint served from remote-185.

Accepts the same request shape jev-ultrafast sends to api.typesafe.ai
(model/state/questions with choice|score|noul questions) and answers with
{answers: {<question>: {choice, probabilities, confidence}}, model, usage}.

Run on the GPU node:
    ~/laya-venv/bin/pip install fastapi uvicorn
    nohup ~/laya-venv/bin/python ~/laya_jev_sidecar.py > ~/laya-sidecar.log 2>&1 &
Then, from any machine with SSH access:
    ssh -N -L 7185:127.0.0.1:8001 remote-185
and point the client at http://127.0.0.1:7185/v1/systemone
"""
import math
import os
import threading
import time

import uvicorn
from fastapi import FastAPI

import laya

MODEL_ID = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")  # root = English, calibrated
STATE_CHAR_BUDGET = int(os.environ.get("LAYA_STATE_CHARS", "1000"))
OPTION_CHARS = int(os.environ.get("LAYA_OPTION_CHARS", "140"))
CHUNK = int(os.environ.get("LAYA_CHUNK", "10"))
PORT = int(os.environ.get("LAYA_PORT", "8001"))

print(f"sidecar: loading {MODEL_ID} (English) ...", flush=True)
AGENT = laya.load(MODEL_ID)
print("sidecar: model loaded", flush=True)
LOCK = threading.Lock()
APP = FastAPI()


def _clip(value, limit):
    return " ".join(str(value or "").split())[:limit]


def _confidence(probs):
    n = len(probs)
    if n <= 1:
        return 1.0
    entropy = -sum(p * math.log2(p) for p in probs.values() if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log2(n)))


def build_state(body):
    """Flatten the TypeSafe state into a short, front-loaded Laya state string."""
    root = body.get("state") or {}
    page = root.get("page") or {}
    goal = ""
    for question in (body.get("questions") or {}).values():
        instruction = question.get("instructions") or {}
        if instruction.get("goal"):
            goal = instruction["goal"]
            break
    parts = [f"Goal: {_clip(goal, 300)}"]
    if page.get("url"):
        parts.append(f"URL: {_clip(page['url'], 110)}")
    if page.get("title"):
        parts.append(f"Title: {_clip(page['title'], 110)}")
    actions = root.get("recent_actions") or []
    if actions:
        recent = "; ".join(
            f"{a.get('kind', '')} {a.get('action', '')}".strip() for a in actions[-3:]
        )
        parts.append(f"Recent: {_clip(recent, 180)}")
    if page.get("text"):
        parts.append(f"Page: {_clip(page['text'], STATE_CHAR_BUDGET)}")
    return "\n".join(parts)


def question_to_laya_chunks(qname, question):
    """Split one TypeSafe question into Laya questions of <=CHUNK options.

    Returns (laya_questions_dict, merge_plan) where merge_plan knows the full
    id list so probabilities can be renormalized across chunks.
    """
    qtype = question.get("type", "choice")
    instruction = question.get("instructions") or {}
    criteria = question.get("criteria") or {}
    if qtype == "noul":
        text = instruction.get("criteria") or instruction.get("instructions") or ""
        return (
            {qname: {"type": "noul", "instructions": _clip(text, 200) or "Answer true or false."}},
            {"kind": "noul", "ids": None},
        )
    options = {}
    for key, value in criteria.items():
        if isinstance(value, dict):
            label = value.get("element") or value.get("label") or key
            current = value.get("current_value") or ""
            text = f"{label} | value: {current}" if current else str(label)
        else:
            text = str(value)
        options[key] = _clip(text, OPTION_CHARS)
    ids = list(options.keys())
    laya_questions, plan_ids = {}, []
    op = instruction.get("operation") or "operation"
    for start in range(0, len(ids), CHUNK):
        part = ids[start : start + CHUNK]
        lid = f"{qname}::c{start // CHUNK}"
        laya_questions[lid] = {
            "type": "choice",
            "instructions": _clip(f"Best target for the next {op}? Choose the option matching the goal.", 150),
            "criteria": {k: options[k] for k in part},
        }
        plan_ids.extend(part)
    return laya_questions, {"kind": "choice", "ids": plan_ids}


@APP.post("/health")
def health():
    return {"ok": True, "model": MODEL_ID}


@APP.post("/v1/systemone")
async def systemone(body: dict):
    started = time.time()
    state = build_state(body)
    laya_questions, plan = {}, {}
    for qname, question in (body.get("questions") or {}).items():
        chunk_questions, merge_plan = question_to_laya_chunks(qname, question)
        laya_questions.update(chunk_questions)
        plan[qname] = merge_plan
    with LOCK:  # one forward pass at a time on this GPU
        result = AGENT.predict(state, laya_questions)
    answers = {}
    for qname, merge_plan in plan.items():
        if merge_plan["kind"] == "noul":
            answer = result["answers"][qname]
            answers[qname] = {
                "noul": float(answer.get("noul", 0.5)),
                "confidence": float(answer.get("confidence", 0.5)),
            }
            continue
        merged = {}
        for lid in [k for k in result["answers"] if k.startswith(qname + "::")]:
            for key, p in result["answers"][lid]["probabilities"].items():
                merged[key] = max(merged.get(key, 0.0), float(p))
        ids = merge_plan["ids"]
        total = sum(merged.get(i, 0.0) for i in ids) or 1.0
        probs = {i: merged.get(i, 0.0) / total for i in ids}
        best = max(ids, key=lambda i: probs[i])
        answers[qname] = {
            "choice": best,
            "probabilities": {i: round(p, 6) for i, p in probs.items()},
            "confidence": round(_confidence(probs), 6),
        }
    return {
        "answers": answers,
        "model": f"laya-421m-english@remote-185",
        "usage": {
            "input_tokens": result.get("usage", {}).get("input_tokens", 0),
            "output_tokens": 0,
        },
        "latency_ms": round((time.time() - started) * 1000),
    }


if __name__ == "__main__":
    # warm up the two batch shapes we care about (single + chunked)
    AGENT.predict("warmup", {"op": {"type": "choice", "instructions": "warm", "criteria": {"a": "first", "b": "second"}}})
    AGENT.predict("warmup", {"op": {"type": "choice", "instructions": "warm", "criteria": {f"o{i}": f"option number {i}" for i in range(10)}}})
    print("sidecar: warm, serving", flush=True)
    uvicorn.run(APP, host="127.0.0.1", port=PORT, log_level="warning")
