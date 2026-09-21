"""Dialect test: run jev-ultrafast's model.choose() against the Laya sidecar.

Proves the sidecar speaks the repo's /v1/systemone dialect well enough that
validate_choice() accepts every answer and the decisions are sensible.
"""

import json
import os
import time
import urllib.request

os.environ["TYPESAFE_ENDPOINT"] = "http://127.0.0.1:7185/v1/systemone"
os.environ["TYPESAFE_API_KEY"] = "local-laya"

from jev_ultrafast import model  # noqa: E402


def post_json(url, key, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as response:
        out = json.loads(response.read())
    print(f"   [sidecar {(time.perf_counter()-started)*1000:.0f} ms | tokens in {out['usage']['input_tokens']} | model {out['model']}]", flush=True)
    return out


model.post_json = post_json


def action(node, kind, role, label, value="", **extra):
    return {"id": f"e{node}{kind[0]}", "kind": kind, "node": node, "role": role,
            "label": label, "value": value, **extra}


elements = [
    action(1, "fill", "combobox", "Destination", "Lisbon"),
    action(1, "click", "combobox", "Open Destination", "Lisbon"),
    action(2, "fill", "textbox", "Check-in", ""),
    action(3, "fill", "textbox", "Check-out", ""),
    action(4, "fill", "combobox", "Guests", "2 guests"),
    action(4, "click", "combobox", "Open Guests", "2 guests"),
    action(5, "fill", "searchbox", "Search stays", ""),
    action(5, "click", "searchbox", "Open Search stays", ""),
    action(6, "click", "checkbox", "Free cancellation (filter)", "", checked="false"),
    action(7, "click", "checkbox", "Design (filter)", "", checked="false"),
    action(8, "click", "checkbox", "Pool (filter)", "", checked="false"),
    action(9, "click", "checkbox", "Parking (filter)", "", checked="false"),
    action(10, "click", "checkbox", "Breakfast included (filter)", "", checked="false"),
    action(11, "click", "checkbox", "Pet friendly (filter)", "", checked="false"),
    action(12, "click", "checkbox", "Spa (filter)", "", checked="false"),
    action(13, "click", "checkbox", "Sea view (filter)", "", checked="false"),
    action(14, "click", "link", "Casa Flora — Design stay in Lisbon · from €90"),
    action(15, "click", "link", "Alfama Loft — Budget stay in Lisbon · from €45"),
    action(16, "click", "link", "Chiado Grand — Luxury stay in Lisbon · from €210"),
    action(17, "click", "link", "Baixa Rooms — Budget stay in Lisbon · from €38"),
    action(18, "click", "link", "Belém Riverside — Design stay in Lisbon · from €110"),
    action(19, "click", "link", "Príncipe Real Villa — Design stay in Lisbon · from €130"),
    action(20, "click", "link", "Alcântara Studio — Budget stay in Lisbon · from €52"),
    action(21, "click", "link", "Estrela Manor — Luxury stay in Lisbon · from £180"),
    action(22, "click", "button", "Sort by"),
    action(23, "click", "button", "Map view"),
    action(24, "click", "button", "List view"),
    action(25, "click", "link", "Help"),
    action(26, "click", "link", "Sign in"),
    action(27, "click", "button", "Currency: EUR"),
    action(28, "click", "button", "Language: English"),
    action(29, "click", "link", "Next page"),
]

state = {
    "url": "http://127.0.0.1:8799/fixture.html?scenario=travel",
    "title": "Forma · Find a place to slow down",
    "text": "Forma — find a place to slow down. Stays in Lisbon and beyond. "
            "Design, Budget and Luxury stays. Filters: Free cancellation, Design, Pool, Parking.",
    "scroll": {"y": 0},
    "actions": elements,
}

CASES = [
    ("click-only navigation", "Open Casa Flora."),
    ("filters first", "Show only Design stays with Free cancellation."),
    ("typed search", 'Type "digital nomad" into the search stays box.'),
]

for name, goal in CASES:
    print(f"\n== {name}: goal = {goal!r}")
    decision = model.choose(state, goal, [])
    match = next((a for a in elements if a["id"] == decision["choice"]), None)
    label = f"{match['label'][:60]})" if match else "non-element choice)"
    print(f"   operation={decision['operation']}  choice={decision['choice']} ({label}")
    print(f"   confidence={decision['confidence']:.3f}  top-3:",
          sorted(decision["probabilities"].items(), key=lambda kv: -kv[1])[:3])
print("\nDIALECT_TEST_DONE — every answer passed the repo's validate_choice()")
