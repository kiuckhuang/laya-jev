#!/usr/bin/env bash
# One-shot Laya sidecar setup on a remote GPU/CPU server.
# Usage: laya/setup_remote.sh [ssh-alias]   (default: hhnode-185)
#
# Notes:
# - Uses `python3 -m venv --system-site-packages` so an existing CUDA torch on
#   the server is reused; otherwise install torch first (CPU build is enough).
# - If plain `ssh` fails with "Bad owner or permissions on /etc/ssh/ssh_config.d/..."
#   export SSH_OPTS="-F $HOME/.ssh/config" before running this script.
set -euo pipefail

HOST_ALIAS="${1:-hhnode-185}"
SSH_CMD=(ssh -o BatchMode=yes ${SSH_OPTS:-} "$HOST_ALIAS")
SCP_CMD=(scp -o BatchMode=yes ${SSH_OPTS:-})

echo "== 1/4 venv + packages on $HOST_ALIAS =="
"${SSH_CMD[@]}" 'python3 -m venv --system-site-packages ~/laya-venv 2>/dev/null || true
~/laya-venv/bin/pip install -q --disable-pip-version-check laya fastapi uvicorn
~/laya-venv/bin/python -c "import laya, torch; print(\"laya OK | torch\", torch.__version__, \"| cuda\", torch.cuda.is_available())"'

echo "== 2/4 deploy sidecar =="
"${SCP_CMD[@]}" "$(dirname "$0")/laya_jev_sidecar.py" "$HOST_ALIAS:~/laya_jev_sidecar.py"

echo "== 3/4 launch (first start downloads ~2.3 GB of weights) =="
"${SSH_CMD[@]}" 'pkill -f "laya_jev_sideca[r]" 2>/dev/null || true; sleep 1
setsid nohup ~/laya-venv/bin/python ~/laya_jev_sidecar.py > ~/laya-sidecar.log 2>&1 < /dev/null &
for i in $(seq 1 24); do
  sleep 5
  if curl -s -X POST http://127.0.0.1:8001/health -H "Content-Type: application/json" -d "{}" 2>/dev/null | grep -q ok; then
    echo "sidecar healthy after ~$((i*5)) s"; exit 0
  fi
done
echo "sidecar did not become healthy; check ~/laya-sidecar.log"; exit 1'

echo "== 4/4 done =="
cat <<'NEXT'

Sidecar is listening on 127.0.0.1:8001 on the server.

From your machine, open a tunnel:
    ssh -N -L 7185:127.0.0.1:8001 HOST_ALIAS &

Then point jev-ultrafast at it (.env):
    TYPESAFE_ENDPOINT=http://127.0.0.1:7185/v1/systemone
    TYPESAFE_API_KEY=local-laya

Conformance check (from the repo checkout):
    uv run python laya/dialect_test.py
NEXT
