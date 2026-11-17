"""One-line E25 status for the conversation monitor: time, controller step, last train updates."""
import json, os, sys, time
S = sys.argv[1]
parts = [time.strftime("%H:%M:%S")]
try:
    parts.append("current=" + json.load(open(f"{S}/status.json"))["current"])
except Exception as e:
    parts.append(f"current=? ({e})")
for m in ("C0", "C1"):
    p = f"{S}/{m}/train_log.jsonl"
    if os.path.exists(p):
        lines = [l for l in open(p) if l.strip()]
        if lines:
            r = json.loads(lines[-1])
            parts.append(f"{m} update {r['update']} loss {r['loss']}")
print("STATUS", " | ".join(parts), flush=True)
