"""Second line of defence: stop supervised jobs whose supervisor died or hung, and everything at the end.

    python code/ops/guardian.py --session runs/E24_session_X [--until-key D_gpu] [--poll 30] [--stale 180]

Runs on its own (nohup), independent of any conversation. Every --poll seconds it reads
<session>/jobs/*.running.json (written by run_supervised.py) and, for each job:

- if the supervisor process is gone (or its PID now belongs to another process, checked by kernel
  start time) or its heartbeat is older than --stale seconds, it stops the job's process group;
- once the session deadline (--until-key, default D_gpu) has passed, it stops every listed group.

A group is only signalled when the recorded child PID still has the recorded start time, so a
reused PID is never killed. Stop = TERM, wait --grace, KILL. Everything is logged to
<session>/events.jsonl. The guardian exits by itself --linger seconds after the deadline, so it
never runs forever. Its own heartbeat is <session>/guardian.heartbeat.json.
"""
import argparse
import glob
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_supervised import event, group_alive, start_time, write_atomic  # noqa: E402


def identity_ok(pid, recorded):
    return recorded is not None and start_time(pid) == recorded


def stop(session, job, why, grace):
    pid, pgid = job["child_pid"], job["pgid"]
    if not identity_ok(pid, job.get("child_start")):
        event(session, job=job["name"], event="guardian_skip", reason=f"{why}; child pid {pid} gone or reused")
        return
    event(session, job=job["name"], event="guardian_stop", reason=why, pgid=pgid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.time() + grace
    while time.time() < end and group_alive(pgid):
        time.sleep(0.5)
    if group_alive(pgid):
        os.killpg(pgid, signal.SIGKILL)
        event(session, job=job["name"], event="guardian_kill", pgid=pgid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--until-key", default="D_gpu")
    ap.add_argument("--until-epoch", type=float, help="override (tests)")
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--stale", type=float, default=180)
    ap.add_argument("--grace", type=float, default=30)
    ap.add_argument("--linger", type=float, default=600)
    a = ap.parse_args()
    session = os.path.realpath(a.session)
    if a.until_epoch:
        until = a.until_epoch
    else:
        with open(f"{session}/session.json", encoding="utf-8") as f:
            until = json.load(f)["deadlines"][a.until_key]["epoch"]
    event(session, job="guardian", event="start", pid=os.getpid(), until=until)
    handled = set()
    while True:
        t = time.time()
        jobs = []
        for path in glob.glob(f"{session}/jobs/*.running.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    jobs.append(json.load(f))
            except (OSError, ValueError):
                continue
        for job in jobs:
            key = (job["name"], job["child_pid"])
            if key in handled:
                continue
            why = None
            if t >= until:
                why = "session deadline reached (guardian)"
            elif not identity_ok(job["supervisor_pid"], job.get("supervisor_start")):
                why = "supervisor is gone"
            else:
                try:
                    with open(job["heartbeat"], encoding="utf-8") as f:
                        beat = json.load(f)["epoch"]
                except (OSError, ValueError, KeyError):
                    beat = os.path.getmtime(f"{session}/jobs/{job['name']}.running.json")
                if t - beat > a.stale:
                    why = f"supervisor heartbeat {t - beat:.0f}s old"
            if why:
                stop(session, job, why, a.grace)
                handled.add(key)
        write_atomic(f"{session}/guardian.heartbeat.json", {"epoch": t, "pid": os.getpid(), "jobs": len(jobs), "until": until})
        if t >= until + a.linger:
            event(session, job="guardian", event="exit", reason="past deadline + linger")
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
