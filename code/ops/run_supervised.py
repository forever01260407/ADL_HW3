"""Run one job under supervision: hard time limits, real-progress checks, clean process-group stop.

    python code/ops/run_supervised.py --session runs/E24_session_X --name b0_train --gpu \
        --progress update:runs/.../train_log.jsonl --stall 600 --init-grace 900 --max-seconds 3600 \
        --expect-seconds 2400 -- python code/train.py ...

Standard library only. What it guarantees, independent of whether anyone is watching:

- The child runs in its own session/process group (start_new_session) from an argv list, with
  stdout/stderr going straight to a file, so the supervisor never blocks on a pipe.
- GPU jobs hold an exclusive flock on <session>/gpu.lock and on a project-wide lock shared by all
  sessions (<runs>/.gpu_global.lock) for their whole life; a second GPU job, from this or another
  session, cannot start (it exits with reason gpu_busy). The kernel drops a lock when its holder dies.
- Every --poll seconds: child alive?, real progress (see below), heartbeat, disk, deadlines.
- Stop reasons: stall (no progress for --stall seconds after the first progress, or no first
  progress within --init-grace), max_seconds, session deadline (--deadline-key in session.json,
  default D_gpu), NaN/Inf loss, low disk, TERM/INT to the supervisor. Stop = TERM to the child's
  process group, wait --grace seconds, then KILL; the group is checked to be empty afterwards.
- A job is refused before starting if --expect-seconds would not fit before the deadline.
- Result: <session>/jobs/<name>.json (status completed only when exit code 0, the child's process
  group is empty afterwards, and, if given, the --done-check command exits 0 within --done-timeout
  seconds (capped by the deadline; on timeout its own process group is killed)); every transition is appended to <session>/events.jsonl.
  While running, <session>/jobs/<name>.running.json names the child pid/pgid and their start
  times so code/ops/guardian.py can stop the group if this supervisor dies or hangs.

Progress kinds:
  update:<train_log.jsonl>  last complete JSON line's "update" strictly increases; loss and
                            grad_norm must be finite.
  lines:<file>              number of complete, distinct "id" values in a JSONL file increases.
  rows:<file>               number of complete non-empty lines increases (E25: judge output).
  none                      only the time limits apply (for jobs that write nothing until the end).
"""
import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time

EXT4_MIN_GB, E_MIN_GB = 6.0, 8.0  # stop the job below these; start refused below START_* (runbook 4.3)
START_EXT4_GB, START_E_GB = 12.0, 15.0
START_DISK_WAIT = int(os.environ.get("HW3_START_DISK_WAIT", "90"))  # seconds of re-checks before a low-disk refusal


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_atomic(path, data):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def event(session, **fields):
    fields = {"time": now_iso(), "epoch": round(time.time(), 1), **fields}
    with open(f"{session}/events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(fields, ensure_ascii=False) + "\n")
        f.flush()


def start_time(pid):
    """Kernel start time of a process (clock ticks since boot), to detect PID reuse."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def free_gb(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def disk_state(fake):
    if fake:  # "EXT4,E" or "@file" holding that text (re-read every poll, for tests)
        text = open(fake[1:]).read() if fake.startswith("@") else fake
        ext4, e = (float(x) for x in text.split(","))
    else:
        ext4 = free_gb("/")
        e = free_gb("/mnt/e") if os.path.isdir("/mnt/e") else ext4
    return round(ext4, 1), round(e, 1)


def read_progress(kind):
    """(value, problem). value is a comparable number; problem is a string when the output is invalid."""
    if kind == "none":
        return None, None
    name, _, path = kind.partition(":")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        raw = f.read()
    lines = raw.split(b"\n")
    complete = lines[:-1]  # the last element is either b"" or a half-written line
    if name == "update":
        for line in reversed(complete):
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError:
                    return None, "unreadable train log line"
                for key in ("loss", "grad_norm"):
                    if key in row and not math.isfinite(float(row[key])):
                        return row.get("update"), f"{key} is not finite"
                return row.get("update"), None
        return None, None
    if name == "rows":  # complete non-empty lines (judge output rows carry "key", not "id")
        return sum(1 for line in complete if line.strip()), None
    if name == "lines":
        ids = set()
        for line in complete:
            if line.strip():
                try:
                    ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    return len(ids), "unreadable prediction line"
        return len(ids), None
    raise SystemExit(f"unknown progress kind {kind}")


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_group(child, grace, session, name):
    pgid = child.pid  # start_new_session: the child leads its own group
    if group_alive(pgid):
        os.killpg(pgid, signal.SIGTERM)
        end = time.monotonic() + grace
        while time.monotonic() < end and group_alive(pgid):
            child.poll()
            time.sleep(0.5)
        if group_alive(pgid):
            os.killpg(pgid, signal.SIGKILL)
            event(session, job=name, event="kill", pgid=pgid)
            end = time.monotonic() + 15
            while time.monotonic() < end and group_alive(pgid):
                child.poll()
                time.sleep(0.5)
    child.poll()
    return not group_alive(pgid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", action="store_true", help="take the session GPU lock")
    ap.add_argument("--progress", default="none")
    ap.add_argument("--stall", type=float, default=600)
    ap.add_argument("--init-grace", type=float, default=900)
    ap.add_argument("--max-seconds", type=float, required=True)
    ap.add_argument("--expect-seconds", type=float, default=0)
    ap.add_argument("--deadline-key", default="D_gpu")
    ap.add_argument("--deadline-epoch", type=float, help="override (tests)")
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--grace", type=float, default=30)
    ap.add_argument("--done-check", help="shell-free command (JSON list) that must exit 0 for status completed")
    ap.add_argument("--done-timeout", type=float, default=120, help="seconds for the done-check (also capped by the deadline)")
    ap.add_argument("--global-lock", help="project-wide GPU lock shared by all sessions (default: <session parent>/.gpu_global.lock)")
    ap.add_argument("--fake-disk", help="EXT4_GB,E_GB (tests)")
    ap.add_argument("--log", help="child stdout/stderr file (default <session>/logs/<name>.log)")
    ap.add_argument("argv", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    argv = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
    if not argv:
        raise SystemExit("no command")
    session = os.path.realpath(a.session)
    for sub in ("jobs", "logs"):
        os.makedirs(f"{session}/{sub}", exist_ok=True)
    result_path = f"{session}/jobs/{a.name}.json"
    running_path = f"{session}/jobs/{a.name}.running.json"
    if os.path.exists(result_path) or os.path.exists(running_path):
        raise SystemExit(f"job {a.name} already exists in {session}/jobs")
    if a.deadline_epoch:
        deadline = a.deadline_epoch
    else:
        with open(f"{session}/session.json", encoding="utf-8") as f:
            deadline = json.load(f)["deadlines"][a.deadline_key]["epoch"]
    log_path = a.log or f"{session}/logs/{a.name}.log"
    began_wall, began = time.time(), time.monotonic()
    result = {"name": a.name, "argv": argv, "progress": a.progress, "deadline_epoch": deadline, "log": log_path,
              "started": now_iso(), "supervisor_pid": os.getpid()}

    def finish(status, reason, **extra):
        result.update(status=status, reason=reason, ended=now_iso(), elapsed=round(time.monotonic() - began, 1), **extra)
        write_atomic(result_path, result)
        if os.path.exists(running_path):
            os.remove(running_path)
        event(session, job=a.name, event="end", status=status, reason=reason, elapsed=result["elapsed"])
        print(json.dumps({k: result[k] for k in ("name", "status", "reason", "elapsed")}, ensure_ascii=False), flush=True)
        sys.exit(0 if status == "completed" else 1)

    ext4, e = disk_state(a.fake_disk)
    waited = 0  # E28 amendment: E: free dips ~12 GB for <20 s right after a vLLM job ends; re-check before refusing
    while (ext4 < START_EXT4_GB or e < START_E_GB) and waited < START_DISK_WAIT:
        time.sleep(5)
        waited += 5
        ext4, e = disk_state(a.fake_disk)
    if ext4 < START_EXT4_GB or e < START_E_GB:
        finish("refused", f"low disk before start (ext4 {ext4} GB, E: {e} GB, re-checked for {waited}s)")
    if began_wall + a.expect_seconds > deadline:
        finish("refused", f"expected {a.expect_seconds:.0f}s does not fit before deadline ({deadline - began_wall:.0f}s left)")
    locks = []
    if a.gpu:
        for path in (f"{session}/gpu.lock", a.global_lock or f"{os.path.dirname(session)}/.gpu_global.lock"):
            lock = open(path, "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                finish("refused", f"gpu_busy: another supervised GPU job holds {path}")
            locks.append(lock)

    stop = {"signal": None}

    def on_signal(signum, _frame):
        stop["signal"] = signum
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    with open(log_path, "ab") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    result.update(child_pid=child.pid, pgid=child.pid)
    write_atomic(running_path, {**result, "child_start": start_time(child.pid), "supervisor_start": start_time(os.getpid()),
                                "heartbeat": f"{session}/jobs/{a.name}.heartbeat.json", "deadline_epoch": deadline})
    event(session, job=a.name, event="start", pid=child.pid, argv=argv)

    last_value, last_progress, first_seen = None, time.monotonic(), False
    reason = None
    while True:
        code = child.poll()
        value, problem = read_progress(a.progress)
        t = time.monotonic()
        if value is not None and (last_value is None or value > last_value):
            if not first_seen:
                event(session, job=a.name, event="first_progress", value=value)
            last_value, last_progress, first_seen = value, t, True
        ext4, e = disk_state(a.fake_disk)
        write_atomic(f"{session}/jobs/{a.name}.heartbeat.json",
                     {"time": now_iso(), "epoch": time.time(), "progress": last_value, "since_progress": round(t - last_progress, 1),
                      "ext4_gb": ext4, "e_gb": e, "deadline_left": round(deadline - time.time())})
        if code is not None:
            break
        if stop["signal"]:
            reason = f"supervisor got signal {stop['signal']}"
        elif problem and "finite" in problem:
            reason = problem
        elif time.time() >= deadline:
            reason = "session deadline"
        elif t - began >= a.max_seconds:
            reason = "max_seconds"
        elif a.progress != "none" and not first_seen and t - began >= a.init_grace:
            reason = "no first progress within init grace"
        elif a.progress != "none" and first_seen and t - last_progress >= a.stall:
            reason = f"stall: no progress for {t - last_progress:.0f}s"
        elif ext4 < EXT4_MIN_GB or e < E_MIN_GB:
            reason = f"low disk (ext4 {ext4} GB, E: {e} GB)"
        if reason:
            event(session, job=a.name, event="stopping", reason=reason)
            clean = stop_group(child, a.grace, session, a.name)
            finish("stopped", reason, exit_code=child.returncode, group_cleaned=clean, progress=last_value)
        time.sleep(a.poll)

    clean = stop_group(child, a.grace, session, a.name)  # reap leftovers of an exited leader (e.g. vLLM workers)
    value, problem = read_progress(a.progress)
    extra = {"exit_code": code, "group_cleaned": clean, "progress": value}
    if code != 0:
        finish("failed", f"exit code {code}", **extra)
    if problem:
        finish("failed", f"output problem: {problem}", **extra)
    if not clean:
        finish("failed", "process group not empty after exit (leftover workers)", **extra)
    if a.done_check:
        limit = max(1.0, min(a.done_timeout, deadline - time.time()))
        check = subprocess.Popen(json.loads(a.done_check), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            out, _ = check.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(check.pid, signal.SIGKILL)
            check.communicate()
            extra["done_check"] = {"exit_code": None, "timeout_seconds": round(limit, 1)}
            finish("failed", "done-check timeout", **extra)
        extra["done_check"] = {"exit_code": check.returncode, "output": out[-2000:]}
        if check.returncode != 0:
            finish("failed", "done-check failed", **extra)
    finish("completed", "exit code 0" + (" and done-check passed" if a.done_check else ""), **extra)


if __name__ == "__main__":
    main()
