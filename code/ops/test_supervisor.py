"""Fault drills for run_supervised.py and guardian.py with CPU dummy processes (runbook 4.4).

    python code/ops/test_supervisor.py --out runs/E24_session_X/audit/supervisor_tests.json

Every drill gets its own scratch session under the output's directory. Ten minutes become a few
seconds. Exits 1 if any drill fails; the JSON lists command, exit code, elapsed, stop reason.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUP = [sys.executable, f"{HERE}/run_supervised.py"]
GUARD = [sys.executable, f"{HERE}/guardian.py"]
PY = sys.executable

PROGRESS = "import json,sys,time\nfor i in range(int(sys.argv[2])):\n    open(sys.argv[1],'a').write(json.dumps({'update':i+1,'loss':0.5,'grad_norm':1.0})+'\\n'); time.sleep(float(sys.argv[3]))\n"
STALL = PROGRESS + "time.sleep(10000)\n"
IGNORE_TERM = ("import signal,subprocess,sys,time,json\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
               "subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10000)'])\n"
               "open(sys.argv[1],'a').write(json.dumps({'update':1,'loss':1.0})+'\\n')\ntime.sleep(10000)\n")
NAN = "import json,sys,time\nopen(sys.argv[1],'a').write(json.dumps({'update':1,'loss':float('nan')})+'\\n')\ntime.sleep(10000)\n"
PRED = "import json,sys\nf=open(sys.argv[1],'a')\nfor i in range(int(sys.argv[2])): f.write(json.dumps({'id':f'x{i}','output':''})+'\\n')\n"
CHECK_IDS = "import json,sys\nids={json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}\nsys.exit(0 if ids=={f'x{i}' for i in range(int(sys.argv[2]))} else 1)\n"


def scratch(root, name, deadline_in=3600):
    path = f"{root}/drill_{name}"
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    with open(f"{path}/session.json", "w") as f:
        json.dump({"deadlines": {"D_gpu": {"epoch": time.time() + deadline_in}}}, f)
    return path


def run(argv, timeout=120):
    t = time.monotonic()
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return p.returncode, round(time.monotonic() - t, 1), p.stdout.strip().splitlines()[-1:] or [""]


def job(session, name):
    with open(f"{session}/jobs/{name}.json") as f:
        return json.load(f)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def no_group_left(result):
    return result.get("group_cleaned") is True


def bystander():
    return subprocess.Popen([PY, "-c", "import time; time.sleep(10000)"], start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.path.dirname(os.path.realpath(a.out)) + "/drills"
    os.makedirs(root, exist_ok=True)
    fast = ["--poll", "0.5", "--grace", "2"]
    results = []

    def record(name, ok, **info):
        results.append({"drill": name, "passed": bool(ok), **info})
        print(f"{'PASS' if ok else 'FAIL'} {name} {info.get('reason', '')}", flush=True)

    # 1. normal progress, clean exit
    s = scratch(root, "normal")
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--progress", f"update:{s}/log.jsonl", "--stall", "3", "--max-seconds", "60", *fast, "--", PY, "-c", PROGRESS, f"{s}/log.jsonl", "6", "0.3"])
    r = job(s, "j")
    record("normal", code == 0 and r["status"] == "completed" and r["progress"] == 6 and no_group_left(r), exit=code, elapsed=el, reason=r["reason"])

    # 2. alive but no progress -> stall stop, heartbeat does not hide it
    s = scratch(root, "stall")
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--progress", f"update:{s}/log.jsonl", "--stall", "3", "--max-seconds", "60", *fast, "--", PY, "-c", STALL, f"{s}/log.jsonl", "2", "0.2"])
    r = job(s, "j")
    record("stall", code == 1 and r["status"] == "stopped" and r["reason"].startswith("stall") and el < 15 and no_group_left(r), exit=code, elapsed=el, reason=r["reason"])

    # 3. no first progress within init grace
    s = scratch(root, "init_grace")
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--progress", f"update:{s}/log.jsonl", "--init-grace", "3", "--max-seconds", "60", *fast, "--", PY, "-c", "import time; time.sleep(10000)"])
    r = job(s, "j")
    record("init_grace", r["status"] == "stopped" and "init grace" in r["reason"] and el < 15, exit=code, elapsed=el, reason=r["reason"])

    # 4. child and grandchild ignore TERM -> KILL the group only; bystander survives
    s = scratch(root, "ignore_term")
    b = bystander()
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--progress", f"update:{s}/log.jsonl", "--stall", "2", "--max-seconds", "60", *fast, "--", PY, "-c", IGNORE_TERM, f"{s}/log.jsonl"])
    r = job(s, "j")
    ok = r["status"] == "stopped" and no_group_left(r) and alive(b.pid)
    record("ignore_term", ok, exit=code, elapsed=el, reason=r["reason"], bystander_alive=alive(b.pid))
    b.kill()

    # 5. child exit 1 -> failed, not completed
    s = scratch(root, "exit1")
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--max-seconds", "30", *fast, "--", PY, "-c", "import sys; sys.exit(1)"])
    r = job(s, "j")
    record("exit1", code == 1 and r["status"] == "failed" and r["exit_code"] == 1, exit=code, elapsed=el, reason=r["reason"])

    # 6. partial predictions: exit 0 but missing ids -> done-check fails -> failed
    s = scratch(root, "partial")
    check = json.dumps([PY, "-c", CHECK_IDS, f"{s}/pred.jsonl", "10"])
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--progress", f"lines:{s}/pred.jsonl", "--max-seconds", "30", "--done-check", check, *fast, "--", PY, "-c", PRED, f"{s}/pred.jsonl", "7"])
    r = job(s, "j")
    ok_partial = r["status"] == "failed" and r["reason"] == "done-check failed"
    s2 = scratch(root, "complete")
    check = json.dumps([PY, "-c", CHECK_IDS, f"{s2}/pred.jsonl", "10"])
    code2, _, _ = run(SUP + ["--session", s2, "--name", "j", "--progress", f"lines:{s2}/pred.jsonl", "--max-seconds", "30", "--done-check", check, *fast, "--", PY, "-c", PRED, f"{s2}/pred.jsonl", "10"])
    record("partial_predictions", ok_partial and job(s2, "j")["status"] == "completed", exit=code, elapsed=el, reason=r["reason"])

    # 7. supervisor killed -> guardian stops the job's group; bystander survives
    s = scratch(root, "guardian")
    b = bystander()
    sup = subprocess.Popen(SUP + ["--session", s, "--name", "j", "--progress", f"update:{s}/log.jsonl", "--stall", "100", "--max-seconds", "100", *fast, "--", PY, "-c", STALL, f"{s}/log.jsonl", "1", "0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    running = json.load(open(f"{s}/jobs/j.running.json"))
    sup.kill()
    sup.wait()
    child_alive_after_sup_kill = alive(running["child_pid"])
    g = subprocess.Popen(GUARD + ["--session", s, "--until-epoch", str(time.time() + 3600), "--poll", "0.5", "--stale", "3", "--grace", "2", "--linger", "0"])
    time.sleep(6)
    g.terminate()
    g.wait()
    ok = child_alive_after_sup_kill and not group_alive_safe(running["pgid"]) and alive(b.pid)
    record("guardian_after_supervisor_death", ok, reason="guardian stopped orphaned group" if ok else "group still alive or bystander hit", bystander_alive=alive(b.pid))
    b.kill()

    # 8. PID reuse: running record with a wrong start time must not be signalled
    s = scratch(root, "pid_reuse")
    b = bystander()
    os.makedirs(f"{s}/jobs")
    with open(f"{s}/jobs/fake.running.json", "w") as f:
        json.dump({"name": "fake", "child_pid": b.pid, "pgid": b.pid, "child_start": -1, "supervisor_pid": 999999, "supervisor_start": -1, "heartbeat": f"{s}/none"}, f)
    g = subprocess.run(GUARD + ["--session", s, "--until-epoch", str(time.time() + 2), "--poll", "0.5", "--linger", "0"], timeout=30)
    events = [json.loads(l) for l in open(f"{s}/events.jsonl")]
    record("pid_reuse_not_killed", alive(b.pid) and any(e.get("event") == "guardian_skip" for e in events), reason="bystander alive, skip logged" if alive(b.pid) else "bystander killed")
    b.kill()

    # 9. deadline: refuse a job that cannot fit; stop a running job at the deadline
    s = scratch(root, "deadline", deadline_in=5)
    code, el, _ = run(SUP + ["--session", s, "--name", "late", "--max-seconds", "60", "--expect-seconds", "60", *fast, "--", PY, "-c", "print(1)"])
    refused = job(s, "late")["status"] == "refused"
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--max-seconds", "60", *fast, "--", PY, "-c", "import time; time.sleep(10000)"])
    r = job(s, "j")
    record("deadline", refused and r["status"] == "stopped" and r["reason"] == "session deadline" and el < 15, exit=code, elapsed=el, reason=r["reason"])

    # 10. low disk: refuse to start; stop when it drops while running
    s = scratch(root, "disk")
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--max-seconds", "30", "--fake-disk", "10,40", *fast, "--", PY, "-c", "print(1)"])
    refused = job(s, "j")["status"] == "refused"
    disk_file = f"{s}/disk.txt"
    open(disk_file, "w").write("20,40")
    k = subprocess.Popen(SUP + ["--session", s, "--name", "k", "--max-seconds", "30", "--fake-disk", "@" + disk_file, *fast, "--", PY, "-c", "import time; time.sleep(10000)"], stdout=subprocess.DEVNULL)
    time.sleep(2)
    open(disk_file, "w").write("5,40")
    k.wait(timeout=30)
    r = job(s, "k")
    record("low_disk", refused and r["status"] == "stopped" and r["reason"].startswith("low disk") and no_group_left(r), reason=f"start refused at ext4 10 GB; running job: {r['reason']}")

    # 11. GPU lock: a second GPU job is refused while the first holds the lock
    s = scratch(root, "gpu_lock")
    first = subprocess.Popen(SUP + ["--session", s, "--name", "a", "--gpu", "--max-seconds", "8", *fast, "--", PY, "-c", "import time; time.sleep(10000)"], stdout=subprocess.DEVNULL)
    time.sleep(2)
    code, el, _ = run(SUP + ["--session", s, "--name", "b", "--gpu", "--max-seconds", "8", *fast, "--", PY, "-c", "print(1)"])
    second = job(s, "b")
    first.wait(timeout=60)
    record("gpu_lock", second["status"] == "refused" and second["reason"].startswith("gpu_busy") and job(s, "a")["status"] == "stopped", reason=second["reason"])

    # 12. NaN loss -> immediate stop
    s = scratch(root, "nan")
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--progress", f"update:{s}/log.jsonl", "--stall", "60", "--max-seconds", "60", *fast, "--", PY, "-c", NAN, f"{s}/log.jsonl"])
    r = job(s, "j")
    record("nan_loss", r["status"] == "stopped" and "not finite" in r["reason"] and el < 15, exit=code, elapsed=el, reason=r["reason"])

    # 13. supervisor receives TERM -> cleans its child group
    s = scratch(root, "sup_term")
    sup = subprocess.Popen(SUP + ["--session", s, "--name", "j", "--max-seconds", "100", *fast, "--", PY, "-c", "import time; time.sleep(10000)"], stdout=subprocess.DEVNULL)
    time.sleep(2)
    sup.terminate()
    sup.wait(timeout=30)
    r = job(s, "j")
    record("supervisor_term", r["status"] == "stopped" and "signal" in r["reason"] and no_group_left(r), reason=r["reason"])

    # 14 (E25). done-check hangs -> stopped by --done-timeout, job failed, check process group gone
    s = scratch(root, "done_hang")
    hang = json.dumps([PY, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(10000)']); time.sleep(10000)"])
    code, el, _ = run(SUP + ["--session", s, "--name", "j", "--max-seconds", "30", "--done-check", hang, "--done-timeout", "2", *fast, "--", PY, "-c", "print(1)"])
    r = job(s, "j")
    leftover = subprocess.run(["pgrep", "-f", "^" + PY + " -c import time; time.sleep\\(10000\\)$"], stdout=subprocess.PIPE, text=True).stdout.split()
    record("done_check_timeout", code == 1 and r["status"] == "failed" and r["reason"] == "done-check timeout" and el < 15 and not leftover,
           exit=code, elapsed=el, reason=r["reason"], leftover_check_processes=leftover)

    # 15 (E25). project-wide GPU lock: a GPU job of another session is refused; bystander untouched
    parent = f"{root}/sessions_shared"
    shutil.rmtree(parent, ignore_errors=True)
    os.makedirs(parent)
    s1, s2 = scratch(parent, "one"), scratch(parent, "two")
    b = bystander()
    first = subprocess.Popen(SUP + ["--session", s1, "--name", "a", "--gpu", "--max-seconds", "8", *fast, "--", PY, "-c", "import time; time.sleep(10000)"], stdout=subprocess.DEVNULL)
    time.sleep(2)
    code, el, _ = run(SUP + ["--session", s2, "--name", "b", "--gpu", "--max-seconds", "8", *fast, "--", PY, "-c", "print(1)"])
    second = job(s2, "b")
    first.wait(timeout=60)
    record("global_gpu_lock", second["status"] == "refused" and ".gpu_global.lock" in second["reason"] and job(s1, "a")["status"] == "stopped" and alive(b.pid),
           reason=second["reason"], bystander_alive=alive(b.pid))
    b.kill()

    passed = all(x["passed"] for x in results)
    with open(a.out, "x") as f:
        json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "passed": passed, "drills": results}, f, indent=1)
    print("ALL PASSED" if passed else "SOME FAILED")
    sys.exit(0 if passed else 1)


def group_alive_safe(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


if __name__ == "__main__":
    main()
