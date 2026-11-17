"""E25 controller: the whole run after the plan is frozen, with no further instructions needed.

    python code/ops/phase_e25.py --session runs/E25_session_X [--dry-plan PLAN.json]

Order (plan section 12): E23 baseline -> C0 train -> verify -> C0 eval -> C1 train -> verify -> C1 eval
-> official judge on buyer_select_new (E23, C0, C1 in one batch) -> scores -> frozen select gates
-> (only if a provisional candidate) one confirm (buyer_confirm_new + dev_confirm, judge) -> selection.json
-> (only if a new model is selected) candidate copy, full public (both parts + judge) and the user's
one public veto (Part 1 headline >= plan veto) -> handoff.json.

Every GPU job runs through run_supervised.py (both GPU locks, progress, stall, time limits, deadline
keys from session.json: training D_train, selection-stage jobs D_select, the rest D_gpu). A required
step that fails ends the run on the E23 path: handoff.json says what failed, and the exit code is 1.
Choosing E23 because a gate did not pass is a normal outcome (exit 0). Commands are read from
plan.json "commands" so the drills can replace them with CPU dummies (--dry-plan).
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Run:
    def __init__(self, session, plan_path):
        self.S = os.path.realpath(session)
        for name in ("status.json", "handoff.json"):  # amendment 1: keep an earlier attempt's records
            if os.path.exists(f"{self.S}/{name}"):
                k = 1
                while os.path.exists(f"{self.S}/{name[:-5]}_attempt{k}.json"):
                    k += 1
                os.replace(f"{self.S}/{name}", f"{self.S}/{name[:-5]}_attempt{k}.json")
        self.plan_path = plan_path or f"{self.S}/plan.json"
        self.plan = json.load(open(self.plan_path))
        frozen = open(f"{self.S}/plan.sha256").read().split()[0] if os.path.exists(f"{self.S}/plan.sha256") else None
        if frozen != sha(self.plan_path):
            raise SystemExit("plan.json does not match the frozen plan.sha256")
        self.session = json.load(open(f"{self.S}/session.json"))
        self.state_path = f"{self.S}/status.json"
        self.state = {"steps": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def log(self, step, status, **kw):
        self.state["steps"].append({"time": time.strftime("%H:%M:%S"), "step": step, "status": status, **kw})
        self.state["current"] = f"{step}: {status}"
        tmp = self.state_path + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1, ensure_ascii=False)
        os.replace(tmp, self.state_path)
        with open(f"{self.S}/events.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": "phase_e25", "state": f"{step}: {status}", **kw}, ensure_ascii=False) + "\n")
        print(time.strftime("%H:%M:%S"), step, status, kw, flush=True)

    def deadline(self, key):
        return self.session["deadlines"][key]["epoch"]

    def wait_gpu_free(self, limit):
        """Wait until no supervised GPU job holds the project-wide lock (e.g. the E23 baseline)."""
        path = f"{os.path.dirname(self.S)}/.gpu_global.lock"
        end = time.time() + limit
        while True:
            with open(path, "a") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return True
                except BlockingIOError:
                    pass
            if time.time() > end:
                return False
            time.sleep(10)

    def job(self, name, argv, gpu=True, progress="none", deadline_key="D_gpu", max_seconds=1800, expect=0, stall=600, init_grace=900, done_check=None, done_timeout=120):
        cmd = [PY, f"{HERE}/run_supervised.py", "--session", self.S, "--name", name, "--progress", progress, "--deadline-key", deadline_key,
               "--max-seconds", str(max_seconds), "--expect-seconds", str(expect), "--stall", str(stall), "--init-grace", str(init_grace),
               "--poll", str(self.plan.get("poll", 15)), "--done-timeout", str(done_timeout)]
        if gpu:
            cmd.append("--gpu")
        if done_check:
            cmd += ["--done-check", json.dumps(done_check)]
        r = subprocess.run(cmd + ["--"] + argv)
        res = json.load(open(f"{self.S}/jobs/{name}.json"))
        ok = r.returncode == 0 and res["status"] == "completed" and res.get("group_cleaned", True) is not False
        self.log(name, "completed" if ok else "failed", reason=res.get("reason"), elapsed=res.get("elapsed"))
        return ok

    def cpu(self, name, argv, timeout=600):
        t = time.monotonic()
        try:
            r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, start_new_session=True)
            ok, out = r.returncode == 0, r.stdout[-3000:]
        except subprocess.TimeoutExpired as e:
            ok, out = False, f"timeout after {timeout}s"
        os.makedirs(f"{self.S}/logs", exist_ok=True)
        open(f"{self.S}/logs/{name}.log", "w").write(out)
        self.log(name, "completed" if ok else "failed", elapsed=round(time.monotonic() - t, 1))
        return ok

    def fmt(self, argv, **kw):
        out = []
        for a in argv:
            if a in ("{sets}", "{models}"):  # list placeholders become separate arguments
                out += kw[a[1:-1]].split()
            else:  # only known placeholders are replaced; other braces (JSON, code) stay as they are
                for k, v in {"S": self.S, "PY": PY, **kw}.items():
                    a = a.replace("{" + k + "}", str(v))
                out.append(a)
        return out

    def handoff(self, outcome, selected, failure=None, code=0):
        data = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "outcome": outcome, "selected": selected, "failure": failure,
                "steps": self.state["steps"], "release_final_untouched": "this controller never writes release/final"}
        json.dump(data, open(f"{self.S}/handoff.json", "w"), indent=1, ensure_ascii=False)
        self.log("handoff", outcome, selected=selected, failure=failure)
        sys.exit(code)

    def fail(self, what):
        self.handoff("failed_fallback_E23", "E23", failure=what, code=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--dry-plan", help="drills: a plan with dummy commands (still checked against plan.sha256)")
    a = ap.parse_args()
    R = Run(a.session, a.dry_plan)
    try:
        sequence(R)
    except SystemExit:
        raise
    except Exception as e:  # anything unforeseen: stop on the E23 path with the reason, never exit 0
        R.fail(f"controller exception: {type(e).__name__}: {e}")


def sequence(R):
    P, C = R.plan, R.plan["commands"]
    lim = P["limits"]
    R.log("controller", "started", plan_sha256=sha(R.plan_path), pid=os.getpid())

    if not R.wait_gpu_free(lim["wait_gpu"]):
        R.fail("GPU still held by another supervised job")
    if not R.cpu("E23_baseline", R.fmt(C["eval"], tag="E23", adapter=P["parent"], sets=" ".join(P["select_sets_new"])) , timeout=lim["eval_total"]):
        R.fail("E23 baseline generation")

    for m in ("C0", "C1"):
        train = R.fmt(C["train"], m=m)
        log = f"{R.S}/{m}/train_log.jsonl"
        prev = f"{R.S}/jobs/train_{m}.json"
        if os.path.exists(prev) and json.load(open(prev))["status"] == "completed":
            # amendment 1 (resume): a training that already completed is never run again; it still has to
            # pass the same verification (done.json, hashes, architecture) before it is used.
            R.log(f"train_{m}", "reused", reason="completed in an earlier controller attempt")
        else:
            ok = R.job(f"train_{m}", train, progress=f"update:{log}", deadline_key="D_train", max_seconds=lim["train"], expect=lim["train_expect"],
                       stall=lim["train_stall"], init_grace=lim["train_init"])
            if not ok:
                R.fail(f"train {m}")
        integ = f"{R.S}/scores/{m}_integrity.json"
        if os.path.exists(integ):
            if not json.load(open(integ))["passed"]:
                R.fail(f"verify {m} (earlier result failed)")
            R.log(f"verify_{m}", "reused")
        elif not R.cpu(f"verify_{m}", R.fmt(C["verify"], m=m)):
            R.fail(f"verify {m}")
        if not R.cpu(f"eval_{m}", R.fmt(C["eval"], tag=m, adapter=f"{R.S}/{m}/checkpoints/step_{P['updates']}", sets=" ".join(P["select_sets_all"])), timeout=lim["eval_total"]):
            R.fail(f"eval {m}")

    if not R.cpu("judge_build_select", R.fmt(C["judge_build"], split="select", models="E23 C0 C1")):
        R.fail("judge build (select)")
    if not R.job("judge_select", R.fmt(C["judge_run"], split="select"), progress=f"rows:{R.S}/judge/select/judgments.jsonl", deadline_key="D_select",
                 max_seconds=lim["judge"], expect=lim["judge_expect"], stall=lim["judge_stall"], init_grace=lim["judge_init"],
                 done_check=R.fmt(C["judge_check"], split="select")):
        R.fail("judge (select)")
    if not R.cpu("score_select", R.fmt(C["score"], split="select", models="E23 C0 C1")):
        R.fail("scoring (select)")
    if not R.cpu("gates_select", R.fmt(C["gates_select"])):
        R.fail("select gates")
    cand = json.load(open(f"{R.S}/scores/provisional.json"))["provisional_candidate"]
    R.log("provisional", cand)

    selected = "E23"
    if cand != "E23":
        if time.time() > R.deadline("D_select"):
            R.fail("D_select passed before confirm")
        adapter = f"{R.S}/{cand}/checkpoints/step_{P['updates']}"
        if not R.cpu(f"eval_{cand}_confirm", R.fmt(C["eval"], tag=cand, adapter=adapter, sets=" ".join(P["confirm_sets"])), timeout=lim["eval_total"]):
            R.fail("confirm generation")
        if not R.cpu("judge_build_confirm", R.fmt(C["judge_build"], split="confirm", models=f"E23 {cand}")):
            R.fail("judge build (confirm)")
        if not R.job("judge_confirm", R.fmt(C["judge_run"], split="confirm"), progress=f"rows:{R.S}/judge/confirm/judgments.jsonl", deadline_key="D_select",
                     max_seconds=lim["judge"], expect=lim["judge_confirm_expect"], stall=lim["judge_stall"], init_grace=lim["judge_init"],
                     done_check=R.fmt(C["judge_check"], split="confirm")):
            R.fail("judge (confirm)")
        if not R.cpu("score_confirm", R.fmt(C["score"], split="confirm", models=f"E23 {cand}")):
            R.fail("scoring (confirm)")
        if not R.cpu("gates_confirm", R.fmt(C["gates_confirm"], cand=cand)):
            R.fail("confirm gates")
        selected = json.load(open(f"{R.S}/selection_gates.json"))["selected"]
    sel = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "provisional": cand, "selected": selected,
           "adapter": P["parent"] if selected == "E23" else f"{R.S}/{selected}/checkpoints/step_{P['updates']}",
           "public_used_for_selection": False, "note": "public is opened only after this file, once, for the selected new model"}
    json.dump(sel, open(f"{R.S}/selection.json", "x"), indent=1)
    R.log("selection", selected)
    if selected == "E23":
        R.handoff("completed_kept_E23", "E23")

    if not R.cpu("candidate_copy", R.fmt(C["copy_candidate"], m=selected)):
        R.fail("candidate copy")
    if not R.job("final_public", R.fmt(C["final_public"]), progress="none", deadline_key="D_gpu", max_seconds=lim["public"], expect=lim["public_expect"],
                 done_check=R.fmt(C["public_check"])):
        R.fail("full public evaluation")
    if not R.cpu("public_veto", R.fmt(C["public_veto"])):
        R.fail("public veto check")
    verdict = json.load(open(f"{R.S}/final_public/veto.json"))
    R.handoff("completed_new_candidate" if verdict["passes"] else "completed_veto_kept_E23", selected if verdict["passes"] else "E23")


if __name__ == "__main__":
    main()
