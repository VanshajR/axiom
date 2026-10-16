"""Run Axiom's eval set against a running backend and score each answer.

    python evals/run.py                                   # local backend on :8100
    python evals/run.py --url https://targaryen001-wyvern.hf.space --origin https://vanshajraghuvanshi.dev

Each case checks what a visitor would notice: the right lookup, the right page
action (and no page action when they only asked a question), the facts that must
be in the answer and the things that must never be. Results go to evals/results/.
"""
import argparse
import datetime as dt
import json
import pathlib
import re
import statistics
import time

import httpx

HERE = pathlib.Path(__file__).parent


def ask(client: httpx.Client, url: str, origin: str, message: str) -> dict:
    run = {"text": "", "tools": [], "actions": [], "done": None, "error": None}
    started = time.perf_counter()
    with client.stream("POST", f"{url}/chat", json={"message": message, "history": []},
                       headers={"Origin": origin}, timeout=180) as r:
        if r.status_code != 200:
            run["error"] = f"HTTP {r.status_code}"
            return run
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            e = json.loads(line[6:])
            if e["type"] == "text":
                run["text"] += e["text"]
            elif e["type"] == "tool":
                run["tools"].append(e["name"])
            elif e["type"] == "action":
                run["actions"].append({"name": e["name"], "args": e.get("args", {})})
            elif e["type"] == "done":
                run["done"] = e
            elif e["type"] == "error":
                run["error"] = e["message"]
    run["seconds"] = round(time.perf_counter() - started, 2)
    return run


def action_matches(want: dict, got: dict) -> bool:
    if want["name"] != got["name"]:
        return False
    for k, v in want.get("args", {}).items():
        if str(got["args"].get(k, "")).lower() != str(v).lower():
            return False
    for k, v in want.get("args_prefix", {}).items():
        if not str(got["args"].get(k, "")).lower().startswith(v.lower()):
            return False
    return True


def score(case: dict, run: dict) -> list[str]:
    """Problems with this answer; empty means it passed."""
    if run["error"]:
        return [run["error"]]
    problems = []
    text = run["text"]
    if case.get("tools_any") and not set(case["tools_any"]) & set(run["tools"]):
        problems.append(f"expected one of {case['tools_any']}, called {run['tools']}")
    for want in case.get("actions", []):
        if not any(action_matches(want, got) for got in run["actions"]):
            problems.append(f"missing action {want}, got {run['actions']}")
    if case.get("no_actions") and run["actions"]:
        problems.append(f"acted on the page unasked: {run['actions']}")
    if case.get("contains_any") and not any(s in text for s in case["contains_any"]):
        problems.append(f"answer has none of {case['contains_any']}")
    for s in case.get("not_contains", []):
        if s.lower() in text.lower():
            problems.append(f"answer contains {s!r}")
    for pattern in case.get("not_contains_regex", []):
        if re.search(pattern, text):
            problems.append(f"answer matches {pattern!r}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8100")
    p.add_argument("--origin", default="http://localhost:3000")
    p.add_argument("--only", help="comma separated case ids")
    p.add_argument("--pause", type=float, default=0.0, help="seconds between cases (the live Space allows 12 a minute)")
    args = p.parse_args()

    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]

    rows = []
    with httpx.Client() as client:
        for i, case in enumerate(cases):
            if i and args.pause:
                time.sleep(args.pause)
            run = ask(client, args.url, args.origin, case["ask"])
            problems = score(case, run)
            done = run["done"] or {}
            rows.append({"id": case["id"], "passed": not problems, "problems": problems,
                         "seconds": run.get("seconds"), "model": done.get("model"),
                         "attempts": done.get("attempts"), "tools": run["tools"],
                         "actions": [a["name"] for a in run["actions"]], "answer": run["text"][:400]})
            mark = "PASS" if not problems else "FAIL"
            print(f"{mark} {case['id']:<24} {run.get('seconds', 0):>6}s  {done.get('model') or '-'}"
                  + ("" if not problems else f"\n     {'; '.join(problems)}"))

    passed = sum(r["passed"] for r in rows)
    secs = [r["seconds"] for r in rows if r["seconds"] is not None]
    summary = {
        "when": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "url": args.url,
        "passed": passed,
        "total": len(rows),
        "pass_rate": round(passed / len(rows), 3) if rows else 0,
        "latency_s": {"p50": round(statistics.median(secs), 2),
                      "p95": round(sorted(secs)[max(0, int(len(secs) * 0.95) - 1)], 2)} if secs else {},
        "needed_fallback": sum(1 for r in rows if (r["attempts"] or 1) > 1),
        "answered_by": {m: sum(1 for r in rows if r["model"] == m) for m in {r["model"] for r in rows if r["model"]}},
    }
    print(f"\n{passed}/{len(rows)} passed, p50 {summary['latency_s'].get('p50')}s, "
          f"p95 {summary['latency_s'].get('p95')}s, {summary['needed_fallback']} needed a fallback")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    (out / f"{stamp}.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
