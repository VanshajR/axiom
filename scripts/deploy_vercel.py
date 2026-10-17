"""Deploy Axiom to Vercel as a Python function.

    python scripts/deploy_vercel.py

Reads VERCEL_TOKEN and the model keys from .env, stores the keys as Vercel
environment variables (never in the upload), and deploys a clean copy of the
app. The copy has no git metadata: Vercel's Hobby plan blocks deploys whose
commit author it can't match to the account, and the code doesn't need it.
"""
import json
import pathlib
import re
import shutil
import subprocess
import tempfile

import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT = "axiom"
SITE = "https://vanshajraghuvanshi.dev"
SITE_PREVIEW = "https://portfolio-beta-ten-wjzxv5saya.vercel.app"
SHIP = ["index.py", "src", "vercel.json", "requirements.txt", ".python-version", ".vercelignore", "profile.snapshot.json"]
VERCEL = ["npx", "--yes", "vercel@64.4.0"]


def read_env(path: pathlib.Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def vercel(args: list[str], cwd: pathlib.Path, token: str) -> str:
    out = subprocess.run([*VERCEL, *args, "--token", token], cwd=cwd, capture_output=True, text=True,
                         shell=(shutil.which("npx") or "").lower().endswith(".cmd"))
    if out.returncode:
        raise SystemExit(f"vercel {args[0]} failed:\n{out.stderr[-2000:]}")
    return out.stdout.strip()


def set_env(project: dict, token: str, values: dict[str, str]) -> None:
    url = f"https://api.vercel.com/v10/projects/{project['projectId']}/env?upsert=true"
    if project.get("orgId", "").startswith("team_"):
        url += f"&teamId={project['orgId']}"
    for key, value in values.items():
        r = httpx.post(url, headers={"Authorization": f"Bearer {token}"}, timeout=30,
                       json={"key": key, "value": value, "type": "encrypted", "target": ["production"]})
        if r.status_code >= 300:
            raise SystemExit(f"setting {key} failed: {r.status_code} {r.text[:200]}")


def main() -> None:
    env = read_env(ROOT / ".env")
    token = env["VERCEL_TOKEN"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        stage = pathlib.Path(tmp) / PROJECT
        stage.mkdir()
        for name in SHIP:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            else:
                shutil.copy2(src, stage / name)
        if (ROOT / ".vercel").exists():
            shutil.copytree(ROOT / ".vercel", stage / ".vercel")
        else:
            vercel(["link", "--yes", "--project", PROJECT], stage, token)
            shutil.copytree(stage / ".vercel", ROOT / ".vercel")

        project = json.loads((stage / ".vercel" / "project.json").read_text())
        set_env(project, token, {
            **{k: env[k] for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY") if env.get(k)},
            "PROFILE_URL": f"{SITE_PREVIEW}/profile.json",
            "ALLOWED_ORIGINS": f"{SITE},https://www.vanshajraghuvanshi.dev,{SITE_PREVIEW},http://localhost:3000",
        })
        out = vercel(["deploy", "--prod", "--yes"], stage, token)
    urls = re.findall(r"https://[\w.-]+\.vercel\.app", out)
    print(f"deployed: {urls[-1] if urls else out[-300:]}")


if __name__ == "__main__":
    main()
