"""Create or update the Hugging Face Space that runs Axiom.

    python scripts/deploy_space.py   (HF_TOKEN from the environment or .env)

Model keys are read from .env and stored as Space secrets; nothing secret is uploaded.

New Docker Spaces need a PRO plan (since 2026), so this deploys into an existing
Space when there is one and only tries to create it otherwise. Deploying replaces
the Space's files; its earlier versions stay in the Space's git history.
"""
import os
import pathlib
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPACE = os.getenv("AXIOM_SPACE", "targaryen001/wyvern")
SITE = "https://vanshajraghuvanshi.dev"

# Spaces read their settings from front matter at the top of README.md
SPACE_HEADER = """---
title: Axiom
emoji: 🧭
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Portfolio assistant, MCP tools over two LLM providers
---

"""


def read_env(path: pathlib.Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> None:
    env = read_env(ROOT / ".env")
    api = HfApi(token=os.environ.get("HF_TOKEN") or env["HF_TOKEN"])

    if not api.repo_exists(SPACE, repo_type="space"):
        api.create_repo(SPACE, repo_type="space", space_sdk="docker", private=False)
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY"):
        if env.get(key):
            api.add_space_secret(SPACE, key, env[key])
    api.add_space_variable(SPACE, "PROFILE_URL", f"{SITE}/profile.json")
    # localhost stays allowed so the site under development can talk to the deployed backend
    # the Vercel production alias stays allowed so previews and the bare deploy work too
    api.add_space_variable(SPACE, "ALLOWED_ORIGINS", f"{SITE},https://portfolio-beta-ten-wjzxv5saya.vercel.app,http://localhost:3000")

    with tempfile.TemporaryDirectory() as tmp:
        staging = pathlib.Path(tmp)
        for name in ("Dockerfile", "pyproject.toml", "profile.snapshot.json"):
            shutil.copy(ROOT / name, staging / name)
        shutil.copytree(ROOT / "src", staging / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        (staging / "README.md").write_text(SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
        # delete_patterns clears whatever the Space held before, so stale files can't break the build
        api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=staging,
                          commit_message="Deploy Axiom", delete_patterns="*")

    print(f"deployed: https://huggingface.co/spaces/{SPACE}")
    print(f"api:      https://{SPACE.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
