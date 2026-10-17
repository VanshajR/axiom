"""Vercel entry point: Vercel finds `app` here and serves every route with it."""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PROFILE_SNAPSHOT", str(ROOT / "profile.snapshot.json"))

from axiom.app import app  # noqa: E402,F401
