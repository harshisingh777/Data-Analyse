"""Vercel serverless entry point. Vercel serves the WSGI ``app`` defined here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datalens.api import create_app  # noqa: E402

app = create_app()
