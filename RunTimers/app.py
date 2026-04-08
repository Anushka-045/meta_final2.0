"""
app.py — Hugging Face Spaces entry point.
HF Spaces looks for this file by default.
Boots the unified RunTimers server on port 7860.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from unified_server import app  # noqa: F401  (re-exported for uvicorn)
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
    )
