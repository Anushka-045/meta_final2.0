"""
server/app.py — OpenEnv multi-mode deployment entry point.
Re-exports the unified FastAPI app from inference.py.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference import app  # noqa: F401


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("inference:app", host="0.0.0.0", port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()