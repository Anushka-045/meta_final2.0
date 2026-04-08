"""
server/app.py — OpenEnv multi-mode deployment entry point.
Finds a free port if the default is already in use.
"""
import sys
import os
import socket

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference import app  # noqa: F401


def find_free_port(preferred: int) -> int:
    """Return preferred port if free, otherwise let OS pick a free one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            # preferred port is taken — ask OS for a free one
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]


def main():
    import uvicorn
    preferred = int(os.environ.get("PORT", 7860))
    port = find_free_port(preferred)
    if port != preferred:
        print(f"[WARN] Port {preferred} in use, binding on {port} instead.", flush=True)
    uvicorn.run("inference:app", host="0.0.0.0", port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()