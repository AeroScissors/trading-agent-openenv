"""
server/app.py — entry point for multi-mode deployment.
Delegates to the existing FastAPI app in main.py.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401 — re-export for uvicorn


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()