"""Compatibility wrapper for the Quantyze web UI.

The browser frontend is now implemented in ``main.py`` so the supported entry
point is:

    python main.py --ui

This module remains as a thin shim for backwards compatibility.
"""

from __future__ import annotations

import argparse

import main as backend

create_app = backend.create_web_app


def main() -> None:
    """Delegate the legacy ``python app.py`` flow to ``main.py``."""
    parser = argparse.ArgumentParser(description="Quantyze web UI compatibility wrapper")
    parser.add_argument("--port", type=int, default=9000, help="HTTP port (default 9000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    args = parser.parse_args()

    app = create_app()
    print(f"Quantyze web UI -> http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
