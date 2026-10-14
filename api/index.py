"""
Vercel Serverless Function Entry Point
--------------------------------------
Exports Flask application instance for Vercel deployment.
"""
import os
import sys

# Ensure root and backend directory are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

for p in [ROOT_DIR, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app import app

# Export application for WSGI
application = app

if __name__ == "__main__":
    app.run()
