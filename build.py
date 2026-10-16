#!/usr/bin/env python3
"""
build.py — Render pre-build script
Downloads required NLTK corpora at build time so they are available at runtime.
Run automatically by Render via: python build.py
"""
import os
import sys
import tempfile

print("=== Running build.py: Downloading NLTK data ===")

try:
    import nltk

    # Use a persistent directory on Render's filesystem
    nltk_dir = os.environ.get("NLTK_DATA", os.path.join(tempfile.gettempdir(), "nltk_data"))
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    corpora = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
        ("corpora/wordnet",           "wordnet"),
    ]

    for path, name in corpora:
        try:
            nltk.data.find(path)
            print(f"  ✓ Already present: {name}")
        except LookupError:
            print(f"  ↓ Downloading: {name} ...")
            nltk.download(name, download_dir=nltk_dir, quiet=False)
            print(f"  ✓ Downloaded: {name}")

    print("=== NLTK data ready ===")
except Exception as e:
    print(f"WARNING: NLTK download failed ({e}). App will attempt lazy download at runtime.")
    sys.exit(0)   # Non-fatal — do NOT fail the build
