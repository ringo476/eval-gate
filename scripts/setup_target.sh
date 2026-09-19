#!/usr/bin/env bash
# Vendors the target app the harness evaluates, pinned to an exact commit
# so results are reproducible across machines and over time.
set -euo pipefail

TARGET_DIR="targets/rag-fusion"
TARGET_REPO="https://github.com/Raudaschl/rag-fusion.git"
TARGET_COMMIT="e6ae2d44f440cd52356c18ebef2b8bcb5cb436b9"

if [ -d "$TARGET_DIR/.git" ]; then
    echo "Target already present at $TARGET_DIR"
else
    echo "Cloning $TARGET_REPO ..."
    git clone --quiet "$TARGET_REPO" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
git fetch --quiet origin "$TARGET_COMMIT" 2>/dev/null || true
git checkout --quiet "$TARGET_COMMIT"
echo "Pinned targets/rag-fusion to $TARGET_COMMIT"
