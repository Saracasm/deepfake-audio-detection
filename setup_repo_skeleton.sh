#!/bin/bash
# setup_repo_skeleton.sh
# Run this ONCE in your repo root to create the empty directory structure.
# It creates folders and .gitkeep files so git tracks them.

set -e

DIRS=(
  "api"
  "app"
  "config"
  "checkpoints"
  "data/raw"
  "data/processed"
  "data/protocols"
  "docs"
  "notebooks"
  "results/figures/2019"
  "results/figures/2021"
  "results/figures/wavefake"
  "results/metrics"
  "src/data"
  "src/models"
  "src/training"
  "src/evaluation"
  "src/utils"
  "tests"
)

for dir in "${DIRS[@]}"; do
  mkdir -p "$dir"
  # Add .gitkeep to empty folders so git tracks them
  if [ -z "$(ls -A "$dir")" ]; then
    touch "$dir/.gitkeep"
  fi
  echo "  ✅  $dir"
done

# Create empty __init__.py files for Python packages
for pkg in src src/data src/models src/training src/evaluation src/utils api tests; do
  touch "$pkg/__init__.py"
done

echo ""
echo "✅ Repo skeleton created."
echo ""
echo "Next steps:"
echo "  1. Add the Phase 1 files (requirements.txt, .gitignore, README.md, config.yaml, etc.)"
echo "  2. git add . && git commit -m 'Phase 1: initial repo structure'"
echo "  3. git push origin main"
