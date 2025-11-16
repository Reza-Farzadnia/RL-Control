#!/usr/bin/env bash
set -euo pipefail

# Build all TeX sources in tex/ to PDFs in docs/models with aux in tex/build
ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
cd "$ROOT_DIR"

mkdir -p docs/models tex/build

shopt -s nullglob
for f in tex/*.tex; do
  echo "[latexmk] Building $f -> docs/models/"
  latexmk -pdf -silent -f -r "$ROOT_DIR/latexmkrc" "$f"
done
echo "Done. PDFs in docs/models, aux in tex/build."

