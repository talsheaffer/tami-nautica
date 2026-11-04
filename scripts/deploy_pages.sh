#!/usr/bin/env bash
# Rebuild the gh-pages branch from main as ONE fresh commit and force-push it.
# Only web files from prototype/ are published (no wiki/, feedback/, media/, *.py).
# Usage: scripts/deploy_pages.sh          # build into a temp dir only (dry run)
#        scripts/deploy_pages.sh --push   # build + force-push to origin gh-pages
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC="$ROOT/prototype"
OUT="$(mktemp -d)"
REMOTE_URL="$(git -C "$ROOT" remote get-url origin)"

cp -R "$SRC/css" "$SRC/js" "$SRC/index.html" "$OUT/"
mkdir -p "$OUT/assets"
for d in scenes drawings fonts; do cp -R "$SRC/assets/$d" "$OUT/assets/$d"; done
find "$OUT" -name '.DS_Store' -delete
touch "$OUT/.nojekyll"

# Old /beta/ links keep working by redirecting to the site root.
mkdir -p "$OUT/beta"
cat > "$OUT/beta/index.html" <<'HTML'
<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=../"><a href="../">Tami-Nautica</a>
HTML

echo "Built $(find "$OUT" -type f | wc -l) files, $(du -sh "$OUT" | cut -f1) -> $OUT"

if [[ "${1:-}" != "--push" ]]; then
  echo "Dry run. Re-run with --push to force-push to origin gh-pages."
  exit 0
fi

cd "$OUT"
git init -q -b gh-pages
git add -A
git commit -q -m "deploy: build from main@$(git -C "$ROOT" rev-parse --short HEAD)"
git push --force "$REMOTE_URL" gh-pages
echo "Pushed. Temp build kept at $OUT"
