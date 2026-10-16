#!/usr/bin/env sh
# Refresh the bundled fallback copy of the site's data from a local portfolio checkout.
set -e
cd "$(dirname "$0")/.."
(cd "${PORTFOLIO_DIR:-../portfolio}" && node scripts/build-data.mjs >/dev/null)
cp "${PORTFOLIO_DIR:-../portfolio}/public/profile.json" profile.snapshot.json
echo "snapshot updated: $(wc -c < profile.snapshot.json) bytes"
