#!/bin/sh
# Stages Linux builds of the nollama engine tests and surprisal tests for scripts/gorun.py: gostage.sh <dir>
set -e
OUT=${1:?usage: gostage.sh <dir>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NOLLAMA=$(cd "$ROOT" && go list -m -f '{{.Dir}}' github.com/benallfree/nollama)
rm -rf "$OUT" && mkdir -p "$OUT/engine/testdata" "$OUT/surprisal" "$OUT/spec"
(cd "$NOLLAMA" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -c -o "$OUT/engine/engine.test" ./engine)
cp -R "$NOLLAMA/engine/testdata/intmodel" "$OUT/engine/testdata/"
(cd "$ROOT" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -c -o "$OUT/surprisal/surprisal.test" ./surprisal)
cp -R "$ROOT/spec/vectors" "$OUT/spec/"
mkdir -p "$OUT/bin" && (cd "$ROOT" && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o "$OUT/bin/surprisal.test" ./cmd/surprisal)
echo "staged in $OUT: $(du -sh "$OUT" | cut -f1)"
