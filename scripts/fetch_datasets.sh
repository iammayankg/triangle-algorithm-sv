#!/usr/bin/env bash
# Fetch the standard LIBSVM benchmark battery.
# Usage: ./scripts/fetch_datasets.sh [target_dir]   (default: data/)
set -euo pipefail
DIR="${1:-data}"
mkdir -p "$DIR"; cd "$DIR"
BASE="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
get () {  # get <remote-name> [bz2]
  local f="$1"; local out="${f%.bz2}"
  [ -f "$out" ] && { echo "have $out"; return; }
  echo "fetching $f"; curl -fsSLO "$BASE/$f"
  case "$f" in *.bz2) bunzip2 -f "$f";; esac
}
get a9a
get a9a.t
get w8a
get w8a.t
get ijcnn1.bz2
get ijcnn1.t.bz2
get gisette_scale.bz2
get gisette_scale.t.bz2
get covtype.libsvm.binary.scale.bz2
# sparse-only sets (need sparse solver support - see docs/experiments_protocol.md):
get real-sim.bz2
get rcv1_train.binary.bz2
echo "done: $(ls -la)"
