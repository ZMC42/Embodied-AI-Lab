#!/usr/bin/env bash
set -eo pipefail
LAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$LAB_ROOT"
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
for name in data models logs cache; do
  if [[ ! -e "$name" ]]; then
    echo "Missing $name. Run: ln -s <storage>/$name $name" >&2
    exit 1
  fi
done
"$LAB_ROOT/scripts/setup/venv-arena.sh"
"$LAB_ROOT/scripts/setup/venv-groot.sh"
"$LAB_ROOT/scripts/setup/venv-rl.sh"
