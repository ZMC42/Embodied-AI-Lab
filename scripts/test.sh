#!/usr/bin/env bash
set -eo pipefail

usage() {
  echo 'Usage: scripts/test.sh <arena|groot|rl|all> [--gpu]' >&2
  exit 2
}
[[ $# -ge 1 && $# -le 2 ]] || usage
case "$1" in
  all) environments=(arena groot rl) ;;
  arena|groot|rl) environments=("$1") ;;
  *) usage ;;
esac
if [[ $# == 2 ]]; then
  [[ "$2" == --gpu ]] || usage
fi

source "$(dirname "$0")/env.sh"
cd "$LAB_ROOT"
shopt -s nullglob
for environment in "${environments[@]}"; do
  case "$environment" in
    arena) python="$ARENA_PY" ;;
    groot) python="$GROOT_PY" ;;
    rl) python="$RL_PY" ;;
  esac
  if [[ "${2:-}" == --gpu ]]; then
    echo "[$environment] GPU tests"
    "$python" -m unittest discover -s "tests/$environment" -t . -p 'test_gpu_*.py'
    continue
  fi
  files=(tests/"$environment"/test_*.py)
  count=0
  for file in "${files[@]}"; do
    [[ "${file##*/}" == test_gpu_* ]] && continue
    echo "[$environment] ${file##*/}"
    "$python" -m unittest discover -s "tests/$environment" -t . -p "${file##*/}"
    count=$((count + 1))
  done
  if [[ $count == 0 ]]; then
    echo "No regular test modules found in tests/$environment" >&2
    exit 1
  fi
done
