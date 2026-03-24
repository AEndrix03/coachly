#!/usr/bin/env sh

set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

pull_repo() {
  repo_dir="$1"
  repo_name="$(basename "$repo_dir")"

  echo "==> $repo_name"
  (
    cd "$repo_dir"
    git pull --ff-only
  )
}

for dir in "$ROOT_DIR"/*; do
  [ -d "$dir" ] || continue

  if [ -d "$dir/.git" ]; then
    pull_repo "$dir"
    continue
  fi

  for subdir in "$dir"/*; do
    [ -d "$subdir" ] || continue
    [ -d "$subdir/.git" ] || continue
    pull_repo "$subdir"
  done
done
