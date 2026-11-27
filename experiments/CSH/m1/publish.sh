#!/usr/bin/env bash
set -euo pipefail
csh_repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$csh_repository"
csh_branch=opt/m1-CSH-20261127-02
if [[ "$(git branch --show-current)" != "$csh_branch" ]]; then
  echo 'Run this only in the isolated CSH Mission 1 worktree.' >&2
  exit 1
fi
git diff --quiet
git diff --cached --quiet
# Explicit destination: never push main or force-update a ref.
git push -u origin "HEAD:refs/heads/$csh_branch"
