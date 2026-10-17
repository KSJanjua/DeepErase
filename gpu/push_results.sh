#!/bin/bash
# Commit the given result paths and push them, retrying through a rebase.
#
#   bash gpu/push_results.sh "H100: <what>" results/gates/x.json results/logs/x.log ...
#
# Only the paths named are staged, so nothing else in the tree is swept in.
# Paths that do not exist are skipped.
set -u
msg="$1"; shift
paths=()
for p in "$@"; do [ -e "$p" ] && paths+=("$p"); done
[ ${#paths[@]} -eq 0 ] && { echo "push_results: nothing to add"; exit 0; }
git add -- "${paths[@]}" || exit 1
if git diff --cached --quiet; then echo "push_results: no changes"; exit 0; fi
git commit -q -m "$msg" || exit 1
for i in 1 2 3 4 5; do
  if git pull -q --rebase --autostash origin rishabh-development && git push -q origin rishabh-development; then
    echo "push_results: pushed ($(git log --oneline -1))"; exit 0
  fi
  echo "push_results: attempt $i failed, retrying"; sleep $((10 * i))
done
echo "push_results: could not push; the commit stays local"; exit 1
