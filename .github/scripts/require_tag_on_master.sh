#!/usr/bin/env bash
# Refuse to publish a commit that is not on the reviewed branch.
#
# The publish job's only trigger is a `v*` tag, and a tag is not a review
# gate: whoever can push one can push it at any commit, including one that was
# never on master and was never reviewed.  This script is what makes the tag
# mean "the reviewed branch", and it answers from git alone -- reachability,
# not a name, not a message, not who pushed it.
#
#   require_tag_on_master.sh <commit> [remote] [branch]
#
# Exits 0 when <commit> is reachable from <remote>/<branch>; otherwise says
# why and exits non-zero.  Every other outcome is a refusal too: a question
# this script cannot answer is never answered "yes".
#
# It lives here, and not in a `run:` block, so that it has exactly one home
# and so that it can be run.  `tests/test_ci_workflow.py` runs it against real
# repositories -- a commit on the branch, an older commit on it, a commit off
# it, a shallow checkout, and a branch that does not exist.
set -euo pipefail

commit="${1:?usage: require_tag_on_master.sh <commit> [remote] [branch]}"
remote="${2:-origin}"
branch="${3:-master}"

# A shallow clone answers `--is-ancestor` from whatever history it happens to
# hold, which is a different question.  Refuse, and name the setting that
# fixes it.
if [ "$(git rev-parse --is-shallow-repository)" != "false" ]; then
    echo "refused: shallow checkout; reachability cannot be answered from" \
         "grafted history. The publish job's checkout needs fetch-depth: 0" >&2
    exit 1
fi

if ! git fetch --no-tags --quiet "$remote" \
        "+refs/heads/$branch:refs/remotes/$remote/$branch"; then
    echo "refused: could not fetch $remote/$branch; a publish gate does not" \
         "pass on a branch it was unable to read" >&2
    exit 1
fi

if git merge-base --is-ancestor "$commit" "refs/remotes/$remote/$branch"; then
    echo "ok: $commit is reachable from $remote/$branch"
    exit 0
fi

echo "refused: $commit is not reachable from $remote/$branch, so it is not a" \
     "commit this branch reviewed. A tag publishes only what master contains." >&2
exit 1
