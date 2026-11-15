# Land a pull request through the merge queue

Every change reaches `master` through Mergify's merge queue. The configuration
is [`.mergify.yml`](../.mergify.yml). The gate workflow supplies `pr-gate` on
ordinary PRs and `gate` on temporary batch drafts. Development tests run
through admitted Flux jobs, BuildBuddy or GitHub-hosted CI; cite their job or
run IDs, terminal verdicts and logs in the PR body. New Tessera work must not
start using PrismaBuild again under the recorded Flux migration exception.

## Before you queue

1. Follow branch rule 12. Name your branch `tessera-<issue>` or
   `tessera-<issue>-<word>`, with lowercase letters, digits and hyphens in
   the suffix, for example `tessera-<issue>-tl3-<node>`.
2. Reference the same issue in the PR body, for example `Refs #<issue>.`.
   Rule 12 checks that the branch number names a linked issue in this
   repository. Use a closing reference only when the PR completes the issue.
3. Wait for the `branch rule` check and `pr-gate` to pass.

Classic branch protection requires `branch rule` only (until the first real
batch has run `gate`, it still requires `pure`). Mergify applies classic
required checks to your pull request and to the batch draft. Both checks skip
branch rule 12 on an authenticated Mergify draft, so the draft is judged by
`gate`. `pure` still runs on every pull request and inside `gate`.

## Queue and dequeue

- Add the `queue` label. Mergify adds `queued` after accepting the PR.
- Remove `queue` to take the PR out of the queue.
- People with write permission and TL3's GitHub App, `tl3-robtand[bot]`, can
  use the comment commands `@mergifyio queue` and `@mergifyio requeue`.
  Put the command at the start of the comment.

## What happens to a PR

Mergify checks one PR at a time, in `queue` label order. There are no priority lanes.
For each PR, Mergify opens a temporary draft on a `mergify/merge-queue/` branch.
The draft contains the newest `master` and that PR. Its `gate` must pass before merge.
An external merge resets the check. Original PR heads stay unchanged.

Mergify merges each passing PR with a merge commit. A failed PR leaves the queue.
Mergify adds `dequeued` and links the failure. Checks have a 30-minute timeout and no automatic retries.

## Requeue after a failure

A failed PR stays out of the queue even if it still has the `queue` label.
Removing and re-adding that label does not requeue it.

1. Read the failed check and Mergify's comment. Fix the PR, or resolve the
   infrastructure failure, and push any needed changes.
2. Keep the `queue` label and wait for the current head's `pr-gate` and
   required checks to pass.
3. Comment `@mergifyio queue`. Mergify rechecks the conditions and queues the PR.
   The older `@mergifyio requeue` command still works but is deprecated.

Never merge outside the queue.
