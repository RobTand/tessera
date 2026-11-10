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
- Add `release` alongside `queue` to use the release queue, served first.
- People with write permission and TL3's GitHub App, `tl3-robtand[bot]`, can
  use the comment commands `@mergifyio queue` and `@mergifyio requeue`.
  Put the command at the start of the comment.

## TL3's pull requests

TL3 uses branches named `tessera-<issue>-tl3-<node>` and starts each
pull request body with `Refs #<issue>.`, so `branch rule` passes as it
does for every pull request. TL3 tests each change through admitted Flux jobs,
BuildBuddy or GitHub-hosted CI and
reviews it, then posts its `TL3 approval` check on the reviewed head and adds
the `tl3-queue` label. That pair queues the pull request in the default queue
in place of `queue` and `pr-gate`; `branch rule` stays required. TL3
follows Mergify's queue check on the PR head and never reads `pr-gate`, so a
`pr-gate` condition would leave a failing TL3 PR unqueued without TL3 noticing.
The batch's full `gate` still decides every merge. A TL3 change that would
fail `pr-gate` can enter a batch, fail `gate`, and delay other pull requests
while the batch splits. Don't push to a TL3 branch: TL3 approved that exact
head. The rule name `Queue TL3-approved PRs` is fixed because TL3 follows its
`Rule: Queue TL3-approved PRs (queue)` check.

## What happens to a batch

Mergify groups queued PRs into batches. The default queue batches one to seven
PRs and waits at most two minutes to form a batch. The release queue always
checks one PR at a time. Two batches can run checks concurrently, and merges
follow queue order.

For each batch, Mergify opens a temporary draft PR on a
`mergify/merge-queue/` branch. It contains `master`, the batches ahead of it,
and the current batch. Draft checks preserve each original PR's head, which
TL3's approval names. The batch's `gate` check tests the full gate population
on that integrated SHA. A green `pr-gate` on each original PR alone does not
allow the batch to merge.

When `gate` passes, Mergify merges the original PRs into `master` with merge
commits. A failed batch is split to isolate the failing PR. Mergify marks a
removed PR `dequeued` and links the failure. Each batch has a 30-minute checks
timeout and no automatic Mergify check retries.

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
