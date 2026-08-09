# Rollback

RAI's public artifacts are immutable in practice: do not promise that a bad
crate or Git tag can simply be replaced under the same version.

## Before a release

Record the candidate commit, crate archive checksums, container digest (if
published), CI URL, state-format compatibility result, release owner, and the
last known-good versions. Keep a pre-upgrade snapshot backup and its checksum.

## If a candidate fails before publication

Stop the rollout and marketing, keep the previous binary active, restore the
backup only if the candidate wrote state, and open a tracked incident with the
failed gate and logs. Fix forward on a new candidate commit and rerun every
gate.

## If a published release is unsafe

1. Halt promotion and publish a clear advisory describing affected versions.
2. Yank affected crates.io versions when installation should be discouraged;
   yanking does not delete already downloaded code or break existing lockfiles.
3. Mark the GitHub release as affected. Do not move or recreate its tag to hide
   the original artifact.
4. Direct operators to the last known-good version and snapshot backup. Verify
   recovery on a copy before replacing live state.
5. Publish a higher patch version with the fix and complete evidence. Never
   reuse the affected version number.

If a container is later published, deploy by immutable digest and roll back to
the recorded last known-good digest. Removing a mutable tag is not sufficient
because cached images and pulled digests remain accessible.
