# V2 external supervision integration

The vendored production files are unchanged, reviewed dependencies:

- supervise.py SHA-256: 35c91a94456bb23339f971a20c898e0a51140904f720c73dfc4c808be8818429
- evidence.py SHA-256: 179877eae7a13c917f6de545e47b5ebeb090028de8f25cd45e954317c1f3463e

The upstream production implementation had a bounded 26-test synthetic verification
run. This package does not copy upstream operational records. Its own integration
tests use file fixtures and synthetic calls; they do not execute process signals,
pretrained models or the upstream signal-test suite. A separately reviewed cheap
single-process integration canary remains a prerequisite to model execution.

## Actual guarantees and limits

The supervisor owns one trusted Linux worker process including its threads. It
starts a private session, retains exclusive child-reaping ownership, samples
worker and supervisor CPU/RSS at 0.1-second intervals, obtains final wait4 child
CPU/peak-RSS evidence, and rejects invalid or missing accounting. A separate
Python watchdog requests private-group SIGKILL at the wall deadline. Final checks
also reject wall/CPU/RSS overruns. The RSS final bound sums separate child and
supervisor lifetime high-water marks; it is conservative, not a concurrent peak.

This is NOT kernel containment, whole-process-tree accounting, a guarantee that
sampled caps were never transiently exceeded, or a real-time deadline. Worker
forking, external reaping, escaping descendants and unrelated writers violate the
contract. Scheduler/GIL/kernel stalls or a killed/stopped supervisor can delay
enforcement. The worker uses only in-process CPU inference and no subprocesses.

Emergency cleanup requests immediate SIGKILL, not SIGTERM followed by grace.
The existing 15-second grace allowance is unchanged but unused. SIGKILL may
prevent a final failure marker; a previously fsynced planned attempt still counts
and all surviving evidence is retained. A missing acceptance record is failure.

CPU threshold is derived as twice the allowed phase wall time for the two-thread
worker. The supervisor's input includes its fixed five-second subtraction margin,
so the effective check is exactly that derived CPU envelope. Supervisor CPU is
recorded separately. Thread settings remain two intra-op/BLAS, one inter-op.

Disk checks are cooperative pre-save plus final rejection, NOT an external hard
disk quota. They count previous baseline evidence, current phase evidence and
supervisor logs together against the unchanged 1 GiB cap. Writes/logging can
overshoot before final rejection; no evidence is deleted to meet the cap.

## Lifecycle and acceptance

A new phase directory initially contains only status.json with status running.
The supervisor owns that outer directory. The worker creates its previously
nonexistent evidence/ child, preserving Archive's no-overwrite contract. Each
forward is planned and fsynced before invocation, then its full vector and outcome
are retained. Baseline attempts are 1–128; patch attempts are 129–2816.

The worker seals evidence/manifest.json before writing its completion marker.
After child exit, the supervisor seals RESULTS-MANIFEST.json and writes a distinct
SUPERVISOR-RESULT.json binding that manifest digest. A worker complete marker alone
is insufficient. The integration verifier requires accepted supervisor status,
exit zero, every manifest hash and exact file coverage, the bound inner manifest,
phase and cumulative wall limits, final RSS bound and aggregate disk cap. Only
then is the phase accepted by this study. A supervisor acceptance can therefore
still be rejected by the study's additional checks.

Patch launch requires this accepted baseline, a passing frozen competence gate,
matching source/data/config/model bindings and separate exact-stage clearance.
There is no automatic transition. Review idle time is excluded from execution
wall time; baseline measured supervisor wall time reduces the remaining 2,400
seconds. Each new phase also retains its original 300/2,100-second cap.

Execution-clearance files remain external operational inputs, not public
scientific evidence. They are never copied into output archives or this package.
The source/data hashes and resource measurements remain scientific evidence.
