"""Linux single-worker supervision. Review README.md before adoption.

Owns/reaps exactly one child; never Popen.poll()/wait() (they lose wait4 rusage).
The fresh session/process group is signalled only while its leader is unreaped.
"""
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import threading
import time
from evidence import atomic, append, digest


class AccountingError(ValueError):
    def __init__(self, error, evidence):
        super().__init__(str(error))
        self.evidence = evidence


def parse_sample(stat_text, status_text):
    fields = stat_text.rsplit(')', 1)[1].split()
    # Linux /proc/PID/stat fields 14 and 15, relative to field 3 (state).
    user, system = int(fields[11]), int(fields[12])
    rss = [line.split() for line in status_text.splitlines() if line.startswith('VmRSS:')]
    if len(rss) != 1 or len(rss[0]) != 3 or rss[0][2] != 'kB':
        raise ValueError('Missing, duplicate or malformed VmRSS accounting')
    kib = int(rss[0][1])
    if user < 0 or system < 0 or kib <= 0:
        raise ValueError('Invalid CPU/RSS accounting')
    return (user + system) / os.sysconf('SC_CLK_TCK'), kib * 1024


def sample(pid):
    evidence = dict(pid=pid, read_started_monotonic=time.monotonic(),
                    stat_text=None, status_text=None, failed_path=None)
    try:
        for name in ('stat', 'status'):
            evidence['failed_path'] = f'/proc/{pid}/{name}'
            evidence[name + '_text'] = Path(evidence['failed_path']).read_text()
        evidence['failed_path'] = None
        return parse_sample(evidence['stat_text'], evidence['status_text'])
    except (OSError, ValueError, IndexError) as exc:
        evidence.update(read_finished_monotonic=time.monotonic(), error=repr(exc))
        raise AccountingError(exc, evidence) from exc


def _rusage(usage):
    result = dict(user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                  cpu_seconds=usage.ru_utime + usage.ru_stime,
                  peak_rss_bytes=usage.ru_maxrss * 1024)
    if any(not math.isfinite(x) or x < 0 for x in result.values()) or result['peak_rss_bytes'] <= 0:
        raise ValueError('Invalid final rusage')
    return result


def monitor(command, out, cfg, env=None):
    """Run a trusted single-process worker, returning 0 only for verified completion.

    cfg matches the original supervisor; its five-second CPU margin is preserved.
    This function must exclusively own SIGCHLD reaping for its child.
    """
    for key in ('wall_seconds', 'cpu_seconds', 'rss_bytes', 'sampling_seconds'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError('Invalid limit: ' + key)
    if cfg['sampling_seconds'] <= 0 or cfg['cpu_seconds'] <= 5 or cfg['rss_bytes'] <= 0:
        raise ValueError('Positive sampling/RSS and CPU > 5 seconds required')
    if not hasattr(os, 'wait4') or not hasattr(os, 'WNOWAIT'):
        raise RuntimeError('Linux wait4/waitid with WNOWAIT required')
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError('Exclusive default SIGCHLD handling required')
    out = Path(out)
    if set(p.name for p in out.iterdir()) != {'status.json'} or json.loads((out / 'status.json').read_text()) != {'status': 'running'}:
        raise ValueError('Require a fresh evidence.reserve output directory')
    start = time.monotonic()
    deadline = start + cfg['wall_seconds']
    lock = threading.RLock()
    finished = threading.Event()
    child = None
    owned_pid = None
    private_session_verified = False
    reaped = False
    reason = None
    accounting_error = None
    peak = 0
    final_child = None
    final_self = None
    terminal_at = None
    watchdog_events = []
    watchdog = None
    code = None
    result_status = None

    def kill_owned_group():
        # A private session is created by Popen. The unreaped leader pins its PID:
        # this PGID cannot have been recycled into an unrelated process group.
        if child is not None and not reaped:
            if type(owned_pid) is not int or owned_pid <= 1 or child.pid != owned_pid or not private_session_verified:
                raise RuntimeError('Refusing to signal an unverified private-session PID')
            # Verify wait ownership before numeric signalling; never signal ECHILD.
            exited()
            if os.getsid(owned_pid) != owned_pid or os.getpgid(owned_pid) != owned_pid:
                raise RuntimeError('Private-session invariant no longer holds')
            try:
                os.killpg(owned_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def stop(why):
        nonlocal reason
        with lock:
            if reason is None:
                reason = why
            kill_owned_group()

    def wall_watchdog():
        if not finished.wait(max(0, deadline - time.monotonic())):
            with lock:
                if not finished.is_set() and not reaped:
                    stop('wall_budget')
                    watchdog_events.append(dict(event='wall_watchdog_fired', pid=owned_pid,
                                                wall_seconds=time.monotonic() - start))

    def exited():
        # Non-reaping observation: preserve identity and wait4 resource evidence.
        nonlocal reaped
        with lock:
            try:
                info = os.waitid(os.P_PID, owned_pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                reaped = True  # ownership lost: numeric PID must never be signalled
                child.returncode = -255  # internal sentinel ONLY; public code stays unknown
                finished.set()
                raise
            return info is not None and info.si_pid == owned_pid

    def collect():
        nonlocal reaped, code, final_child, final_self, terminal_at, reason
        with lock:
            # Signal this private group before releasing its leader's PID.
            kill_owned_group()
        while True:
            with lock:
                try:
                    pid, status, usage = os.wait4(owned_pid, os.WNOHANG)
                except ChildProcessError:
                    reaped = True
                    child.returncode = -255  # internal sentinel; never reported as exit
                    finished.set()
                    raise
                if pid:
                    reaped = True
                    terminal_at = time.monotonic()
                    if terminal_at >= deadline and reason is None:
                        reason = 'wall_budget'
                    code = os.waitstatus_to_exitcode(status)
                    child.returncode = code
                    finished.set()
                    if pid != owned_pid:
                        raise RuntimeError('wait4 returned a different child')
                    final_child = _rusage(usage)
                    final_self = _rusage(resource.getrusage(resource.RUSAGE_SELF))
                    return
            # No lock held across a blocking reap; watchdog remains operational.
            time.sleep(.001)

    def terminal_candidate(evidence):
        # Only a narrow teardown-shaped sample can obtain bounded observation.
        # This pattern is NOT proof of exit or a diagnosis of the historic run.
        try:
            raw = evidence['stat_text']
            status = evidence['status_text']
            fields = raw.rsplit(')', 1)[1].split()
            return (evidence['pid'] == owned_pid and int(raw.split('(', 1)[0]) == owned_pid
                and len(fields) >= 22 and int(fields[20]) == 0 and int(fields[21]) == 0
                and int(fields[11]) >= 0 and int(fields[12]) >= 0
                and isinstance(status, str) and bool(status.strip())
                and not any(line.startswith('VmRSS:') for line in status.splitlines()))
        except (KeyError, ValueError, IndexError, TypeError, AttributeError):
            return False

    def observe_terminal(evidence):
        begun = time.monotonic()
        end = min(begun + .025, deadline)
        verified = False
        while time.monotonic() < end and reason is None:
            if exited():
                verified = True
                break
            time.sleep(min(.001, max(0, end - time.monotonic())))
        append(out / 'resources.jsonl', dict(evidence, event='terminal_observation',
            pid=owned_pid, role='child', child_exit_verified=verified,
            observation_seconds=time.monotonic() - begun,
            wall_seconds=time.monotonic() - start))
        return verified

    try:
        with (out / 'console.log').open('wb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            owned_pid = child.pid
            if type(owned_pid) is not int or owned_pid <= 1:
                raise RuntimeError('Invalid launched child PID')
            if os.getsid(owned_pid) != owned_pid or os.getpgid(owned_pid) != owned_pid:
                raise RuntimeError('Launch did not establish a private session and group')
            private_session_verified = True
            watchdog = threading.Thread(target=wall_watchdog, name='owned-worker-wall-watchdog', daemon=True)
            watchdog.start()
            append(out / 'resources.jsonl', dict(event='worker_launched', pid=owned_pid,
                supervisor_pid=os.getpid(), pgid=owned_pid, wall_seconds=time.monotonic() - start))
            while True:
                if time.monotonic() >= deadline:
                    stop('wall_budget')
                if exited():
                    break
                if reason is not None:
                    # SIGKILL already issued; do not fabricate accounting samples.
                    finished.wait(.001)
                    continue
                failed = False
                values = []
                for role, pid in (('child', owned_pid), ('supervisor', os.getpid())):
                    try:
                        values.append(sample(pid))
                    except (OSError, ValueError, IndexError) as exc:
                        accounting_error = repr(exc)
                        evidence = dict(getattr(exc, 'evidence', {}))
                        verified_exit = exited()
                        observed = False
                        if role == 'child' and not verified_exit and terminal_candidate(evidence):
                            observed = observe_terminal(evidence)
                            verified_exit = observed
                        event = ('exit_verified_after_teardown_shaped_sample' if observed else
                            'verified_child_exit_during_sample' if role == 'child' and verified_exit else 'accounting_failure')
                        # A supervisor read failure can NEVER be excused by child exit.
                        if event == 'accounting_failure':
                            stop('accounting_failure')
                        append(out / 'resources.jsonl', dict(evidence, event=event,
                            role=role, pid=pid, sample_error=repr(exc),
                            child_exit_verified=verified_exit, wall_seconds=time.monotonic() - start))
                        failed = True
                        break
                if failed:
                    break
                (cpu, rss), (_, own_rss) = values
                peak = max(peak, rss + own_rss)
                append(out / 'resources.jsonl', dict(event='sample', pid=owned_pid,
                    wall_seconds=time.monotonic() - start, child_cpu_seconds=cpu,
                    combined_rss_bytes=rss + own_rss))
                if time.monotonic() >= deadline:
                    stop('wall_budget')
                elif cpu >= cfg['cpu_seconds'] - 5:
                    stop('cpu_budget')
                elif rss + own_rss > cfg['rss_bytes']:
                    stop('sampled_rss_budget')
                if reason is None:
                    finished.wait(min(cfg['sampling_seconds'], max(0, deadline - time.monotonic())))
            collect()
        if watchdog is not None:
            watchdog.join()
        for event in watchdog_events:
            append(out / 'resources.jsonl', event)
        # Sum of separate high-water marks is an upper bound, not an observed
        # concurrent peak. Conservative rejection is deliberate (never relax cap).
        final_bound = final_child['peak_rss_bytes'] + final_self['peak_rss_bytes']
        if reason is None and final_child['cpu_seconds'] >= cfg['cpu_seconds'] - 5:
            reason = 'cpu_budget'
        if reason is None and final_bound > cfg['rss_bytes']:
            reason = 'final_rss_budget'
        try:
            existing = json.loads((out / 'status.json').read_text())
        except (OSError, ValueError) as exc:
            existing = dict(status='missing_or_malformed', error=repr(exc))
        if not isinstance(existing, dict):
            existing = dict(status='malformed', raw=existing)
        incomplete = bool(reason or code != 0 or existing.get('status') != 'complete')
        if incomplete:
            reason = reason or 'worker_failed_or_missing_completion'
            atomic(out / 'status.json', dict(status='incomplete', reason=reason,
                returncode=code, worker_status=existing, accounting_error=accounting_error))
        atomic(out / 'resource_summary.json', dict(
            sampled_peak_combined_rss_bytes=peak, final_child_rusage=final_child,
            final_supervisor_rusage=final_self, final_rss_upper_bound_bytes=final_bound,
            wall_seconds=terminal_at - start, returncode=code, stop_reason=reason,
            accounting_scope='single worker process including threads; not a process tree',
            child_pid=owned_pid, final_accounting_source='Linux wait4'))
        result_status = 'rejected' if incomplete else 'accepted'
        if not incomplete:
            return 0
        return 124 if reason in ('wall_budget', 'cpu_budget', 'sampled_rss_budget', 'final_rss_budget') else (code if code > 0 else 125)
    except BaseException as exc:
        if child is not None and not reaped and private_session_verified:
            stop('supervisor_exception')
            try:
                collect()
            finally:
                finished.set()
        finished.set()
        if watchdog is not None:
            watchdog.join()
        atomic(out / 'status.json', dict(status='interrupted', error=repr(exc)))
        raise
    finally:
        # Direct child reaped and our log closed before hashing. Descendant writers
        # are outside the single-process contract; see README limitations.
        atomic(out / 'RESULTS-MANIFEST.json', {str(p.relative_to(out)): digest(p)
            for p in out.rglob('*') if p.is_file() and p.name not in ('RESULTS-MANIFEST.json', 'SUPERVISOR-RESULT.json')})
        if result_status is not None:
            atomic(out / 'SUPERVISOR-RESULT.json', dict(status=result_status,
                manifest_sha256=digest(out / 'RESULTS-MANIFEST.json'), returncode=code, stop_reason=reason))
