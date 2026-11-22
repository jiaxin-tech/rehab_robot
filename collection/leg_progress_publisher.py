"""Publish latest-only acquisition progress without blocking the sampling loop.

Only progress notifications are coalesced; acquired samples and their CSV writer
do not pass through this publisher.  At most one unsent recording snapshot and
one terminal notification are retained, in addition to the in-flight snapshot.
"""
from __future__ import annotations

import copy
import math
from numbers import Real
import threading
import time
from typing import Callable


class ProgressPublisher:
    """Send copied notifications on a daemon thread; never refresh sample time.

    ``publish`` performs no transport I/O.  It replaces an unsent ``recording``
    snapshot with the newest one.  A single ``completed`` or ``failed`` snapshot
    seals the stream: already pending progress precedes it and later publishes
    are rejected.  ``close`` drains the stream and waits only up to its timeout.
    A blocked transport cannot be cancelled by this class, so its thread is a
    daemon and a timeout must be treated as an unconfirmed publication cleanup.
    """

    def __init__(self, notify: Callable[[dict], object]):
        if not callable(notify):
            raise TypeError("progress notify must be callable")
        self._notify = notify
        self._condition = threading.Condition()
        self._pending_progress = None
        self._pending_terminal = None
        self._terminal_submitted = False
        self._terminal_sent = False
        self._terminal_state = None
        self._terminal_failure_reason = None
        self._closing = False
        self._closed = False
        self._error = None
        self._sending_state = None
        self._submitted_progress = 0
        self._superseded_progress = 0
        self._sent_progress = 0
        self._max_send_time_s = 0.
        self._thread = threading.Thread(
            target=self._send, name="leg-progress-publisher", daemon=True)
        self._thread.start()

    def _raise_sender_error(self):
        if self._error is not None:
            raise RuntimeError(
                f"leg progress sender failed: {type(self._error).__name__}: {self._error}"
            ) from self._error

    def publish(self, record: dict) -> None:
        """Queue a snapshot, superseding only an unsent recording notification."""
        if not isinstance(record, dict) or record.get("state") not in (
                "recording", "completed", "failed"):
            raise ValueError("progress state must be recording, completed or failed")
        # Snapshot nested axis state as well as its outer container.  No sender
        # may observe later mutations by the acquisition loop or caller.
        snapshot = copy.deepcopy(record)
        with self._condition:
            self._raise_sender_error()
            if self._terminal_submitted:
                raise RuntimeError("leg progress terminal already submitted; stream is sealed")
            if self._closing or self._closed:
                raise RuntimeError("leg progress publisher is closing or closed")
            if snapshot["state"] == "recording":
                self._submitted_progress += 1
                if self._pending_progress is not None:
                    self._superseded_progress += 1
                self._pending_progress = snapshot
            else:
                self._terminal_submitted = True
                self._terminal_state = snapshot["state"]
                self._terminal_failure_reason = (copy.deepcopy(snapshot.get("error"))
                                                 if snapshot["state"] == "failed" else None)
                self._pending_terminal = snapshot
                self._closing = True
            self._condition.notify()

    def _send(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending_progress is not None
                        or self._pending_terminal is not None or self._closing)
                    if self._pending_progress is not None:
                        record = self._pending_progress
                        self._pending_progress = None
                    elif self._pending_terminal is not None:
                        record = self._pending_terminal
                        self._pending_terminal = None
                    else:
                        return
                    self._sending_state = record["state"]
                started = time.perf_counter()
                try:
                    # Intentionally outside the condition: stdout/flush may
                    # block forever while acquisition keeps sampling to CSV.
                    self._notify(record)
                finally:
                    duration = max(0., time.perf_counter() - started)
                    with self._condition:
                        self._max_send_time_s = max(self._max_send_time_s, duration)
                with self._condition:
                    self._sending_state = None
                    if record["state"] == "recording":
                        self._sent_progress += 1
                    else:
                        self._terminal_sent = True
                        return
        except BaseException as exc:
            # Preserve the original transport failure for both publish and
            # close; it must not masquerade as a successful terminal delivery.
            with self._condition:
                self._error = exc
        finally:
            with self._condition:
                self._sending_state = None
                self._closed = True
                self._condition.notify_all()

    def stats(self) -> dict:
        """Return transport diagnostics, not acquisition-health timestamps."""
        with self._condition:
            return dict(
                submitted_progress=self._submitted_progress,
                superseded_progress=self._superseded_progress,
                sent_progress=self._sent_progress,
                terminal_submitted=self._terminal_submitted,
                terminal_sent=self._terminal_sent,
                terminal_state=self._terminal_state,
                terminal_failure_reason=copy.deepcopy(self._terminal_failure_reason),
                pending_progress=self._pending_progress is not None,
                pending_terminal=self._pending_terminal is not None,
                sending_state=self._sending_state,
                max_send_time_s=self._max_send_time_s,
                closed=self._closed,
                sender_alive=self._thread.is_alive(),
                sender_error=(f"{type(self._error).__name__}: {self._error}"
                              if self._error is not None else None),
            )

    def close(self, timeout_s: float = 1.) -> dict:
        """Drain progress/terminal and join within a finite, bounded duration.

        With no terminal supplied, closing just drains the latest progress.
        Return means notify returned for every retained message and the sender
        exited.  It does not prove that an external receiver processed stdout.
        """
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, Real)
                or not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("publisher close timeout_s must be finite and nonnegative")
        if threading.current_thread() is self._thread:
            raise RuntimeError("progress sender cannot close itself")
        with self._condition:
            self._closing = True
            self._condition.notify()
        self._thread.join(timeout=float(timeout_s))
        if self._thread.is_alive():
            raise TimeoutError("leg progress publication cleanup unconfirmed: sender did not exit before timeout")
        with self._condition:
            self._raise_sender_error()
            if self._terminal_submitted and not self._terminal_sent:
                raise RuntimeError("leg progress terminal delivery unconfirmed")
        return self.stats()
