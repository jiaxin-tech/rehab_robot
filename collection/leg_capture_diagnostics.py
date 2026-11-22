"""Bounded receive-side evidence; never a substitute for a device heartbeat."""
from collections import deque
import hashlib
import platform
import sys
import time


class CaptureDiagnostics:
    def __init__(self, config, started_ns, *, capacity=256):
        self.config = config
        self.started_ns = started_ns
        self.reads = self.accepted = self.image_changes = self.same_toggle_image_changes = 0
        self.previous_image = self.previous_toggle = self.previous_end_ns = None
        self.last_accepted_ns = None
        self.max_query_ns = self.max_between_reads_ns = self.max_publish_ns = 0
        self.max_observed_sample_age_ns = 0
        self.latest_read_error = None
        self.records = deque(maxlen=capacity)

    def read(self, start_ns, end_ns, image, toggle):
        between = None if self.previous_end_ns is None else start_ns-self.previous_end_ns
        changed = None if self.previous_image is None else image != self.previous_image
        toggle_changed = self.previous_toggle is None or toggle != self.previous_toggle
        age = None if self.last_accepted_ns is None else end_ns-self.last_accepted_ns
        self.reads += 1
        self.image_changes += int(changed is True)
        self.same_toggle_image_changes += int(changed is True and not toggle_changed)
        self.max_query_ns = max(self.max_query_ns, end_ns-start_ns)
        self.max_between_reads_ns = max(self.max_between_reads_ns, between or 0)
        self.max_observed_sample_age_ns = max(self.max_observed_sample_age_ns, age or 0)
        self.records.append(dict(read_index=self.reads, query_start_ns=start_ns, query_end_ns=end_ns,
            between_reads_ns=between, sample_age_before_accept_ns=age, input_toggle=toggle,
            image_changed=changed, toggle_changed=toggle_changed, accepted=False, image=bytes(image)))
        self.previous_image, self.previous_toggle, self.previous_end_ns = bytes(image), toggle, end_ns

    def accept(self, end_ns):
        self.accepted += 1
        self.last_accepted_ns = end_ns
        self.records[-1]["accepted"] = True

    def read_error(self, start_ns, end_ns, exc):
        between = None if self.previous_end_ns is None else start_ns-self.previous_end_ns
        self.max_query_ns = max(self.max_query_ns, end_ns-start_ns)
        self.max_between_reads_ns = max(self.max_between_reads_ns, between or 0)
        self.latest_read_error = dict(query_start_ns=start_ns, query_end_ns=end_ns,
                                     between_reads_ns=between,
                                     error=f"{type(exc).__name__}: {exc}")

    def published(self, start_ns, end_ns):
        self.max_publish_ns = max(self.max_publish_ns, end_ns-start_ns)

    def report(self, ended_ns, *, completed, error=None, cleanup_error=None):
        records = []
        for entry in self.records:
            item = {key: value for key, value in entry.items() if key != "image"}
            item.update(raw_image_hex=entry["image"].hex(), image_sha256=hashlib.sha256(entry["image"]).hexdigest())
            records.append(item)
        return dict(schema_version=1, completed=completed, error=error, cleanup_error=cleanup_error,
            started_host_ns=self.started_ns, finished_host_ns=ended_ns,
            last_accepted_host_ns=self.last_accepted_ns, successful_reads=self.reads,
            accepted_samples=self.accepted, image_changes=self.image_changes,
            same_toggle_image_changes=self.same_toggle_image_changes,
            maximum_query_duration_ms=self.max_query_ns/1e6,
            maximum_between_reads_ms=self.max_between_reads_ns/1e6,
            maximum_notify_duration_ms=self.max_publish_ns/1e6,
            maximum_sample_age_observed_before_accept_ms=self.max_observed_sample_age_ns/1e6,
            recent_read_capacity=self.records.maxlen, recent_reads=records, last_read_error=self.latest_read_error,
            thresholds={key: self.config[key] for key in
                ("no_new_sample_timeout_s", "max_progress_age_s", "poll_interval_s", "ads_timeout_ms")},
            runtime=dict(python_version=sys.version, python_executable=sys.executable,
                         platform=platform.platform(), perf_counter=time.get_clock_info("perf_counter").__dict__),
            interpretation="Host receive evidence only. Equal one-bit values cannot prove a frozen device; changing images cannot prove 500 Hz or replace toggle freshness.",
            motion_commanded=False, training_authorized=False)
