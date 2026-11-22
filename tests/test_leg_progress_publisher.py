"""A blocked progress transport must not block sampling or rewrite sample time."""
import threading
import time

import pytest

from collection.leg_progress_publisher import ProgressPublisher


def test_blocked_transport_keeps_only_newest_unsent_progress_then_terminal():
    entered, release, published = threading.Event(), threading.Event(), threading.Event()
    sent = []

    def notify(record):
        if not sent:
            entered.set()
            assert release.wait(2.)
        sent.append(record)

    publisher = ProgressPublisher(notify)
    publisher.publish(dict(state="recording", rows=1, last_sample_ns=100, latest={"axis1": [1]}))
    assert entered.wait(1.)
    latest = dict(state="recording", rows=500, last_sample_ns=50000, latest={"axis1": [500]})
    terminal = dict(state="completed", rows=500, last_sample_ns=50000)

    def sampling_loop():
        for count in range(2, 500):
            publisher.publish(dict(state="recording", rows=count, last_sample_ns=count * 100))
        publisher.publish(latest)
        publisher.publish(terminal)
        published.set()

    sampler = threading.Thread(target=sampling_loop, daemon=True)
    sampler.start()
    try:
        assert published.wait(.5), "progress transport blocked the sampling publisher"
        before = publisher.stats()
        assert before["submitted_progress"] == 500
        assert before["superseded_progress"] == 498
        assert before["pending_progress"] is True and before["pending_terminal"] is True
        assert before["terminal_sent"] is False
        latest["last_sample_ns"] = 999999
        latest["latest"]["axis1"][0] = 999
        terminal["rows"] = 999
        with pytest.raises(RuntimeError, match="terminal already submitted"):
            publisher.publish(dict(state="recording", rows=501))
        with pytest.raises(RuntimeError, match="terminal already submitted"):
            publisher.publish(dict(state="failed", error="later terminal"))
    finally:
        release.set()
        final = publisher.close(timeout_s=1.)
    assert [record["state"] for record in sent] == ["recording", "recording", "completed"]
    assert [record["rows"] for record in sent] == [1, 500, 500]
    assert sent[1]["latest"] == {"axis1": [500]}
    assert [record["last_sample_ns"] for record in sent] == [100, 50000, 50000]
    assert final["sent_progress"] == 2 and final["terminal_sent"] is True
    assert final["sender_alive"] is False and final["closed"] is True
    assert final["sender_error"] is None


def test_terminal_failure_reason_is_delivered_unchanged_after_latest_progress():
    entered, release = threading.Event(), threading.Event()
    sent = []

    def notify(record):
        if not sent:
            entered.set()
            assert release.wait(1.)
        sent.append(record)

    publisher = ProgressPublisher(notify)
    publisher.publish(dict(state="recording", rows=2, last_sample_ns=42))
    assert entered.wait(1.)
    publisher.publish(dict(state="recording", rows=3, last_sample_ns=43))
    publisher.publish(dict(state="failed", error="axis1 fault 0x0022", rows=3))
    release.set()
    stats = publisher.close(1.)
    assert sent == [dict(state="recording", rows=2, last_sample_ns=42),
                    dict(state="recording", rows=3, last_sample_ns=43),
                    dict(state="failed", error="axis1 fault 0x0022", rows=3)]
    assert stats["terminal_sent"] is True


def test_close_times_out_boundedly_and_can_confirm_later_cleanup():
    entered, release = threading.Event(), threading.Event()
    sent = []

    def notify(record):
        entered.set()
        assert release.wait(2.)
        sent.append(record)

    publisher = ProgressPublisher(notify)
    publisher.publish(dict(state="completed", rows=0, last_sample_ns=None))
    assert entered.wait(1.)
    started = time.perf_counter()
    try:
        with pytest.raises(TimeoutError, match="cleanup unconfirmed"):
            publisher.close(timeout_s=.01)
        assert time.perf_counter() - started < .5
        stats = publisher.stats()
        assert stats["terminal_submitted"] is True and stats["terminal_sent"] is False
        assert stats["sender_alive"] is True
    finally:
        release.set()
        stats = publisher.close(timeout_s=1.)
    assert len(sent) == 1 and stats["terminal_sent"] is True
    assert stats["max_send_time_s"] >= .01
    assert publisher.close(timeout_s=0.)["sender_alive"] is False


@pytest.mark.parametrize("terminal", [False, True])
def test_transport_failure_is_preserved_by_close_and_subsequent_publish(terminal):
    failure = BrokenPipeError("parent progress reader closed")

    def notify(record):
        raise failure

    publisher = ProgressPublisher(notify)
    publisher.publish(dict(state="failed", error="original acquisition failure") if terminal
                      else dict(state="recording", rows=2))
    with pytest.raises(RuntimeError, match="BrokenPipeError: parent progress reader closed") as exc:
        publisher.close(1.)
    assert exc.value.__cause__ is failure
    stats = publisher.stats()
    assert stats["terminal_sent"] is False
    assert stats["terminal_failure_reason"] == ("original acquisition failure" if terminal else None)
    assert stats["sender_alive"] is False
    assert "parent progress reader closed" in stats["sender_error"]
    with pytest.raises(RuntimeError, match="progress sender failed") as subsequent:
        publisher.publish(dict(state="failed", error="replacement"))
    assert subsequent.value.__cause__ is failure


def test_close_without_terminal_drains_latest_progress_and_rejects_later_publish():
    sent = []
    publisher = ProgressPublisher(sent.append)
    publisher.publish(dict(state="recording", rows=12, last_sample_ns=1234))
    stats = publisher.close(1.)
    assert sent == [dict(state="recording", rows=12, last_sample_ns=1234)]
    assert stats["terminal_submitted"] is False
    assert stats["terminal_sent"] is False and stats["closed"] is True
    with pytest.raises(RuntimeError, match="closing or closed"):
        publisher.publish(dict(state="recording", rows=13))


@pytest.mark.parametrize("record", [None, [], {}, {"state": "starting"}, {"state": True}])
def test_only_progress_or_terminal_states_are_accepted(record):
    publisher = ProgressPublisher(lambda value: None)
    try:
        with pytest.raises(ValueError, match="state must be"):
            publisher.publish(record)
    finally:
        publisher.close(1.)


@pytest.mark.parametrize("timeout", [-1., float("inf"), float("nan"), True, "1"])
def test_close_requires_finite_bounded_timeout(timeout):
    publisher = ProgressPublisher(lambda value: None)
    try:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            publisher.close(timeout)
    finally:
        publisher.close(1.)
