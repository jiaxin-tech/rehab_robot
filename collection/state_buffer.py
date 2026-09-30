"""Bounded retention of immutable state frames during explicit acquisition.

This queue retains frames already accepted by the host adapter. It does not
make claims about controller packets or the native SDK receive queue.
"""

from __future__ import annotations

from collections import deque
import threading

from collection.state import KinematicStateFrame


class StateBufferOverflow(RuntimeError):
    """The current capture cannot claim complete host-frame retention."""


class StateFrameBuffer:
    """Single-consumer FIFO; overflow stays visible for this buffer's lifetime.

    No item is silently evicted. A full queue rejects the new publication and
    latches an error, which subsequent publication and drain calls also raise.
    """

    def __init__(self, capacity: int = 1024) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("state buffer capacity must be a positive integer")
        self.capacity = capacity
        self._frames: deque[KinematicStateFrame] = deque()
        self._lock = threading.Lock()
        self._overflow_reason: str | None = None

    def publish(self, frame: KinematicStateFrame) -> None:
        if not isinstance(frame, KinematicStateFrame):
            raise TypeError("state buffer requires KinematicStateFrame")
        with self._lock:
            if self._overflow_reason is not None:
                raise StateBufferOverflow(self._overflow_reason)
            if len(self._frames) >= self.capacity:
                self._overflow_reason = (
                    f"state_frame_buffer_overflow:capacity={self.capacity}:"
                    f"rejected_sequence={frame.sequence_id}"
                )
                raise StateBufferOverflow(self._overflow_reason)
            self._frames.append(frame)

    def drain(self) -> tuple[KinematicStateFrame, ...]:
        with self._lock:
            if self._overflow_reason is not None:
                raise StateBufferOverflow(self._overflow_reason)
            frames = tuple(self._frames)
            self._frames.clear()
            return frames


__all__ = ["StateBufferOverflow", "StateFrameBuffer"]
