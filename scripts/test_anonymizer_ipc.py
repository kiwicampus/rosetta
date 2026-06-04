#!/usr/bin/env python3
"""Regression test for the anonymizer IPC framing.

Guards against the desync bug where a frame buffer containing an off-shape or
non-uint8 frame (e.g. a float32 image zero-pad for a missing frame) made the
wire payload the wrong size, desyncing the daemon (OverflowError / BrokenPipe).

_send_chunk must ALWAYS write exactly 16 (header) + N*H*W*C bytes in pipe mode,
coercing any mismatched frame. Run: python3 test_anonymizer_ipc.py
"""
import io
import struct
import sys

import numpy as np

from rosetta.common.anonymizer import _DaemonWorker, IPC_MODE


class _FakeStdin:
    def __init__(self):
        self.buf = bytearray()

    def write(self, b):
        self.buf += bytes(b)

    def flush(self):
        pass


class _FakeStdout:
    def __init__(self, data):
        self.io = io.BytesIO(data)

    def read(self, n):
        return self.io.read(n)


class _FakeProc:
    def __init__(self, resp):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(resp)

    def poll(self):
        return None


def _worker(resp_modified=0):
    w = _DaemonWorker.__new__(_DaemonWorker)
    w._worker_id = 0
    w._chunk_size = 128
    w._mm = w._mm_fd = w._buf_path = None
    w._mm_cap = 0
    w._proc = _FakeProc(struct.pack(">I", resp_modified))  # M modified frames
    return w


def test_mixed_dtype_and_shape_stays_in_sync():
    assert IPC_MODE == "pipe", f"test assumes pipe IPC, got {IPC_MODE!r}"
    H, W, C, N = 8, 8, 3, 5
    frames = [np.full((H, W, C), 7, np.uint8) for _ in range(N)]
    frames[2] = np.zeros((H, W, C), np.float32)   # float32 image zero-pad (the bug)
    frames[3] = np.zeros((H, W, C), np.uint8)     # plain ok frame
    w = _worker()
    out = w._send_chunk(frames)
    expected = 16 + N * H * W * C
    got = len(w._proc.stdin.buf)
    assert got == expected, f"wire payload {got} != {expected} (desync!)"
    assert struct.unpack(">4I", bytes(w._proc.stdin.buf[:16])) == (N, H, W, C)
    assert len(out) == N and all(o.shape == (H, W, C) for o in out)


def test_all_uint8_exact():
    H, W, C, N = 4, 4, 3, 3
    w = _worker()
    w._send_chunk([np.zeros((H, W, C), np.uint8) for _ in range(N)])
    assert len(w._proc.stdin.buf) == 16 + N * H * W * C


if __name__ == "__main__":
    test_mixed_dtype_and_shape_stays_in_sync()
    test_all_uint8_exact()
    print("OK: anonymizer IPC framing stays in sync (mixed dtype/shape coerced)")
    sys.exit(0)
