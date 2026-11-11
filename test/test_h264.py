"""Bitstream helpers and skip frames, checked against libx264 streams and libavcodec decodes."""

from fractions import Fraction
import hashlib

import av
import numpy as np
import pytest

from rosetta.common import h264


def encode_x264(cabac: int, frames: int = 14, width: int = 72, height: int = 40) -> list[bytes]:
    """Annex B access units of random frames; 40 rows is not a multiple of 16, so it crops."""
    ctx = av.CodecContext.create("libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = width, height, "yuv420p"
    ctx.time_base = Fraction(1, 30)
    ctx.options = {"x264-params": f"cabac={cabac}:keyint=6:min-keyint=6:bframes=0:scenecut=0"}
    packets = []
    rng = np.random.default_rng(cabac)
    for i in range(frames):
        frame = av.VideoFrame.from_ndarray(
            rng.integers(0, 255, (height, width, 3), dtype=np.uint8), format="rgb24"
        )
        frame.pts = i
        packets += [bytes(p) for p in ctx.encode(frame)]
    packets += [bytes(p) for p in ctx.encode(None)]
    return packets


def decode_hashes(packets: list[bytes]) -> list[str]:
    ctx = av.CodecContext.create("h264", "r")
    frames = [f for p in packets for f in ctx.decode(av.Packet(p))] + list(ctx.decode(None))
    return [hashlib.md5(f.to_ndarray(format="yuv420p").tobytes()).hexdigest() for f in frames]


def test_exp_golomb_round_trip():
    w = h264.BitWriter()
    values = [0, 1, 2, 7, 255, 3600]
    signed = [0, 1, -1, 5, -26]
    for v in values:
        w.ue(v)
    for v in signed:
        w.se(v)
    w.u(3, 0b101)
    w.align_zero()
    r = h264.BitReader(w.to_bytes())
    assert [r.ue() for _ in values] == values
    assert [r.se() for _ in signed] == signed
    assert r.u(3) == 0b101


def test_emulation_prevention_round_trip():
    rbsp = bytes([0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 9])
    escaped = h264.escape_rbsp(rbsp)
    assert b"\x00\x00\x00" not in escaped and b"\x00\x00\x01" not in escaped
    assert h264.unescape_rbsp(escaped) == rbsp


def test_iter_nal_units_strips_both_start_code_lengths():
    data = b"\x00\x00\x00\x01\x67\xaa\x00\x00\x01\x68\xbb\x00\x00\x00\x01\x65\xcc"
    assert list(h264.iter_nal_units(data)) == [b"\x67\xaa", b"\x68\xbb", b"\x65\xcc"]
    assert h264.is_keyframe(data)
    assert not h264.is_keyframe(b"\x00\x00\x00\x01\x41\xcc")


@pytest.mark.parametrize("cabac", [0, 1])
def test_skip_frames_repeat_the_previous_picture(cabac):
    packets = encode_x264(cabac)
    reference = decode_hashes(packets)
    state = h264.StreamState()
    out, expected = [], []
    for i, data in enumerate(packets):
        state.parse_access_unit(data)
        out.append(data)
        expected.append(reference[i])
        # Up to three in a row, including right after the IDRs at 0 and 6.
        for _ in range(i % 4):
            out.append(state.skip_frame())
            expected.append(reference[i])
    assert decode_hashes(out) == expected


def test_skip_frame_header_round_trips_for_poc_type_0():
    sps = h264.Sps(0, 1, False, 8, 0, 8, False, 80, 45, True, False)
    pps = h264.Pps(0, 0, True, False, 1, False, 26, True, False)
    data = h264.build_skip_frame(sps, pps, frame_num=260, poc_lsb=255)
    (nal,) = h264.iter_nal_units(data)
    assert nal[0] == 0x01  # nal_ref_idc 0, coded slice
    info = h264.parse_slice_header(nal, {0: sps}, {0: pps})
    assert (info.nal_ref_idc, info.is_idr, info.first_mb) == (0, False, 0)
    assert info.frame_num == 260 % 256
    assert info.poc_lsb == 255


def test_poc_type_1_is_refused():
    sps = h264.Sps(0, 1, False, 8, 1, 0, False, 5, 3, True, False)
    pps = h264.Pps(0, 0, True, False, 1, False, 26, True, False)
    state = h264.StreamState({0: sps}, {0: pps}, h264.AccessUnit(True, True, 0, None, 0))
    with pytest.raises(h264.H264Error, match="POC type 1"):
        state.skip_frame()


def test_skip_frame_refuses_to_repeat_a_non_reference_picture():
    state = h264.StreamState()
    with pytest.raises(h264.H264Error, match="no picture"):
        state.skip_frame()
    state.parse_access_unit(encode_x264(1, frames=1)[0]).is_reference = False
    with pytest.raises(h264.H264Error, match="non reference"):
        state.skip_frame()
