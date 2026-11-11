"""The H.264 camera track, driven through its interface with libx264 packets in a tiny MCAP."""

import hashlib
from types import SimpleNamespace

import av
import numpy as np
import pytest

from rosetta.common import h264
from rosetta.common.ros2_utils import stamp_from_header_ns
from rosetta.h264_passthrough import MSG_TYPE, StartGate, write_episode_video
from test_h264 import decode_hashes, encode_x264

TOPIC = "/cam/color_h264"
SIZE = (40, 72)
R = "skip"


def write_bag(path, packets, fmt="h264"):
    """One CompressedVideo message per packet; message i is logged at (i + 1) ms."""
    from mcap.writer import Writer
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.utilities import get_message

    msg_type = get_message(MSG_TYPE)
    with open(path, "wb") as fh:
        writer = Writer(fh)
        writer.start(profile="ros2", library="test")
        schema = writer.register_schema(name=MSG_TYPE, encoding="ros2msg", data=b"")
        channel = writer.register_channel(topic=TOPIC, message_encoding="cdr", schema_id=schema)
        for i, data in enumerate(packets):
            msg = msg_type()
            msg.data = data
            msg.format = fmt
            t = log_time(i)
            writer.add_message(channel, log_time=t, data=serialize_message(msg), publish_time=t)
        writer.finish()
    return str(path)


def log_time(index):
    return (index + 1) * 1_000_000


def avcc_nals(sample):
    out, i = [], 0
    while i < len(sample):
        n = int.from_bytes(sample[i : i + 4], "big")
        out.append(sample[i + 4 : i + 4 + n])
        i += 4 + n
    return out


def shown(mp4, packets):
    """Per MP4 sample, the source packet index it carries, or R for a skip frame."""
    by_nals = {tuple(h264.iter_nal_units(p)): i for i, p in enumerate(packets)}
    out = []
    with av.open(str(mp4)) as container:
        for sample in container.demux(container.streams.video[0]):
            if sample.size:
                nals = tuple(avcc_nals(bytes(sample)))
                out.append(by_nals.get(nals, R))
    return out


@pytest.fixture(scope="module")
def packets():
    return encode_x264(cabac=1)  # IDRs at 0, 6 and 12


@pytest.fixture
def bag(tmp_path, packets):
    return write_bag(tmp_path / "bag.mcap", packets)


@pytest.mark.parametrize(
    "hold, expected",
    [
        ([0, 1, 2, 3], [0, 1, 2, 3]),
        # A tick with no new message is a skip frame, never a repeated packet.
        ([0, 1, 1, 2], [0, 1, R, 2]),
        # Two messages in one tick: fall one behind instead of dropping a P frame, then catch up.
        ([0, 2, 3, 3, 4], [0, 1, 2, 3, 4]),
        # Falling behind ends at an IDR: the frames before it may be dropped.
        ([3, 5, 7, 8], [0, 1, 6, 7]),
        # The first row starts on the IDR at or before what hold picked.
        ([8, 9, 10], [6, 7, 8]),
    ],
)
def test_rows_carry_packets_in_order(tmp_path, bag, packets, hold, expected):
    out = tmp_path / "cam.mp4"
    video = write_episode_video(bag, TOPIC, [log_time(h) for h in hold], str(out), 30, SIZE)
    assert shown(out, packets) == expected
    assert (video.rows, video.skip_frames) == (len(hold), expected.count(R))


def test_mp4_decodes_to_the_original_pictures_with_the_streams_own_avcc(tmp_path, bag, packets):
    out = tmp_path / "cam.mp4"
    hold = [0, 1, 1, 3, 3, 4, 7, 7, 8, 9]
    video = write_episode_video(bag, TOPIC, [log_time(h) for h in hold], str(out), 30, SIZE)
    assert video.max_lag == 1

    reference = decode_hashes(packets)
    rows = shown(out, packets)
    sources = [s if s != R else rows[i - 1] for i, s in enumerate(rows)]
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        avcc = bytes(stream.codec_context.extradata)
        decoded = [
            hashlib.md5(f.to_ndarray(format="yuv420p").tobytes()).hexdigest()
            for f in container.decode(stream)
        ]
    assert decoded == [reference[s] for s in sources]
    sps, pps = [n for n in h264.iter_nal_units(packets[0]) if h264.nal_type(n) in (7, 8)]
    sps_end = 8 + int.from_bytes(avcc[6:8], "big")
    assert avcc[8:sps_end] == sps
    assert (
        avcc[sps_end + 3 : sps_end + 3 + int.from_bytes(avcc[sps_end + 1 : sps_end + 3], "big")]
        == pps
    )


def test_stats_decode_the_idr_nearest_each_stats_row(tmp_path, bag):
    hold = list(range(14))
    video = write_episode_video(
        bag, TOPIC, [log_time(h) for h in hold], str(tmp_path / "c.mp4"), 30, SIZE, [1, 2, 5, 13]
    )
    # Rows 1 and 2 map to the IDR at row 0, row 5 to the one at 6, row 13 to the one at 12.
    assert len(video.stats_images) == 3
    assert all(img.shape == (3, *SIZE) and img.dtype.name == "uint8" for img in video.stats_images)


@pytest.mark.parametrize(
    "packets_fmt, hold, size, stats, match",
    [
        ("h265", [0], SIZE, (), "only h264"),
        ("h264", [2, 3], SIZE, (), None),  # resolved below: a bag without IDRs
        ("h264", [3, 2], SIZE, (), "backwards"),
        ("h264", [0, 1], (48, 72), [0], "cannot resize"),
    ],
)
def test_bad_input_is_rejected(tmp_path, packets, packets_fmt, hold, size, stats, match):
    if match is None:
        packets, match = packets[1:6], "no IDR precedes"
    bag = write_bag(tmp_path / "bag.mcap", packets, fmt=packets_fmt)
    with pytest.raises(ValueError, match=match):
        write_episode_video(
            bag, TOPIC, [log_time(h) for h in hold], str(tmp_path / "c.mp4"), 30, size, stats
        )


def test_unknown_log_time_is_rejected(tmp_path, bag):
    with pytest.raises(ValueError, match="exactly one message"):
        write_episode_video(bag, TOPIC, [log_time(0) + 1], str(tmp_path / "c.mp4"), 30, SIZE)


def test_start_gate_counts_from_the_first_idr_and_finds_the_start_row(packets):
    idr, p_frame = packets[0], packets[1]
    gate = StartGate()
    assert not gate.accept(5, p_frame)
    assert gate.accept(10, idr)
    assert gate.accept(15, p_frame)
    assert gate.first_start_row([10, 20, 30, 40]) == 0
    gate.accept(25, idr)
    # The IDR at 25 arrived in the tick of row 2 even though the row holds a later message.
    assert gate.first_start_row([12, 20, 30, 40]) == 2
    with pytest.raises(ValueError, match="no IDR"):
        gate.first_start_row([30, 40])
    # A row without a message (zeros when a stamp ran ahead of the tick) is skipped, not a start.
    assert gate.first_start_row([None, np.zeros(3), 20, 30]) == 3


def test_foxglove_timestamp_counts_as_a_header_stamp():
    msg = SimpleNamespace(timestamp=SimpleNamespace(sec=3, nanosec=7))
    assert stamp_from_header_ns(msg) == 3_000_000_007
