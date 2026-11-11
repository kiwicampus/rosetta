"""H.264 Annex B helpers: parameter set and slice header parsing, and skip frames.
A skip frame is a non reference P picture of P_Skip macroblocks: it decodes to an exact copy
of the last reference picture and leaves frame_num and the reference list untouched."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

NAL_SLICE = 1
NAL_IDR = 5
NAL_SPS = 7
NAL_PPS = 8

_HIGH_PROFILES = frozenset((100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135))

# H.264 Table 9-44: rangeTabLPS indexed by [pStateIdx][qCodIRangeIdx].
_RANGE_TAB_LPS = (
    (128, 176, 208, 240),
    (128, 167, 197, 227),
    (128, 158, 187, 216),
    (123, 150, 178, 205),
    (116, 142, 169, 195),
    (111, 135, 160, 185),
    (105, 128, 152, 175),
    (100, 122, 144, 166),
    (95, 116, 137, 158),
    (90, 110, 130, 150),
    (85, 104, 123, 142),
    (81, 99, 117, 135),
    (77, 94, 111, 128),
    (73, 89, 105, 122),
    (69, 85, 100, 116),
    (66, 80, 95, 110),
    (62, 76, 90, 104),
    (59, 72, 86, 99),
    (56, 69, 81, 94),
    (53, 65, 77, 89),
    (51, 62, 73, 85),
    (48, 59, 69, 80),
    (46, 56, 66, 76),
    (43, 53, 63, 72),
    (41, 50, 59, 69),
    (39, 48, 56, 65),
    (37, 45, 54, 62),
    (35, 43, 51, 59),
    (33, 41, 48, 56),
    (32, 39, 46, 53),
    (30, 37, 43, 50),
    (29, 35, 41, 48),
    (27, 33, 39, 45),
    (26, 31, 37, 43),
    (24, 30, 35, 41),
    (23, 28, 33, 39),
    (22, 27, 32, 37),
    (21, 26, 30, 35),
    (20, 24, 29, 33),
    (19, 23, 27, 31),
    (18, 22, 26, 30),
    (17, 21, 25, 28),
    (16, 20, 23, 27),
    (15, 19, 22, 25),
    (14, 18, 21, 24),
    (14, 17, 20, 23),
    (13, 16, 19, 22),
    (12, 15, 18, 21),
    (12, 14, 17, 20),
    (11, 14, 16, 19),
    (11, 13, 15, 18),
    (10, 12, 15, 17),
    (10, 12, 14, 16),
    (9, 11, 13, 15),
    (9, 11, 12, 14),
    (8, 10, 12, 14),
    (8, 9, 11, 13),
    (7, 9, 11, 12),
    (7, 9, 10, 12),
    (7, 8, 10, 11),
    (6, 8, 9, 11),
    (6, 7, 9, 10),
    (6, 7, 8, 9),
    (2, 2, 2, 2),
)

# H.264 Table 9-45: state transition after a least probable symbol.
_TRANS_IDX_LPS = (
    0,
    0,
    1,
    2,
    2,
    4,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    11,
    11,
    12,
    13,
    13,
    15,
    15,
    16,
    16,
    18,
    18,
    19,
    19,
    21,
    21,
    22,
    22,
    23,
    24,
    24,
    25,
    26,
    26,
    27,
    27,
    28,
    29,
    29,
    30,
    30,
    30,
    31,
    32,
    32,
    33,
    33,
    33,
    34,
    34,
    35,
    35,
    35,
    36,
    36,
    36,
    37,
    37,
    37,
    38,
    38,
    63,
)

# H.264 Table 9-13, ctxIdx 11 (mb_skip_flag of a P slice, ctxIdxInc 0): (m, n) per cabac_init_idc.
_MB_SKIP_CTX_INIT = ((23, 33), (22, 25), (29, 16))


class H264Error(ValueError):
    """The bitstream uses a feature these helpers do not handle, or is malformed."""


# =============================================================================
# NAL units
# =============================================================================


def iter_nal_units(data: bytes) -> Iterator[bytes]:
    """Yield each NAL unit (header byte included, start code excluded) of an Annex B buffer."""
    starts: List[Tuple[int, int]] = []
    i = data.find(b"\x00\x00\x01")
    while i >= 0:
        starts.append((i, i + 3))
        i = data.find(b"\x00\x00\x01", i + 3)
    for k, (_, payload_start) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(data)
        # A four byte start code leaves its leading zero on the previous unit.
        while k + 1 < len(starts) and end > payload_start and data[end - 1] == 0:
            end -= 1
        if end > payload_start:
            yield data[payload_start:end]


def nal_type(nal: bytes) -> int:
    return nal[0] & 0x1F


def is_keyframe(data: bytes) -> bool:
    """True when an Annex B access unit holds an IDR slice."""
    return any(nal_type(n) == NAL_IDR for n in iter_nal_units(data))


def unescape_rbsp(nal_payload: bytes) -> bytes:
    """Drop the emulation prevention bytes (the 0x03 of every 0x000003)."""
    out = bytearray()
    zeros = 0
    for b in nal_payload:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def escape_rbsp(rbsp: bytes) -> bytes:
    """Insert emulation prevention bytes so no start code appears inside the payload."""
    out = bytearray()
    zeros = 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


# =============================================================================
# Bit IO
# =============================================================================


class BitReader:
    """MSB first reader over an RBSP (emulation prevention already removed)."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def u(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self._pos >> 3
            if byte >= len(self._data):
                raise H264Error("read past the end of the RBSP")
            v = (v << 1) | ((self._data[byte] >> (7 - (self._pos & 7))) & 1)
            self._pos += 1
        return v

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise H264Error("Exp-Golomb code longer than 32 bits")
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


class BitWriter:
    """MSB first writer producing an RBSP."""

    def __init__(self) -> None:
        self._bits: List[int] = []

    def u(self, n: int, v: int) -> None:
        for i in range(n - 1, -1, -1):
            self._bits.append((v >> i) & 1)

    def ue(self, v: int) -> None:
        code = v + 1
        n = code.bit_length()
        self.u(n - 1, 0)
        self.u(n, code)

    def se(self, v: int) -> None:
        self.ue(2 * v - 1 if v > 0 else -2 * v)

    @property
    def byte_aligned(self) -> bool:
        return len(self._bits) % 8 == 0

    def align_zero(self) -> None:
        while not self.byte_aligned:
            self._bits.append(0)

    def to_bytes(self) -> bytes:
        if not self.byte_aligned:
            raise H264Error("RBSP is not byte aligned")
        out = bytearray()
        for i in range(0, len(self._bits), 8):
            byte = 0
            for bit in self._bits[i : i + 8]:
                byte = (byte << 1) | bit
            out.append(byte)
        return bytes(out)


# =============================================================================
# Parameter sets and slice headers
# =============================================================================


@dataclass(frozen=True)
class Sps:
    sps_id: int
    chroma_format_idc: int
    separate_colour_plane: bool
    log2_max_frame_num: int
    poc_type: int
    log2_max_poc_lsb: int
    delta_pic_order_always_zero: bool
    pic_width_in_mbs: int
    pic_height_in_map_units: int
    frame_mbs_only: bool
    mb_adaptive_frame_field: bool

    @property
    def frame_size_in_mbs(self) -> int:
        return (
            self.pic_width_in_mbs * self.pic_height_in_map_units * (1 if self.frame_mbs_only else 2)
        )


@dataclass(frozen=True)
class Pps:
    pps_id: int
    sps_id: int
    entropy_coding_mode: bool
    bottom_field_pic_order_in_frame_present: bool
    num_slice_groups: int
    weighted_pred: bool
    pic_init_qp: int
    deblocking_filter_control_present: bool
    redundant_pic_cnt_present: bool


@dataclass(frozen=True)
class SliceInfo:
    nal_ref_idc: int
    is_idr: bool
    first_mb: int
    pps_id: int
    frame_num: int
    poc_lsb: Optional[int]


def _skip_scaling_list(r: BitReader, size: int) -> None:
    last, nxt = 8, 8
    for _ in range(size):
        if nxt != 0:
            nxt = (last + r.se() + 256) % 256
        last = last if nxt == 0 else nxt


def parse_sps(nal: bytes) -> Sps:
    r = BitReader(unescape_rbsp(nal[1:]))
    profile_idc = r.u(8)
    r.u(16)  # constraint flags, reserved bits, level_idc
    sps_id = r.ue()
    chroma_format_idc, separate = 1, False
    if profile_idc in _HIGH_PROFILES:
        chroma_format_idc = r.ue()
        if chroma_format_idc == 3:
            separate = bool(r.u(1))
        r.ue()  # bit_depth_luma_minus8
        r.ue()  # bit_depth_chroma_minus8
        r.u(1)  # qpprime_y_zero_transform_bypass_flag
        if r.u(1):  # seq_scaling_matrix_present_flag
            for i in range(8 if chroma_format_idc != 3 else 12):
                if r.u(1):
                    _skip_scaling_list(r, 16 if i < 6 else 64)
    log2_max_frame_num = r.ue() + 4
    poc_type = r.ue()
    log2_max_poc_lsb, always_zero = 0, False
    if poc_type == 0:
        log2_max_poc_lsb = r.ue() + 4
    elif poc_type == 1:
        always_zero = bool(r.u(1))
        r.se()  # offset_for_non_ref_pic
        r.se()  # offset_for_top_to_bottom_field
        for _ in range(r.ue()):
            r.se()
    r.ue()  # max_num_ref_frames
    r.u(1)  # gaps_in_frame_num_value_allowed_flag
    width = r.ue() + 1
    height = r.ue() + 1
    frame_mbs_only = bool(r.u(1))
    mbaff = bool(r.u(1)) if not frame_mbs_only else False
    return Sps(
        sps_id,
        chroma_format_idc,
        separate,
        log2_max_frame_num,
        poc_type,
        log2_max_poc_lsb,
        always_zero,
        width,
        height,
        frame_mbs_only,
        mbaff,
    )


def parse_pps(nal: bytes) -> Pps:
    r = BitReader(unescape_rbsp(nal[1:]))
    pps_id = r.ue()
    sps_id = r.ue()
    entropy = bool(r.u(1))
    bottom_present = bool(r.u(1))
    num_slice_groups = r.ue() + 1
    if num_slice_groups > 1:
        raise H264Error("slice groups (FMO) are not supported")
    r.ue()  # num_ref_idx_l0_default_active_minus1
    r.ue()  # num_ref_idx_l1_default_active_minus1
    weighted_pred = bool(r.u(1))
    r.u(2)  # weighted_bipred_idc
    pic_init_qp = 26 + r.se()
    r.se()  # pic_init_qs_minus26
    r.se()  # chroma_qp_index_offset
    deblocking = bool(r.u(1))
    r.u(1)  # constrained_intra_pred_flag
    redundant = bool(r.u(1))
    return Pps(
        pps_id,
        sps_id,
        entropy,
        bottom_present,
        num_slice_groups,
        weighted_pred,
        pic_init_qp,
        deblocking,
        redundant,
    )


def parse_slice_header(
    nal: bytes, sps_by_id: Dict[int, Sps], pps_by_id: Dict[int, Pps]
) -> SliceInfo:
    """Parse the leading fields of a slice header, up to pic_order_cnt_lsb."""
    nal_ref_idc = (nal[0] >> 5) & 3
    is_idr = nal_type(nal) == NAL_IDR
    # 32 bytes hold these fields in any realistic stream, so the slice is never unescaped whole.
    r = BitReader(unescape_rbsp(nal[1:33]))
    first_mb = r.ue()
    r.ue()  # slice_type
    pps_id = r.ue()
    if pps_id not in pps_by_id:
        raise H264Error(f"slice references PPS {pps_id} before it was seen")
    if pps_by_id[pps_id].sps_id not in sps_by_id:
        raise H264Error(
            f"PPS {pps_id} references SPS {pps_by_id[pps_id].sps_id} before it was seen"
        )
    sps = sps_by_id[pps_by_id[pps_id].sps_id]
    if sps.separate_colour_plane:
        r.u(2)
    frame_num = r.u(sps.log2_max_frame_num)
    if not sps.frame_mbs_only and r.u(1):  # field_pic_flag
        raise H264Error("field coded pictures are not supported")
    if is_idr:
        r.ue()  # idr_pic_id
    poc_lsb = r.u(sps.log2_max_poc_lsb) if sps.poc_type == 0 else None
    return SliceInfo(nal_ref_idc, is_idr, first_mb, pps_id, frame_num, poc_lsb)


# =============================================================================
# CABAC encoder (H.264 clause 9.3.4), just enough for an all skip P slice
# =============================================================================


class _CabacEncoder:
    def __init__(self, out: BitWriter) -> None:
        self._out = out
        self._low = 0
        self._range = 510
        self._first = True
        self._outstanding = 0

    def _put_bit(self, b: int) -> None:
        if self._first:
            self._first = False
        else:
            self._out.u(1, b)
        while self._outstanding > 0:
            self._out.u(1, 1 - b)
            self._outstanding -= 1

    def _renorm(self) -> None:
        while self._range < 256:
            if self._low < 256:
                self._put_bit(0)
            elif self._low >= 512:
                self._low -= 512
                self._put_bit(1)
            else:
                self._low -= 256
                self._outstanding += 1
            self._range <<= 1
            self._low <<= 1

    def decision(self, ctx: List[int], bin_val: int) -> None:
        """Encode one bin; ctx is a mutable [pStateIdx, valMPS]."""
        state, mps = ctx
        lps = _RANGE_TAB_LPS[state][(self._range >> 6) & 3]
        self._range -= lps
        if bin_val != mps:
            self._low += self._range
            self._range = lps
            if state == 0:
                ctx[1] = 1 - mps
            ctx[0] = _TRANS_IDX_LPS[state]
        else:
            ctx[0] = min(state + 1, 62)
        self._renorm()

    def terminate(self, bin_val: int) -> None:
        self._range -= 2
        if bin_val:
            self._low += self._range
            # EncodeFlush: its last written bit is the rbsp_stop_one_bit.
            self._range = 2
            self._renorm()
            self._put_bit((self._low >> 9) & 1)
            self._out.u(2, ((self._low >> 7) & 3) | 1)
        else:
            self._renorm()


def _init_ctx(m: int, n: int, slice_qp: int) -> List[int]:
    pre = min(126, max(1, ((m * min(51, max(0, slice_qp))) >> 4) + n))
    return [63 - pre, 0] if pre <= 63 else [pre - 64, 1]


# =============================================================================
# Skip frames
# =============================================================================


def build_skip_frame(sps: Sps, pps: Pps, frame_num: int, poc_lsb: Optional[int]) -> bytes:
    """Annex B access unit of a non reference P picture repeating the last reference picture.

    frame_num is PrevRefFrameNum + 1; poc_lsb is only written for POC type 0."""
    if sps.mb_adaptive_frame_field:
        raise H264Error("MBAFF streams are not supported")
    w = BitWriter()
    w.ue(0)  # first_mb_in_slice
    w.ue(5)  # slice_type: P, every slice of the picture
    w.ue(pps.pps_id)
    if sps.separate_colour_plane:
        w.u(2, 0)
    w.u(sps.log2_max_frame_num, frame_num % (1 << sps.log2_max_frame_num))
    if not sps.frame_mbs_only:
        w.u(1, 0)  # field_pic_flag
    if sps.poc_type == 0:
        w.u(sps.log2_max_poc_lsb, (poc_lsb or 0) % (1 << sps.log2_max_poc_lsb))
        if pps.bottom_field_pic_order_in_frame_present:
            w.se(0)
    elif sps.poc_type == 1 and not sps.delta_pic_order_always_zero:
        w.se(0)
        if pps.bottom_field_pic_order_in_frame_present:
            w.se(0)
    if pps.redundant_pic_cnt_present:
        w.ue(0)
    w.u(1, 1)  # num_ref_idx_active_override_flag
    w.ue(0)  # num_ref_idx_l0_active_minus1: P_Skip only ever reads refIdx 0
    w.u(1, 0)  # ref_pic_list_modification_flag_l0
    if pps.weighted_pred:
        w.ue(0)  # luma_log2_weight_denom
        if sps.chroma_format_idc != 0 and not sps.separate_colour_plane:
            w.ue(0)  # chroma_log2_weight_denom
            w.u(2, 0)  # luma_weight_l0_flag, chroma_weight_l0_flag
        else:
            w.u(1, 0)
    # nal_ref_idc is 0, so there is no dec_ref_pic_marking.
    if pps.entropy_coding_mode:
        w.ue(0)  # cabac_init_idc
    w.se(0)  # slice_qp_delta
    if pps.deblocking_filter_control_present:
        w.ue(1)  # disable_deblocking_filter_idc: nothing to filter on a copy anyway

    num_mbs = sps.frame_size_in_mbs
    if pps.entropy_coding_mode:
        while not w.byte_aligned:
            w.u(1, 1)  # cabac_alignment_one_bit
        # Every neighbour is skipped, so ctxIdxInc is always 0.
        ctx = _init_ctx(*_MB_SKIP_CTX_INIT[0], pps.pic_init_qp)
        enc = _CabacEncoder(w)
        for mb in range(num_mbs):
            enc.decision(ctx, 1)  # mb_skip_flag
            enc.terminate(1 if mb == num_mbs - 1 else 0)  # end_of_slice_flag
    else:
        w.ue(num_mbs)  # mb_skip_run
        w.u(1, 1)  # rbsp_stop_one_bit
    w.align_zero()
    return b"\x00\x00\x00\x01" + bytes([NAL_SLICE]) + escape_rbsp(w.to_bytes())


@dataclass
class AccessUnit:
    """Facts about one coded picture that skip frame insertion needs."""

    is_idr: bool
    is_reference: bool
    frame_num: int
    poc_lsb: Optional[int]
    pps_id: int


@dataclass
class StreamState:
    """Parameter sets and the last picture seen in decode order; makes skip frames repeating it."""

    sps_by_id: Dict[int, Sps] = field(default_factory=dict)
    pps_by_id: Dict[int, Pps] = field(default_factory=dict)
    last: Optional[AccessUnit] = None
    repeats: int = 0

    def parse_access_unit(self, data: bytes) -> AccessUnit:
        info: Optional[SliceInfo] = None
        for nal in iter_nal_units(data):
            t = nal_type(nal)
            if t == NAL_SPS:
                sps = parse_sps(nal)
                self.sps_by_id[sps.sps_id] = sps
            elif t == NAL_PPS:
                pps = parse_pps(nal)
                self.pps_by_id[pps.pps_id] = pps
            elif t in (NAL_SLICE, NAL_IDR) and info is None:
                info = parse_slice_header(nal, self.sps_by_id, self.pps_by_id)
        if info is None:
            raise H264Error("access unit holds no slice")
        self.last = AccessUnit(
            info.is_idr, info.nal_ref_idc != 0, info.frame_num, info.poc_lsb, info.pps_id
        )
        self.repeats = 0
        return self.last

    def skip_frame(self) -> bytes:
        """A frame repeating the last parsed picture; call it again for a longer repeat."""
        last = self.last
        if last is None:
            raise H264Error("no picture to repeat yet")
        if not last.is_reference:
            # P_Skip copies the last reference picture, which is not the one on screen.
            raise H264Error("cannot repeat a non reference picture")
        pps = self.pps_by_id[last.pps_id]
        sps = self.sps_by_id[pps.sps_id]
        if sps.poc_type == 1:
            raise H264Error("POC type 1 is not supported")
        # Repeats share a POC (on type 2, consecutive ones bend the spec); decoders output them in
        # decode order regardless, because the stream has no reordering.
        poc = None if last.poc_lsb is None else last.poc_lsb + 1
        self.repeats += 1
        return build_skip_frame(sps, pps, last.frame_num + 1, poc)
