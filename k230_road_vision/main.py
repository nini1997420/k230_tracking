# K230 Road Vision - final single-file hardware debug version
# Upload this file to K230 as /sdcard/main.py.
# -*- coding: utf-8 -*-

import time
import os
import gc
import sys
import math

BUILD_ID = "K230_ROAD_DUAL_BAND_FULL_WIDTH_20260723_06"

# ---- 板端硬件库（仅在 K230 上可用） ----
try:
    from machine import FPIOA, UART, Pin
    from media.sensor import *
    from media.display import *
    from media.media import *
    ON_K230 = True
except ImportError:
    ON_K230 = False
    print("WARN: not on K230 hardware, running in simulation mode")


# ============================================================================
# road_config.py — 集中配置模块
# ============================================================================

# 摄像头与图像
SENSOR_ID = 2
CAM_INPUT_WIDTH = 1280
CAM_INPUT_HEIGHT = 960
CAM_FPS = 90

DETECT_WIDTH = 320
DETECT_HEIGHT = 240
DETECT_PIXFMT = "RGB888"

# 道路 ROI
ROI_TOP = 30
ROI_BOTTOM = 220
ROI_LEFT = 0
ROI_RIGHT = DETECT_WIDTH

# Two narrow processing bands for the fixed real-camera view.
CONTROL_BAND_TOP = 148
CONTROL_BAND_BOTTOM = 177
CONTROL_SAMPLE_ROWS = 7
LOOKAHEAD_BAND_TOP = 68
LOOKAHEAD_BAND_BOTTOM = 101
LOOKAHEAD_SAMPLE_ROWS = 5

# 黑线检测阈值
# RGB colour spread rejects the red centre tape.
LINE_GRAY_THRESH = 105
LINE_MAX_COLOR_SPREAD = 45
LINE_ALWAYS_DARK_MAX = 70
CANNY_LOW = 30
CANNY_HIGH = 90
MORPH_KERNEL = 3

# 道路几何 — 采样行
NUM_SAMPLE_ROWS = CONTROL_SAMPLE_ROWS + LOOKAHEAD_SAMPLE_ROWS
MIN_LINE_WIDTH = 4
MAX_LINE_WIDTH = 40
EXPECTED_ROAD_WIDTH_MIN = 30
EXPECTED_ROAD_WIDTH_MAX = DETECT_WIDTH - 1
BOUNDARY_SEARCH_MARGIN = 10

WIDE_JUNCTION_MIN_WIDTH = 55

# 道路几何 — 航向偏差计算
HEADING_NEAR_ROWS = [0, 1, 2, 3]
HEADING_FAR_ROWS = [8, 9, 10, 11]

# 单边界降级
SINGLE_BOUNDARY_MAX_FRAMES = 15
SINGLE_BOUNDARY_MAX_MS = 500
ROAD_WIDTH_SMOOTH_ALPHA = 0.3

# 可信度评估权重
CONF_WEIGHT_VALID_POINTS = 0.30
CONF_WEIGHT_FIT_RESIDUAL = 0.25
CONF_WEIGHT_WIDTH = 0.20
CONF_WEIGHT_CONTINUITY = 0.15
CONF_WEIGHT_STABILITY = 0.10

CONF_HIGH_THRESH = 70
CONF_LOW_THRESH = 30

# 路口检测
JUNCTION_WIDTH_INCREASE_RATIO = 1.5
JUNCTION_CONFIRM_FRAMES = 5
JUNCTION_COOLDOWN_FRAMES = 20
JUNCTION_DISTANCE_NEAR_PX = 40
JUNCTION_DISTANCE_MID_PX = 80
# Real-road log showed simultaneous left/right/intersection false triggers.
# Keep the detector visible on LCD, but do not transmit junction decisions
# until the camera pose and thresholds have been calibrated.
ENABLE_JUNCTION_OUTPUT = False

# 转弯/重捕获
REACQUIRE_CONFIRM_FRAMES = 8
TURNING_SEND_INTERVAL_MS = 100

# UART 通信
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_ID = 3
UART_BAUD = 460800

FRAME_SEND_INTERVAL_MS = 20
FRAME_SEND_MIN_MS = 15

# 显示
# Physical LCD debug mode. IDE JPEG remains off to limit the frame-rate cost.
ENABLE_DISPLAY = True
DISPLAY_TO_IDE = False
DISPLAY_QUALITY = 35
DISPLAY_EVERY_N = 10

# 内存与性能
GC_INTERVAL_FRAMES = 100
GC_FREE_THRESH = 120000
IDLE_SLEEP_MS = 50


# ============================================================================
# vision_protocol.py — 通信协议层 (full version with field validation)
# ============================================================================

PROTOCOL_VERSION = 0x02
MSG_TYPE_ROAD_VISION = 0x10
FRAME_SIZE = 24
CRC_DATA_SIZE = 22

HEADER0 = 0xAA
HEADER1 = 0x55

INVALID_S16 = 0x8000
INVALID_U16 = 0xFFFF

MODE_IDLE = 0
MODE_TRACK = 1
MODE_TURNING = 2
MODE_REACQUIRE = 3
MODE_FAULT = 4
MODE_NUMBER = 5

FLAG_VISION_VALID = 0x01
FLAG_DEGRADED = 0x02
FLAG_LEFT_VALID = 0x04
FLAG_RIGHT_VALID = 0x08
FLAG_LEFT_BRANCH = 0x10
FLAG_RIGHT_BRANCH = 0x20
FLAG_INTERSECTION = 0x40

ANOMALY_BLUR = 0x01
ANOMALY_OVEREXPOSED = 0x02
ANOMALY_UNDEREXPOSED = 0x04
ANOMALY_EDGE_NOISE = 0x08

VALID_MODES = {MODE_IDLE, MODE_TRACK, MODE_TURNING, MODE_REACQUIRE, MODE_FAULT, MODE_NUMBER}
VALID_STAGES = {0, 1, 2, 3, 4}
VALID_DISTANCES = {0, 1, 2, 3}
VALID_FLAGS_MASK = 0x7F
VALID_ANOMALY_MASK = 0x0F


def _make_crc16_ccitt_false_table():
    table = []
    for value in range(256):
        crc = value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


CRC16_TABLE = _make_crc16_ccitt_false_table()


def crc16_ccitt_false(data, length):
    crc = 0xFFFF
    for index in range(int(length)):
        crc = ((crc << 8) ^ CRC16_TABLE[((crc >> 8) ^ data[index]) & 0xFF]) & 0xFFFF
    return crc


def _write_le16(buf, offset, value):
    value = int(value) & 0xFFFF
    buf[offset] = value & 0xFF
    buf[offset + 1] = (value >> 8) & 0xFF


def _write_le32(buf, offset, value):
    value = int(value) & 0xFFFFFFFF
    buf[offset] = value & 0xFF
    buf[offset + 1] = (value >> 8) & 0xFF
    buf[offset + 2] = (value >> 16) & 0xFF
    buf[offset + 3] = (value >> 24) & 0xFF


def _read_le16(buf, offset):
    return (buf[offset] | (buf[offset + 1] << 8)) & 0xFFFF


def _read_le32(buf, offset):
    return (buf[offset] | (buf[offset + 1] << 8) |
            (buf[offset + 2] << 16) | (buf[offset + 3] << 24)) & 0xFFFFFFFF


class RoadFrameBuilder:
    def __init__(self):
        self.sequence = 0
        self.buf = bytearray(FRAME_SIZE)

    def build(self, timestamp_ms, mode, flags, lateral_error_raw, heading_error_raw,
              road_width_raw, junction_stage, junction_distance,
              confidence, anomaly_flags) -> bytes:
        buf = self.buf
        buf[0] = HEADER0
        buf[1] = HEADER1
        buf[2] = PROTOCOL_VERSION
        buf[3] = MSG_TYPE_ROAD_VISION
        _write_le16(buf, 4, self.sequence)
        _write_le32(buf, 6, int(timestamp_ms) & 0xFFFFFFFF)
        buf[10] = int(mode) & 0xFF
        buf[11] = int(flags) & 0xFF
        _write_le16(buf, 12, int(lateral_error_raw) & 0xFFFF)
        _write_le16(buf, 14, int(heading_error_raw) & 0xFFFF)
        _write_le16(buf, 16, int(road_width_raw) & 0xFFFF)
        buf[18] = int(junction_stage) & 0xFF
        buf[19] = int(junction_distance) & 0xFF
        buf[20] = int(confidence) & 0xFF
        buf[21] = int(anomaly_flags) & 0xFF
        crc = crc16_ccitt_false(buf, CRC_DATA_SIZE)
        _write_le16(buf, 22, crc)
        self.sequence = (self.sequence + 1) & 0xFFFF
        return bytes(buf)

    def frame_size(self):
        return FRAME_SIZE


class RoadFrameDecoder:
    def __init__(self):
        self.last_seq = -1
        self.seq_wrap_count = 0

    def decode(self, frame: bytes):
        if len(frame) != FRAME_SIZE:
            return None, "length_error: expected %d, got %d" % (FRAME_SIZE, len(frame))
        crc_rx = _read_le16(frame, 22)
        crc_calc = crc16_ccitt_false(frame, CRC_DATA_SIZE)
        if crc_rx != crc_calc:
            return None, "crc_error: rx=0x%04X calc=0x%04X" % (crc_rx, crc_calc)
        if frame[0] != HEADER0 or frame[1] != HEADER1:
            return None, "header_error"
        if frame[2] != PROTOCOL_VERSION:
            return None, "version_error: 0x%02X" % frame[2]
        if frame[3] != MSG_TYPE_ROAD_VISION:
            return None, "msg_type_error: 0x%02X" % frame[3]

        mode = frame[10]
        flags = frame[11]
        lateral_raw = _read_le16(frame, 12)
        heading_raw = _read_le16(frame, 14)
        width_raw = _read_le16(frame, 16)
        stage = frame[18]
        distance = frame[19]
        confidence = frame[20]
        anomaly = frame[21]

        if mode not in VALID_MODES:
            return None, "field_error: illegal mode 0x%02X" % mode
        if stage not in VALID_STAGES:
            return None, "field_error: illegal junction_stage %d" % stage
        if distance not in VALID_DISTANCES:
            return None, "field_error: illegal junction_distance %d" % distance
        if confidence > 100:
            return None, "field_error: confidence %d > 100" % confidence
        if flags & ~VALID_FLAGS_MASK:
            return None, "field_error: flags reserved bits set (0x%02X)" % flags
        if anomaly & ~VALID_ANOMALY_MASK:
            return None, "field_error: anomaly reserved bits set (0x%02X)" % anomaly

        vision_valid = bool(flags & FLAG_VISION_VALID)
        if vision_valid:
            if lateral_raw == INVALID_S16:
                return None, "field_error: vision_valid=1 but lateral_error=INVALID"
            if heading_raw == INVALID_S16:
                return None, "field_error: vision_valid=1 but heading_error=INVALID"
            if width_raw == INVALID_U16:
                return None, "field_error: vision_valid=1 but road_width=INVALID"
            if confidence == 0:
                return None, "field_error: vision_valid=1 but confidence=0"

        seq = _read_le16(frame, 4)
        seq_ok, seq_err = self._check_sequence(seq)
        if not seq_ok:
            return None, seq_err

        self.last_seq = seq
        return {
            "sequence": seq,
            "timestamp_ms": _read_le32(frame, 6),
            "mode": mode,
            "flags": flags,
            "vision_valid": vision_valid,
            "degraded": bool(flags & FLAG_DEGRADED),
            "left_valid": bool(flags & FLAG_LEFT_VALID),
            "right_valid": bool(flags & FLAG_RIGHT_VALID),
            "left_branch": bool(flags & FLAG_LEFT_BRANCH),
            "right_branch": bool(flags & FLAG_RIGHT_BRANCH),
            "intersection_candidate": bool(flags & FLAG_INTERSECTION),
            "lateral_error_0_1mm": self._to_s16(lateral_raw),
            "heading_error_0_01deg": self._to_s16(heading_raw),
            "road_width_0_1mm": width_raw,
            "junction_stage": stage,
            "junction_distance": distance,
            "confidence": confidence,
            "anomaly_flags": anomaly,
        }, None

    def _check_sequence(self, seq):
        if self.last_seq < 0:
            return True, None
        delta = (seq - self.last_seq) & 0xFFFF
        if self.last_seq > 0xF000 and seq < 0x0FFF:
            self.seq_wrap_count += 1
            return True, None
        if delta > 0x8000:
            return False, "sequence_error: rewind last=%d current=%d" % (self.last_seq, seq)
        if delta == 0:
            return False, "sequence_error: duplicate seq=%d" % seq
        return True, None

    def reset_sequence(self):
        self.last_seq = -1
        self.seq_wrap_count = 0

    @staticmethod
    def decode_validate(frame: bytes) -> bool:
        if len(frame) != FRAME_SIZE:
            return False
        if frame[0] != HEADER0 or frame[1] != HEADER1:
            return False
        if frame[2] != PROTOCOL_VERSION:
            return False
        if frame[3] != MSG_TYPE_ROAD_VISION:
            return False
        crc_rx = _read_le16(frame, 22)
        crc_calc = crc16_ccitt_false(frame, CRC_DATA_SIZE)
        return crc_rx == crc_calc

    @staticmethod
    def _to_s16(val):
        val = val & 0xFFFF
        if val >= 0x8000:
            return val - 0x10000
        return val


# ============================================================================
# road_geometry.py — 道路几何层
# ============================================================================

class VisionState:
    NORMAL = 0
    DEGRADED = 1
    INVALID = 2

    @staticmethod
    def name(state):
        return {0: "NORMAL", 1: "DEGRADED", 2: "INVALID"}.get(state, "???")


class BoundaryResult:
    def __init__(self):
        self.left_points = []
        self.right_points = []
        self.left_valid = False
        self.right_valid = False
        self.center_points = []
        self.num_valid_rows = 0
        self.sample_y = []
        self.control_center_points = []
        self.lookahead_center_points = []
        self.all_segments = []     # 每行所有黑线段 [(start, end), ...] per row


class GeometryResult:
    def __init__(self):
        self.lateral_error = 0.0
        self.heading_error = 0.0
        self.road_width = 0.0
        self.left_valid = False
        self.right_valid = False
        self.vision_state = VisionState.NORMAL
        self.confidence = 0
        self.degraded = False
        self.num_valid_rows = 0
        self.fit_residual = 0.0


def _gray_pixel(image_array, y, x, ndim=None):
    """
    Read one grayscale value from a PC list, K230 grayscale ndarray, or
    K230 RGB888 ndarray.  Only the configured sample rows call this helper.
    """
    if ndim is None:
        try:
            ndim = len(image_array.shape)
        except Exception:
            ndim = 0

    if ndim == 2:
        return int(image_array[y, x]) & 0xFF

    if ndim >= 3:
        channels = int(image_array.shape[2])
        if channels <= 1:
            return int(image_array[y, x, 0]) & 0xFF
        r = int(image_array[y, x, 0]) & 0xFF
        g = int(image_array[y, x, 1]) & 0xFF
        b = int(image_array[y, x, 2]) & 0xFF
        return (77 * r + 150 * g + 29 * b) >> 8

    value = image_array[y][x]
    if isinstance(value, (tuple, list)):
        if len(value) >= 3:
            r = int(value[0]) & 0xFF
            g = int(value[1]) & 0xFF
            b = int(value[2]) & 0xFF
            return (77 * r + 150 * g + 29 * b) >> 8
        if value:
            return int(value[0]) & 0xFF
        return 0
    return int(value) & 0xFF


def _is_black_pixel(image_array, y, x, ndim=None):
    """Detect dark neutral tape while rejecting chromatic red tape."""
    if ndim is None:
        try:
            ndim = len(image_array.shape)
        except Exception:
            ndim = 0

    if ndim >= 3 and int(image_array.shape[2]) >= 3:
        r = int(image_array[y, x, 0]) & 0xFF
        g = int(image_array[y, x, 1]) & 0xFF
        b = int(image_array[y, x, 2]) & 0xFF
        return _is_black_rgb(r, g, b)

    value = image_array[y][x] if ndim == 0 else image_array[y, x]
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        r = int(value[0]) & 0xFF
        g = int(value[1]) & 0xFF
        b = int(value[2]) & 0xFF
        return _is_black_rgb(r, g, b)
    return (int(value) & 0xFF) < LINE_GRAY_THRESH


def _is_black_rgb(r, g, b):
    """Fast RGB predicate for values already copied out of the ndarray."""
    hi = max(r, g, b)
    if hi <= LINE_ALWAYS_DARK_MAX:
        return True
    lo = min(r, g, b)
    if hi - lo > LINE_MAX_COLOR_SPREAD:
        return False
    return ((77 * r + 150 * g + 29 * b) >> 8) < LINE_GRAY_THRESH


class RoadBoundaryExtractor:
    def __init__(self, detect_w=320, detect_h=240,
                 roi_top=ROI_TOP, roi_bottom=ROI_BOTTOM,
                 roi_left=ROI_LEFT, roi_right=ROI_RIGHT,
                 gray_thresh=80,
                 num_rows=NUM_SAMPLE_ROWS,
                 min_line_w=MIN_LINE_WIDTH,
                 max_line_w=MAX_LINE_WIDTH,
                 search_margin=BOUNDARY_SEARCH_MARGIN):
        self.detect_w = detect_w
        self.detect_h = detect_h
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_left = roi_left
        self.roi_right = roi_right
        self.gray_thresh = gray_thresh
        self.num_rows = num_rows
        self.min_line_w = min_line_w
        self.max_line_w = max_line_w
        self.search_margin = search_margin

        self.control_sample_y = self._make_band_rows(
            CONTROL_BAND_TOP, CONTROL_BAND_BOTTOM, CONTROL_SAMPLE_ROWS)
        self.lookahead_sample_y = self._make_band_rows(
            LOOKAHEAD_BAND_TOP, LOOKAHEAD_BAND_BOTTOM, LOOKAHEAD_SAMPLE_ROWS)
        self.sample_y = self.control_sample_y + self.lookahead_sample_y
        self.rgb_path_reported = False

    @staticmethod
    def _make_band_rows(top, bottom, count):
        rows = []
        for i in range(count):
            y = bottom - 1 - i * (bottom - top - 1) // max(1, count - 1)
            rows.append(y)
        return rows

    def extract(self, gray_img) -> BoundaryResult:
        result = BoundaryResult()
        result.sample_y = list(self.sample_y)
        try:
            shape = gray_img.shape
            ndim = len(shape)
            image_h = int(shape[0])
            image_w = int(shape[1])
        except Exception:
            ndim = 0
            image_h = len(gray_img)
            image_w = len(gray_img[0]) if image_h else 0

        for idx, y in enumerate(self.sample_y):
            if y < 0 or y >= self.detect_h or y >= image_h:
                continue

            segments = self._find_black_segments(gray_img, y, image_w, ndim)
            result.all_segments.append(segments)

            mid_x = (self.roi_left + self.roi_right) // 2
            left_x, right_x = self._match_boundary_pair(segments, mid_x)

            if left_x is not None:
                result.left_points.append((left_x, y))
            if right_x is not None:
                result.right_points.append((right_x, y))
            if left_x is not None and right_x is not None:
                cx = (left_x + right_x) // 2
                result.center_points.append((cx, y))
                if y in self.control_sample_y:
                    result.control_center_points.append((cx, y))
                else:
                    result.lookahead_center_points.append((cx, y))
                result.num_valid_rows += 1

        min_valid = max(2, CONTROL_SAMPLE_ROWS // 2)
        control_y = set(self.control_sample_y)
        result.left_valid = sum(1 for _, y in result.left_points if y in control_y) >= min_valid
        result.right_valid = sum(1 for _, y in result.right_points if y in control_y) >= min_valid
        return result

    def _find_black_segments(self, image_array, y, image_w, ndim):
        segments = []
        x_end = min(self.roi_right, image_w)

        # Common fast path: convert only one 280-pixel grayscale row view to a
        # temporary Python list.  This avoids thousands of slow ulab scalar
        # indexing calls while never copying the full frame.
        row_values = None
        if ndim == 2:
            try:
                row_view = image_array[y, self.roi_left:x_end]
                try:
                    row_values = row_view.tolist()
                except Exception:
                    row_values = list(row_view)
            except Exception:
                row_values = None
        elif ndim >= 3:
            try:
                # Critical K230 fast path: three native channel slices per row.
                # Separate flat lists avoid thousands of slow ndarray scalar
                # reads and avoid allocating 280 small RGB sub-lists per row.
                row_values = (
                    image_array[y, self.roi_left:x_end, 0].tolist(),
                    image_array[y, self.roi_left:x_end, 1].tolist(),
                    image_array[y, self.roi_left:x_end, 2].tolist(),
                )
            except Exception:
                try:
                    row_view = image_array[y, self.roi_left:x_end, 0:3]
                    row_values = row_view.tolist()
                except Exception:
                    row_values = None
            if not self.rgb_path_reported:
                if isinstance(row_values, tuple):
                    print("[PERF] RGB row path: channel-slice fast")
                elif row_values is not None:
                    print("[PERF] RGB row path: packed-row fast")
                else:
                    print("[PERF] WARNING: RGB row slicing unavailable; "
                          "slow scalar fallback")
                self.rgb_path_reported = True
        elif ndim == 0:
            try:
                row_values = image_array[y][self.roi_left:x_end]
            except Exception:
                row_values = None

        if row_values is not None:
            i = 0
            is_rgb_planes = (
                isinstance(row_values, tuple) and len(row_values) == 3 and
                isinstance(row_values[0], list)
            )
            row_len = len(row_values[0]) if is_rgb_planes else len(row_values)
            is_rgb_row = (
                not is_rgb_planes and row_len > 0 and
                isinstance(row_values[0], (tuple, list)) and
                len(row_values[0]) >= 3
            )
            is_black_rgb = _is_black_rgb
            while i < row_len:
                if is_rgb_planes:
                    is_dark = is_black_rgb(
                        int(row_values[0][i]) & 0xFF,
                        int(row_values[1][i]) & 0xFF,
                        int(row_values[2][i]) & 0xFF)
                else:
                    value = row_values[i]
                if is_rgb_row:
                    is_dark = _is_black_rgb(
                        int(value[0]) & 0xFF,
                        int(value[1]) & 0xFF,
                        int(value[2]) & 0xFF)
                elif not is_rgb_planes:
                    is_dark = int(value) < self.gray_thresh
                if is_dark:
                    start = i
                    i += 1
                    while i < row_len:
                        if is_rgb_planes:
                            is_dark = is_black_rgb(
                                int(row_values[0][i]) & 0xFF,
                                int(row_values[1][i]) & 0xFF,
                                int(row_values[2][i]) & 0xFF)
                        else:
                            value = row_values[i]
                        if is_rgb_row:
                            is_dark = is_black_rgb(
                                int(value[0]) & 0xFF,
                                int(value[1]) & 0xFF,
                                int(value[2]) & 0xFF)
                        elif not is_rgb_planes:
                            is_dark = int(value) < self.gray_thresh
                        if not is_dark:
                            break
                        i += 1
                    end = i - 1
                    width = end - start + 1
                    if width >= self.min_line_w:
                        segments.append((
                            self.roi_left + start,
                            self.roi_left + end,
                        ))
                else:
                    i += 1
            return segments

        # RGB888 or unusual ndarray fallback.  It still scans only sample rows.
        i = self.roi_left
        while i < x_end:
            if _is_black_pixel(image_array, y, i, ndim):
                start = i
                while i < x_end and \
                      _is_black_pixel(image_array, y, i, ndim):
                    i += 1
                end = i - 1
                w = end - start + 1
                if w >= self.min_line_w:
                    segments.append((start, end))
            else:
                i += 1
        return segments

    def _match_boundary_pair(self, segments, mid_x):
        left_best = None
        right_best = None
        left_dist = float("inf")
        right_dist = float("inf")

        for seg in segments:
            cx = (seg[0] + seg[1]) // 2
            w = seg[1] - seg[0] + 1
            if cx < mid_x:
                d = mid_x - cx
                if d < left_dist and w <= self.max_line_w:
                    left_dist = d
                    left_best = seg
            else:
                d = cx - mid_x
                if d < right_dist and w <= self.max_line_w:
                    right_dist = d
                    right_best = seg

        lx = (left_best[0] + left_best[1]) // 2 if left_best else None
        rx = (right_best[0] + right_best[1]) // 2 if right_best else None

        if lx is not None and rx is not None:
            road_w = rx - lx
            if road_w < EXPECTED_ROAD_WIDTH_MIN or road_w > EXPECTED_ROAD_WIDTH_MAX:
                return lx, None

        return lx, rx


class RoadGeometry:
    def __init__(self, detect_w=320, detect_h=240,
                 scale_x=640.0 / 320.0, scale_y=480.0 / 240.0):
        self.detect_w = detect_w
        self.detect_h = detect_h
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.cx_det = detect_w / 2.0
        self.cy_det = detect_h / 2.0

        self.hist_road_width = 0.0
        self.hist_width_ready = False
        self.single_boundary_frames = 0
        self.single_boundary_start_ms = 0
        self.last_geometry = GeometryResult()

    def compute(self, boundary: BoundaryResult, now_ms=0) -> GeometryResult:
        g = GeometryResult()
        g.left_valid = boundary.left_valid
        g.right_valid = boundary.right_valid
        g.num_valid_rows = boundary.num_valid_rows

        if boundary.left_valid and boundary.right_valid:
            lp = boundary.left_points
            rp = boundary.right_points

            centers = []
            control_centers = []
            control_widths = []
            for (lx, ly) in lp:
                rx = self._find_right_at_y(rp, ly, 10)
                if rx is not None:
                    center = ((lx + rx) / 2.0, ly)
                    width = abs(rx - lx)
                    centers.append(center)
                    if CONTROL_BAND_TOP <= ly < CONTROL_BAND_BOTTOM:
                        control_centers.append(center)
                        control_widths.append(width)

            if len(control_centers) >= 3:
                # Steering position and physical road width come only from the
                # lower control band. Upper samples affect heading/pre-judgment.
                g.road_width = float(sum(control_widths)) / len(control_widths)
                self.hist_road_width = g.road_width
                self.hist_width_ready = True

                near_pts = sorted(control_centers, key=lambda p: p[1], reverse=True)[:4]
                avg_near_x = sum(p[0] for p in near_pts) / max(len(near_pts), 1)
                g.lateral_error = avg_near_x - self.cx_det

                g.heading_error = self._compute_heading(centers)
                g.fit_residual = self._compute_residual(centers)

                g.vision_state = VisionState.NORMAL
                g.degraded = False
                self.single_boundary_frames = 0
                self.single_boundary_start_ms = 0
            else:
                g = self._degrade_result(g, "insufficient_center_points")

        elif boundary.left_valid and not boundary.right_valid:
            g = self._single_boundary_estimate(boundary, side="left", now_ms=now_ms)
            self.single_boundary_frames += 1
        elif boundary.right_valid and not boundary.left_valid:
            g = self._single_boundary_estimate(boundary, side="right", now_ms=now_ms)
            self.single_boundary_frames += 1
        else:
            g.vision_state = VisionState.INVALID
            g.degraded = False
            g.confidence = 0
            self.single_boundary_frames += 1

        g.confidence = self._compute_confidence(boundary, g)
        if g.confidence < CONF_LOW_THRESH:
            g.vision_state = VisionState.INVALID

        self.last_geometry = g
        return g

    def _single_boundary_estimate(self, boundary, side, now_ms):
        g = GeometryResult()
        g.left_valid = boundary.left_valid
        g.right_valid = boundary.right_valid
        g.degraded = True

        if not self.hist_width_ready:
            g.vision_state = VisionState.INVALID
            g.confidence = 0
            return g

        if now_ms and self.single_boundary_start_ms == 0:
            self.single_boundary_start_ms = now_ms
        if (self.single_boundary_frames > SINGLE_BOUNDARY_MAX_FRAMES or
            (now_ms and self.single_boundary_start_ms and
             now_ms - self.single_boundary_start_ms > SINGLE_BOUNDARY_MAX_MS)):
            g.vision_state = VisionState.INVALID
            g.confidence = 0
            return g

        pts = boundary.left_points if side == "left" else boundary.right_points
        if not pts:
            g.vision_state = VisionState.INVALID
            g.confidence = 0
            return g

        near_pts = sorted(pts, key=lambda p: p[1], reverse=True)[:4]
        avg_x = sum(p[0] for p in near_pts) / max(len(near_pts), 1)
        if side == "left":
            est_cx = avg_x + self.hist_road_width / 2.0
        else:
            est_cx = avg_x - self.hist_road_width / 2.0

        # FIXED: lateral_error = est_cx - cx_det (was cx_det - est_cx)
        g.lateral_error = est_cx - self.cx_det
        g.road_width = self.hist_road_width
        g.heading_error = 0.0
        g.vision_state = VisionState.DEGRADED
        g.confidence = int(CONF_LOW_THRESH + 10)
        g.num_valid_rows = boundary.num_valid_rows
        return g

    def _find_right_at_y(self, right_points, y, tolerance):
        best = None
        best_d = tolerance + 1
        for (rx, ry) in right_points:
            d = abs(ry - y)
            if d < best_d:
                best_d = d
                best = rx
        return best

    def _compute_heading(self, centers):
        if len(centers) < 4:
            return 0.0
        sorted_pts = sorted(centers, key=lambda p: p[1])

        n = len(sorted_pts)
        near_pts = sorted_pts[n - n // 3:]
        far_pts = sorted_pts[:n // 3]

        if len(near_pts) < 2 or len(far_pts) < 2:
            return 0.0

        near_avg_x = sum(p[0] for p in near_pts) / len(near_pts)
        near_avg_y = sum(p[1] for p in near_pts) / len(near_pts)
        far_avg_x = sum(p[0] for p in far_pts) / len(far_pts)
        far_avg_y = sum(p[1] for p in far_pts) / len(far_pts)

        dy = near_avg_y - far_avg_y
        if abs(dy) < 1.0:
            return 0.0
        dx = near_avg_x - far_avg_x
        return math.atan2(dx, dy)

    def _compute_residual(self, centers):
        if len(centers) < 4:
            return 0.0
        xs = [p[0] for p in centers]
        ys = [p[1] for p in centers]
        n = len(xs)
        if n < 2:
            return 0.0
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((ys[i] - mean_y) ** 2 for i in range(n))
        if abs(den) < 1e-6:
            return 0.0
        slope = num / den
        intercept = mean_x - slope * mean_y
        residual_sum = sum((xs[i] - (slope * ys[i] + intercept)) ** 2
                           for i in range(n))
        return math.sqrt(residual_sum / n)

    def _compute_confidence(self, boundary, geom):
        score = 0.0
        total_w = 0.0

        if boundary.num_valid_rows > 0:
            point_ratio = boundary.num_valid_rows / float(NUM_SAMPLE_ROWS)
            score += CONF_WEIGHT_VALID_POINTS * point_ratio * 100
        total_w += CONF_WEIGHT_VALID_POINTS

        if geom.fit_residual >= 0:
            residual_score = max(0, 100 - geom.fit_residual * 20)
            score += CONF_WEIGHT_FIT_RESIDUAL * residual_score
        total_w += CONF_WEIGHT_FIT_RESIDUAL

        if geom.road_width > 0:
            if EXPECTED_ROAD_WIDTH_MIN <= geom.road_width <= EXPECTED_ROAD_WIDTH_MAX:
                width_score = 100
            else:
                dist = min(abs(geom.road_width - EXPECTED_ROAD_WIDTH_MIN),
                          abs(geom.road_width - EXPECTED_ROAD_WIDTH_MAX))
                width_score = max(0, 100 - dist * 2)
            score += CONF_WEIGHT_WIDTH * width_score
        total_w += CONF_WEIGHT_WIDTH

        cont_score = 50
        if boundary.left_valid and boundary.right_valid:
            cont_score = 100
        elif boundary.left_valid or boundary.right_valid:
            cont_score = 50
        score += CONF_WEIGHT_CONTINUITY * cont_score
        total_w += CONF_WEIGHT_CONTINUITY

        if self.last_geometry and self.last_geometry.road_width > 0:
            if geom.road_width > 0:
                change = abs(geom.road_width - self.last_geometry.road_width)
                stability_score = max(0, 100 - change * 3)
            else:
                stability_score = 0
        else:
            stability_score = 80
        score += CONF_WEIGHT_STABILITY * stability_score
        total_w += CONF_WEIGHT_STABILITY

        if total_w <= 0:
            return 0
        return int(min(100, max(0, score / total_w)))

    def _degrade_result(self, g, reason):
        g.vision_state = VisionState.DEGRADED
        g.degraded = True
        g.confidence = CONF_LOW_THRESH + 5
        return g

    def reset_history(self):
        self.hist_width_ready = False
        self.hist_road_width = 0.0
        self.single_boundary_frames = 0
        self.single_boundary_start_ms = 0

    def to_logical(self, det_val, axis="x"):
        return det_val * (self.scale_x if axis == "x" else self.scale_y)


# ============================================================================
# road_structure.py — 道路结构识别层
# ============================================================================

class JunctionStage:
    NONE = 0
    APPROACHING = 1
    NEAR = 2
    AT = 3
    PASSED = 4

    @staticmethod
    def name(stage):
        return {0: "NONE", 1: "APPROACHING", 2: "NEAR",
                3: "AT", 4: "PASSED"}.get(stage, "???")


class StructureResult:
    def __init__(self):
        self.left_branch = False
        self.right_branch = False
        self.intersection_candidate = False
        self.junction_stage = JunctionStage.NONE
        self.junction_distance = 0
        self.junction_distance_px = 0
        self.structure_confirmed = False


class RoadStructureDetector:
    def __init__(self):
        self.left_branch_hits = 0
        self.right_branch_hits = 0
        self.intersection_hits = 0

        self.prev_left_branch = False
        self.prev_right_branch = False
        self.prev_intersection = False
        self.cooldown_counter = 0

        self.width_history = []
        self.max_width_history = 10

        self._confirmed_junction_active = False
        self.junction_y_px = -1
        self._last_junction_stage = JunctionStage.NONE
        self._last_junction_distance = 0
        self._last_junction_distance_px = 0

    def detect(self, boundary, geom) -> StructureResult:
        result = StructureResult()
        self.cooldown_counter = max(0, self.cooldown_counter - 1)

        if geom.vision_state == 2:
            self._decay_counts()
            if self._confirmed_junction_active:
                self._transition_to_passed()
            return result

        if geom.road_width > 0:
            self.width_history.append(geom.road_width)
            if len(self.width_history) > self.max_width_history:
                self.width_history.pop(0)

        width_increase = False
        if len(self.width_history) >= 3 and geom.road_width > 0:
            avg_old = sum(self.width_history[:-1]) / max(len(self.width_history) - 1, 1)
            if avg_old > 0:
                ratio = geom.road_width / avg_old
                if ratio > JUNCTION_WIDTH_INCREASE_RATIO:
                    width_increase = True

        left_expanded = self._check_boundary_expansion(boundary.left_points, side="left")
        right_expanded = self._check_boundary_expansion(boundary.right_points, side="right")

        extra_left, extra_right, extra_segments_y = self._check_extra_segments(boundary)

        has_left_feature = extra_left or left_expanded
        has_right_feature = extra_right or right_expanded

        if has_left_feature:
            self.left_branch_hits += 1
        else:
            self.left_branch_hits = max(0, self.left_branch_hits - 1)

        if has_right_feature:
            self.right_branch_hits += 1
        else:
            self.right_branch_hits = max(0, self.right_branch_hits - 1)

        wide_cross = self._has_wide_crossbar(boundary)
        if wide_cross or (width_increase and has_left_feature and has_right_feature):
            self.intersection_hits += 1
        else:
            self.intersection_hits = max(0, self.intersection_hits - 1)

        any_hit = (self.left_branch_hits > 0 or self.right_branch_hits > 0 or
                   self.intersection_hits > 0)

        if not self._confirmed_junction_active and self.cooldown_counter <= 0:
            if (self.left_branch_hits >= JUNCTION_CONFIRM_FRAMES or
                    self.right_branch_hits >= JUNCTION_CONFIRM_FRAMES or
                    self.intersection_hits >= JUNCTION_CONFIRM_FRAMES):
                self._confirmed_junction_active = True
                self.junction_y_px = self._compute_junction_y(
                    extra_segments_y, boundary, width_increase)

        if self._confirmed_junction_active:
            if any_hit:
                current_y = self._compute_junction_y(
                    extra_segments_y, boundary, width_increase)
                if current_y > 0 and current_y > self.junction_y_px:
                    self.junction_y_px = current_y

            result.left_branch = True
            result.right_branch = True
            result.intersection_candidate = True
            result.structure_confirmed = True

            if not any_hit:
                self._transition_to_passed()

            if self.junction_y_px > 0:
                stage, distance, dist_px = self._estimate_junction_stage(self.junction_y_px)
            else:
                stage, distance, dist_px = JunctionStage.APPROACHING, 1, JUNCTION_DISTANCE_MID_PX

            if self._last_junction_stage == JunctionStage.PASSED:
                stage = JunctionStage.PASSED
                distance = self._last_junction_distance
                dist_px = self._last_junction_distance_px

            result.junction_stage = stage
            result.junction_distance = distance
            result.junction_distance_px = dist_px

            self._last_junction_stage = stage
            self._last_junction_distance = distance
            self._last_junction_distance_px = dist_px

            return result

        result.left_branch = (self.left_branch_hits >= JUNCTION_CONFIRM_FRAMES and
                              self.cooldown_counter <= 0)
        result.right_branch = (self.right_branch_hits >= JUNCTION_CONFIRM_FRAMES and
                               self.cooldown_counter <= 0)
        result.intersection_candidate = (self.intersection_hits >= JUNCTION_CONFIRM_FRAMES and
                                         self.cooldown_counter <= 0)

        result.junction_stage, result.junction_distance, result.junction_distance_px = \
            self._estimate_junction_stage(self.junction_y_px)

        result.structure_confirmed = (result.left_branch or result.right_branch or
                                      result.intersection_candidate)

        self.prev_left_branch = result.left_branch
        self.prev_right_branch = result.right_branch
        self.prev_intersection = result.intersection_candidate

        return result

    def _check_boundary_expansion(self, points, side):
        if not points or len(points) < 3:
            return False
        sorted_pts = sorted(points, key=lambda p: p[1], reverse=True)
        near = sorted_pts[:3]
        far = sorted_pts[-3:]
        if len(near) < 2 or len(far) < 2:
            return False
        near_avg_x = sum(p[0] for p in near) / len(near)
        far_avg_x = sum(p[0] for p in far) / len(far)
        diff = far_avg_x - near_avg_x
        if side == "left":
            return diff < -8
        else:
            return diff > 8

    def _check_extra_segments(self, boundary):
        if not hasattr(boundary, 'all_segments') or not boundary.all_segments:
            return False, False, -1

        extra_left = False
        extra_right = False
        extra_y = -1

        left_by_y = {}
        right_by_y = {}
        for x, y in boundary.left_points:
            left_by_y[int(y)] = x
        for x, y in boundary.right_points:
            right_by_y[int(y)] = x

        num_rows = len(boundary.all_segments)
        if num_rows == 0:
            return False, False, -1

        for i, segments in enumerate(boundary.all_segments):
            if not segments:
                continue
            y = boundary.sample_y[i] if i < len(boundary.sample_y) else -1

            lx = left_by_y.get(y)
            rx = right_by_y.get(y)
            if lx is None or rx is None:
                continue

            for seg_start, seg_end in segments:
                seg_cx = (seg_start + seg_end) // 2
                seg_w = seg_end - seg_start + 1

                if seg_w < 3:
                    continue

                if lx + 5 < seg_cx < rx - 5:
                    extra_left = True
                    extra_right = True
                    if y > extra_y:
                        extra_y = y
                elif seg_cx < lx - 12:
                    extra_left = True
                    if y > extra_y:
                        extra_y = y
                elif seg_cx > rx + 12:
                    extra_right = True
                    if y > extra_y:
                        extra_y = y

        return extra_left, extra_right, extra_y

    def _has_wide_crossbar(self, boundary):
        """Detect a horizontal crossbar on at least two look-ahead rows."""
        hits = 0
        for i, segments in enumerate(boundary.all_segments):
            y = boundary.sample_y[i] if i < len(boundary.sample_y) else -1
            if not (LOOKAHEAD_BAND_TOP <= y < LOOKAHEAD_BAND_BOTTOM):
                continue
            if any((end - start + 1) >= WIDE_JUNCTION_MIN_WIDTH
                   for start, end in segments):
                hits += 1
        return hits >= 2

    def _estimate_junction_stage(self, junction_y_px):
        if junction_y_px <= 0:
            return JunctionStage.NONE, 0, 0

        dist_from_bottom = DETECT_HEIGHT - junction_y_px

        if dist_from_bottom <= JUNCTION_DISTANCE_NEAR_PX:
            stage = JunctionStage.AT
            distance = 3
        elif dist_from_bottom <= JUNCTION_DISTANCE_MID_PX:
            stage = JunctionStage.NEAR
            distance = 2
        else:
            stage = JunctionStage.APPROACHING
            distance = 1

        return stage, distance, dist_from_bottom

    def _compute_junction_y(self, extra_segments_y, boundary, width_increase):
        y = extra_segments_y
        if y <= 0:
            all_pts = boundary.left_points + boundary.right_points
            if all_pts:
                y = max(p[1] for p in all_pts)
        return y

    def _transition_to_passed(self):
        self._confirmed_junction_active = False
        self.cooldown_counter = JUNCTION_COOLDOWN_FRAMES
        self._last_junction_stage = JunctionStage.PASSED
        # Distance protocol only defines 0..3. PASSED has no active distance.
        self._last_junction_distance = 0
        self._last_junction_distance_px = 0
        self.junction_y_px = -1

    def _decay_counts(self):
        self.left_branch_hits = max(0, self.left_branch_hits - 1)
        self.right_branch_hits = max(0, self.right_branch_hits - 1)
        self.intersection_hits = max(0, self.intersection_hits - 1)

    def _reset_counts(self):
        self._decay_counts()
        self.left_branch_hits = 0
        self.right_branch_hits = 0
        self.intersection_hits = 0
        self.prev_left_branch = False
        self.prev_right_branch = False
        self.prev_intersection = False
        self.cooldown_counter = 0
        self._confirmed_junction_active = False
        self.junction_y_px = -1
        self._last_junction_stage = JunctionStage.NONE
        self._last_junction_distance = 0
        self._last_junction_distance_px = 0
        self.width_history.clear()

    def on_turning_complete(self):
        self._reset_counts()


# ============================================================================
# main_k230_road_vision.py — 板端主程序
# ============================================================================

class ImagePreprocessor:
    """
    Board preprocessing: consume the RGB888 ndarray reference directly.
    It scans only sparse points on the 12 configured ROI rows, with no full
    frame copy and no full-frame Python list.
    """
    _shape_printed = False

    def __init__(self, sample_y_list):
        self.sample_y = sample_y_list
        self.roi_left = ROI_LEFT
        self.roi_right = ROI_RIGHT
        self.detect_w = DETECT_WIDTH
        self.frame_index = 0
        self.last_anomaly_flags = 0

    def extract_rows(self, img):
        try:
            # Keep RGB so the extractor can reject the red centre tape.
            image_np = img.to_numpy_ref()
        except Exception as e:
            try:
                gray_img = img.to_grayscale()
                if gray_img is None:
                    gray_img = img
                image_np = gray_img.to_numpy_ref()
                if not ImagePreprocessor._shape_printed:
                    print("[PREPROC] RGB888 unavailable; grayscale fallback "
                          "cannot reject red tape:", repr(e))
            except Exception as fallback_error:
                print("[PREPROC] image conversion failed:", repr(fallback_error))
                return None, 0

        if not ImagePreprocessor._shape_printed:
            try:
                print("[PREPROC] working ndarray shape:", image_np.shape)
            except Exception as e:
                print("[PREPROC] ndarray shape unavailable:", repr(e))
            ImagePreprocessor._shape_printed = True

        try:
            ndim = len(image_np.shape)
            h = int(image_np.shape[0])
            w = int(image_np.shape[1])
        except Exception as e:
            print("[PREPROC] invalid ndarray:", repr(e))
            return None, 0

        # Exposure is slow-changing telemetry, not steering input. Sampling it
        # every tenth frame removes another 720 RGB ndarray scalar reads from
        # nine out of ten frames.
        self.frame_index += 1
        if self.frame_index % 10 == 1:
            total = 0
            count = 0
            step = max(1, (self.roi_right - self.roi_left) // 20)
            x_end = min(self.roi_right, w)
            for y in self.sample_y:
                if y < 0 or y >= h:
                    continue
                for x in range(self.roi_left, x_end, step):
                    total += _gray_pixel(image_np, y, x, ndim)
                    count += 1

            flags = 0
            if count:
                avg = total / count
                if avg < 25:
                    flags |= ANOMALY_UNDEREXPOSED
                elif avg > 230:
                    flags |= ANOMALY_OVEREXPOSED
            self.last_anomaly_flags = flags

        return image_np, self.last_anomaly_flags


class RoadUART:
    def __init__(self, tx_pin, rx_pin, uart_id, baud):
        self.uart = None
        self.rx_buf = ""
        try:
            fpioa = FPIOA()
            fpioa.set_function(int(tx_pin), FPIOA.UART3_TXD, ie=0, oe=1)
            fpioa.set_function(int(rx_pin), FPIOA.UART3_RXD, ie=1, oe=0)
            uid = getattr(UART, "UART3", int(uart_id))
            try:
                self.uart = UART(uid, baudrate=int(baud), bits=8,
                                 parity=None, stop=1, timeout=0)
            except TypeError:
                self.uart = UART(uid, baudrate=int(baud))
            print("[UART] ready GPIO%d(TX) GPIO%d(RX) @ %d" % (tx_pin, rx_pin, baud))
        except Exception as e:
            print("[UART] init failed:", e)

    def send_raw(self, raw_bytes):
        try:
            if self.uart and raw_bytes:
                self.uart.write(raw_bytes)
        except Exception:
            pass

    def recv(self, max_pkts=4):
        pkts = []
        if not self.uart:
            return pkts
        try:
            if hasattr(self.uart, "any") and self.uart.any() <= 0:
                return pkts
            data = self.uart.read()
        except Exception:
            return pkts
        if not data:
            return pkts
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except Exception:
                data = ""
        self.rx_buf += str(data)
        if len(self.rx_buf) > 1024:
            pos = self.rx_buf.rfind("[")
            self.rx_buf = self.rx_buf[pos:] if pos >= 0 else ""

        while len(pkts) < max_pkts and "[" in self.rx_buf and "]" in self.rx_buf:
            s = self.rx_buf.find("[")
            e = self.rx_buf.find("]", s)
            if e <= s:
                break
            parts = [p.strip() for p in self.rx_buf[s + 1:e].split(",")]
            self.rx_buf = self.rx_buf[e + 1:]
            if parts and parts[0]:
                pkts.append(parts)
        return pkts

    def deinit(self):
        if self.uart:
            try:
                self.uart.deinit()
            except Exception:
                pass


class ModeStateMachine:
    def __init__(self):
        self.mode = MODE_IDLE
        self.prev_mode = MODE_IDLE
        self.mode_changed = False
        self.reacquire_count = 0
        self._on_turning_to_reacquire_cb = None

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            prev = self.mode
            self.prev_mode = prev
            self.mode = new_mode
            self.mode_changed = True
            self.reacquire_count = 0
            if prev == MODE_TURNING and new_mode == MODE_REACQUIRE:
                if self._on_turning_to_reacquire_cb is not None:
                    try:
                        self._on_turning_to_reacquire_cb()
                    except Exception:
                        pass

    def update(self, road_geom, struct_result):
        if self.mode == MODE_REACQUIRE:
            if road_geom.vision_state == VisionState.NORMAL and \
               road_geom.confidence >= CONF_HIGH_THRESH:
                self.reacquire_count += 1
            else:
                self.reacquire_count = max(0, self.reacquire_count - 1)
            if self.reacquire_count >= REACQUIRE_CONFIRM_FRAMES:
                self.set_mode(MODE_TRACK)


class K230RoadVision:
    def __init__(self):
        self.mode_fsm = ModeStateMachine()

        self.boundary_extractor = RoadBoundaryExtractor(
            detect_w=DETECT_WIDTH, detect_h=DETECT_HEIGHT,
            roi_top=ROI_TOP, roi_bottom=ROI_BOTTOM,
            roi_left=ROI_LEFT, roi_right=ROI_RIGHT,
            gray_thresh=LINE_GRAY_THRESH,
            min_line_w=MIN_LINE_WIDTH, max_line_w=MAX_LINE_WIDTH,
            search_margin=BOUNDARY_SEARCH_MARGIN,
        )

        self.road_geom = RoadGeometry(
            detect_w=DETECT_WIDTH, detect_h=DETECT_HEIGHT,
            scale_x=2.0, scale_y=2.0,
        )
        self.structure_detector = RoadStructureDetector()
        self.frame_builder = RoadFrameBuilder()
        self.preprocessor = ImagePreprocessor(self.boundary_extractor.sample_y)

        self.mode_fsm._on_turning_to_reacquire_cb = self._on_turning_to_reacquire

        self.uart = None
        self.frame_count = 0
        self.fps = 0.0
        self.last_send_ms = 0
        self.last_status_ms = 0
        self.last_fps_calc_ms = 0
        self.last_fps_count = 0

        self.display_ok = False
        self.display_x = (640 - DETECT_WIDTH) // 2
        self.display_y = (480 - DETECT_HEIGHT) // 2

    def _on_turning_to_reacquire(self):
        self.road_geom.reset_history()
        self.structure_detector.on_turning_complete()

    def _should_output_valid(self):
        m = self.mode_fsm.mode
        if m in (MODE_IDLE, MODE_FAULT, MODE_TURNING, MODE_NUMBER):
            return False
        if m == MODE_REACQUIRE:
            return self.mode_fsm.reacquire_count >= REACQUIRE_CONFIRM_FRAMES
        return True

    def run(self):
        sensor = None

        try:
            print("BUILD:", BUILD_ID)
            print("[CAM] request sensor id=%d input=%dx%d@%d" % (
                SENSOR_ID, CAM_INPUT_WIDTH, CAM_INPUT_HEIGHT, CAM_FPS))
            self.uart = RoadUART(UART_TX_PIN, UART_RX_PIN, UART_ID, UART_BAUD)

            print("=" * 55)
            print("K230 Road Vision v1.0")
            print("  DETECT: %dx%d  ROI: [%d:%d, %d:%d]" % (
                DETECT_WIDTH, DETECT_HEIGHT, ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT))
            print("  UART3: GPIO%d/GPIO%d @ %d baud" % (
                UART_TX_PIN, UART_RX_PIN, UART_BAUD))
            print("  MSPM0 receiver: UNCONFIRMED (protocol v0x%02X)" % PROTOCOL_VERSION)
            print("=" * 55)

            try:
                print("[CAM] creating Sensor(id=%d)" % SENSOR_ID)
                sensor = Sensor(
                    id=SENSOR_ID,
                    width=CAM_INPUT_WIDTH,
                    height=CAM_INPUT_HEIGHT,
                    fps=CAM_FPS,
                )
            except Exception as e:
                print("[FATAL] Sensor init failed:", repr(e))
                raise
            print("[CAM] sensor created")
            sensor.reset()
            print("[CAM] sensor reset completed")

            sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT,
                                 chn=CAM_CHN_ID_0)
            print("[CAM] CH0 framesize=%dx%d" % (DETECT_WIDTH, DETECT_HEIGHT))
            sensor.set_pixformat(Sensor.RGB888, chn=CAM_CHN_ID_0)
            print("[CAM] CH0 pixformat=RGB888")

            if ENABLE_DISPLAY:
                try:
                    Display.init(Display.ST7701, width=640, height=480,
                                 to_ide=DISPLAY_TO_IDE, quality=DISPLAY_QUALITY)
                    self.display_ok = True
                    print("[DISPLAY] ST7701 ready, to_ide=%d" %
                          (1 if DISPLAY_TO_IDE else 0))
                except Exception as e:
                    self.display_ok = False
                    print("[DISPLAY] init failed, continue headless:", repr(e))

            print("[CAM] MediaManager.init")
            MediaManager.init()
            print("[CAM] sensor.run")
            sensor.run()
            print("[CAM] sensor.run completed")

            print("warming up camera...")
            time.sleep_ms(300)
            for i in range(15):
                os.exitpoint()
                try:
                    img = sensor.snapshot(chn=CAM_CHN_ID_0)
                except Exception as e:
                    print("[FATAL] warmup snapshot %d failed:" % i, repr(e))
                    raise
                time.sleep_ms(10)
            print("[CAM] warmup completed")

            print("[CAM] SENSOR_ID=%d  input=%dx%d@%dfps" % (
                SENSOR_ID, CAM_INPUT_WIDTH, CAM_INPUT_HEIGHT, CAM_FPS))
            print("[CAM] CAM_CHN_ID_0  detect=%dx%d  pixformat=RGB888" % (
                DETECT_WIDTH, DETECT_HEIGHT))
            try:
                print("[CAM] first image: width=%d height=%d" % (img.width(), img.height()))
            except Exception:
                pass

            self.mode_fsm.set_mode(MODE_TRACK)
            self.uart.send_raw(self._build_status_frame())

            now_ms = time.ticks_ms()
            self.last_send_ms = now_ms
            self.last_status_ms = now_ms
            self.last_fps_calc_ms = now_ms

            print("entering main loop...")
            print("commands: [mode,idle] [mode,track] [mode,turning] [mode,fault]")
            print("-" * 55)

            while True:
                os.exitpoint()
                loop_start = time.ticks_ms()

                img = sensor.snapshot(chn=CAM_CHN_ID_0)

                self._handle_commands()

                if self.mode_fsm.mode == MODE_IDLE:
                    time.sleep_ms(IDLE_SLEEP_MS)
                    if self.uart:
                        self.uart.send_raw(self._build_status_frame())
                    self._update_fps(loop_start)
                    continue

                elif self.mode_fsm.mode == MODE_FAULT:
                    self.uart.send_raw(self._build_invalid_frame())
                    time.sleep_ms(50)
                    self._update_fps(loop_start)
                    continue

                gray_np, anomaly = self.preprocessor.extract_rows(img)
                if gray_np is None:
                    if self.uart:
                        self.uart.send_raw(self._build_invalid_frame())
                    self._update_fps(loop_start)
                    continue

                boundary = self.boundary_extractor.extract(gray_np)

                if self.mode_fsm.mode in (MODE_TURNING, MODE_NUMBER):
                    struct = self.structure_detector.detect(boundary, self.road_geom.last_geometry)
                    self.mode_fsm.update(self.road_geom.last_geometry, struct)

                    elapsed = ticks_ms_diff(loop_start, now_ms)
                    if ticks_ms_diff(loop_start, self.last_send_ms) >= FRAME_SEND_MIN_MS:
                        self.last_send_ms = loop_start
                        frame = self._build_road_frame(
                            self.road_geom.last_geometry, struct, anomaly, loop_start)
                        if self.uart:
                            self.uart.send_raw(frame)

                    self._update_fps(loop_start)
                    elapsed = ticks_ms_diff(time.ticks_ms(), loop_start)
                    if elapsed < 8:
                        time.sleep_ms(8 - elapsed)
                    continue

                geom = self.road_geom.compute(boundary, now_ms=loop_start)

                struct = self.structure_detector.detect(boundary, geom)

                self.mode_fsm.update(geom, struct)

                elapsed = ticks_ms_diff(loop_start, now_ms)
                if ticks_ms_diff(loop_start, self.last_send_ms) >= FRAME_SEND_MIN_MS:
                    self.last_send_ms = loop_start
                    frame = self._build_road_frame(geom, struct, anomaly, loop_start)
                    if self.uart:
                        self.uart.send_raw(frame)

                if ticks_ms_diff(loop_start, self.last_status_ms) >= 1000:
                    self.last_status_ms = loop_start
                    state_s = VisionState.name(geom.vision_state)
                    print("fps=%.1f state=%s conf=%d lateral=%.1f head=%.3f w=%.1f L=%d R=%d br=(%d,%d)" % (
                        self.fps, state_s, geom.confidence,
                        geom.lateral_error, geom.heading_error,
                        geom.road_width,
                        1 if geom.left_valid else 0,
                        1 if geom.right_valid else 0,
                        1 if struct.left_branch else 0,
                        1 if struct.right_branch else 0,
                    ))

                if self.display_ok and self.frame_count % DISPLAY_EVERY_N == 0:
                    self._draw_overlay(img, boundary, geom, struct)
                    try:
                        Display.show_image(img, x=self.display_x, y=self.display_y)
                    except Exception:
                        pass

                self._update_fps(loop_start)

                if self.frame_count % GC_INTERVAL_FRAMES == 0:
                    try:
                        if gc.mem_free() < GC_FREE_THRESH:
                            gc.collect()
                    except Exception:
                        pass

                elapsed = ticks_ms_diff(time.ticks_ms(), loop_start)
                if elapsed < 8:
                    time.sleep_ms(8 - elapsed)

        except KeyboardInterrupt:
            print("\nuser stop")
        except BaseException as e:
            print("[FATAL] main exception:", repr(e))
            try:
                sys.print_exception(e)
            except Exception:
                pass
        finally:
            print("cleanup...")
            if self.uart:
                try:
                    self.uart.send_raw(self._build_invalid_frame())
                except Exception:
                    pass

            if isinstance(sensor, Sensor):
                try:
                    sensor.stop()
                except Exception:
                    pass
            if self.display_ok:
                try:
                    Display.deinit()
                except Exception:
                    pass
            try:
                os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            except Exception:
                pass
            time.sleep_ms(100)

            try:
                MediaManager.deinit()
            except Exception:
                pass

            if self.uart:
                self.uart.deinit()
            print("program exited safely.")

    def _build_road_frame(self, geom, struct, anomaly, timestamp_ms):
        valid = (
            self._should_output_valid() and
            geom.vision_state != VisionState.INVALID and
            geom.road_width > 0
        )

        if valid:
            flags = FLAG_VISION_VALID
            if geom.degraded:
                flags |= FLAG_DEGRADED
            if geom.left_valid:
                flags |= FLAG_LEFT_VALID
            if geom.right_valid:
                flags |= FLAG_RIGHT_VALID
            if ENABLE_JUNCTION_OUTPUT:
                if struct.left_branch:
                    flags |= FLAG_LEFT_BRANCH
                if struct.right_branch:
                    flags |= FLAG_RIGHT_BRANCH
                if struct.intersection_candidate:
                    flags |= FLAG_INTERSECTION

            px_to_mm = 1.5
            lateral_raw = int(round(geom.lateral_error * px_to_mm * 10))
            heading_raw = int(round(geom.heading_error * 180.0 / 3.14159 * 100))
            width_raw = int(round(geom.road_width * px_to_mm * 10))

            lateral_raw = max(-32767, min(32767, lateral_raw))
            heading_raw = max(-32767, min(32767, heading_raw))
            width_raw = max(0, min(65534, width_raw))
        else:
            flags = 0
            lateral_raw = INVALID_S16
            heading_raw = INVALID_S16
            width_raw = INVALID_U16

        confidence = geom.confidence if valid else 0
        junction_stage = struct.junction_stage if ENABLE_JUNCTION_OUTPUT else 0
        junction_distance = struct.junction_distance if ENABLE_JUNCTION_OUTPUT else 0

        return self.frame_builder.build(
            timestamp_ms=timestamp_ms,
            mode=self.mode_fsm.mode,
            flags=flags,
            lateral_error_raw=lateral_raw,
            heading_error_raw=heading_raw,
            road_width_raw=width_raw,
            junction_stage=junction_stage,
            junction_distance=junction_distance,
            confidence=confidence,
            anomaly_flags=anomaly,
        )

    def _build_status_frame(self):
        return self.frame_builder.build(
            timestamp_ms=time.ticks_ms(),
            mode=self.mode_fsm.mode,
            flags=0,
            lateral_error_raw=INVALID_S16,
            heading_error_raw=INVALID_S16,
            road_width_raw=INVALID_U16,
            junction_stage=0,
            junction_distance=0,
            confidence=0,
            anomaly_flags=0,
        )

    def _build_invalid_frame(self):
        return self.frame_builder.build(
            timestamp_ms=time.ticks_ms(),
            mode=self.mode_fsm.mode,
            flags=0,
            lateral_error_raw=INVALID_S16,
            heading_error_raw=INVALID_S16,
            road_width_raw=INVALID_U16,
            junction_stage=0,
            junction_distance=0,
            confidence=0,
            anomaly_flags=ANOMALY_BLUR,
        )

    def _handle_commands(self):
        if not self.uart:
            return
        for p in self.uart.recv():
            if not p:
                continue
            cmd = p[0].lower()
            try:
                if cmd == "mode" and len(p) >= 2:
                    req = p[1].lower().strip()
                    mode_map = {
                        "idle": MODE_IDLE, "track": MODE_TRACK,
                        "turning": MODE_TURNING, "reacquire": MODE_REACQUIRE,
                        "fault": MODE_FAULT, "number": MODE_NUMBER,
                        "0": MODE_IDLE, "1": MODE_TRACK, "2": MODE_TURNING,
                        "3": MODE_REACQUIRE, "4": MODE_FAULT, "5": MODE_NUMBER,
                    }
                    if req in mode_map:
                        self.mode_fsm.set_mode(mode_map[req])
            except Exception:
                pass

    def _update_fps(self, now_ms):
        self.frame_count += 1
        if ticks_ms_diff(now_ms, self.last_fps_calc_ms) >= 1000:
            elapsed_ms = max(1, ticks_ms_diff(now_ms, self.last_fps_calc_ms))
            n = self.frame_count - self.last_fps_count
            self.fps = float(n) * 1000.0 / float(elapsed_ms)
            self.last_fps_calc_ms = now_ms
            self.last_fps_count = self.frame_count

    def _draw_overlay(self, img, boundary, geom, struct):
        try:
            for (lx, ly) in boundary.left_points:
                img.draw_circle(int(lx), int(ly), 2, color=(0, 0, 255), fill=True)
            for (rx, ry) in boundary.right_points:
                img.draw_circle(int(rx), int(ry), 2, color=(255, 0, 0), fill=True)
            for (cx, cy) in boundary.center_points:
                img.draw_circle(int(cx), int(cy), 1, color=(0, 255, 0), fill=True)
            # Camera/vehicle reference center.
            cx_line = int(DETECT_WIDTH / 2)
            img.draw_line(cx_line, LOOKAHEAD_BAND_TOP, cx_line, CONTROL_BAND_BOTTOM,
                          color=(255, 255, 0), thickness=1)

            # Connect detected road-center samples.
            centers = sorted(boundary.center_points, key=lambda p: p[1])
            for i in range(1, len(centers)):
                x0, y0 = centers[i - 1]
                x1, y1 = centers[i]
                img.draw_line(int(x0), int(y0), int(x1), int(y1),
                              color=(0, 255, 0), thickness=2)

            img.draw_rectangle(ROI_LEFT, CONTROL_BAND_TOP,
                               ROI_RIGHT - ROI_LEFT,
                               CONTROL_BAND_BOTTOM - CONTROL_BAND_TOP,
                               color=(0, 255, 0), thickness=1)
            img.draw_rectangle(ROI_LEFT, LOOKAHEAD_BAND_TOP,
                               ROI_RIGHT - ROI_LEFT,
                               LOOKAHEAD_BAND_BOTTOM - LOOKAHEAD_BAND_TOP,
                               color=(255, 128, 0), thickness=1)

            state_name = VisionState.name(geom.vision_state)
            mode_name = {
                MODE_IDLE: "IDLE",
                MODE_TRACK: "TRACK",
                MODE_TURNING: "TURN",
                MODE_REACQUIRE: "REACQ",
                MODE_FAULT: "FAULT",
                MODE_NUMBER: "NUMBER",
            }.get(self.mode_fsm.mode, "?")
            branch_text = "%d%d%d" % (
                1 if struct.left_branch else 0,
                1 if struct.right_branch else 0,
                1 if struct.intersection_candidate else 0,
            )

            img.draw_string(
                2, 2,
                "FPS:%.1f %s %s C:%d" % (
                    self.fps, mode_name, state_name, geom.confidence),
                color=(255, 255, 0), scale=1,
            )
            img.draw_string(
                2, 14,
                "E:%.1fpx H:%.1fdeg W:%.1f" % (
                    geom.lateral_error,
                    geom.heading_error * 57.2958,
                    geom.road_width),
                color=(255, 255, 0), scale=1,
            )
            img.draw_string(
                2, 26,
                "L:%d R:%d J:%s OUT:%d" % (
                    1 if geom.left_valid else 0,
                    1 if geom.right_valid else 0,
                    branch_text,
                    1 if ENABLE_JUNCTION_OUTPUT else 0),
                color=(255, 255, 0), scale=1,
            )
        except Exception:
            pass


def ticks_ms_diff(now, old):
    diff = now - old
    if diff < -0x40000000:
        diff += 0x100000000
    return diff & 0x7FFFFFFF


if __name__ == "__main__":
    if ON_K230:
        K230RoadVision().run()
    else:
        print("This program requires K230/CanMV MicroPython hardware.")
        print("Run the PC tests instead: python tests/test_road_geometry.py")
