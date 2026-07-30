# -*- coding: utf-8 -*-
"""
K230 车载平衡滚球 - 管道矩形 + 局部球跟踪高速版

每帧正常路径：
    snapshot
    -> 上次球心附近约 100x70 ROI 内一次 find_blobs
    -> x 相对管道矩形左右边界换算 0~25 cm
    -> UART

低频路径：
    每 8 帧在限定搜索区内找一次白色长矩形并缓存。

丢球路径：
    先扩大局部 ROI；连续丢失后每 2 帧在缓存的管道矩形内重捕获。

没有 RGB888、to_numpy_ref、Python 逐像素循环、Hough 圆、find_rects。
"""

import gc
import os
import sys
import time

from machine import FPIOA, UART
from media.sensor import *
from media.display import *
from media.media import *


BUILD_ID = "K230_BALL_PIPE_FUSION_GREEN_20260730_19"

# ---------------------------------------------------------------------------
# 摄像头
# ---------------------------------------------------------------------------

SENSOR_ID = 2
CAM_INPUT_WIDTH = 1280
CAM_INPUT_HEIGHT = 960
CAM_FPS = 90
DETECT_WIDTH = 640
DETECT_HEIGHT = 480

# CH0保持灰度检测，完整保留已经验证准确的钢球识别链路。
# CH1仅用于彩色显示，使管道矩形可以用绿色绘制。
DETECT_CHANNEL = CAM_CHN_ID_0
DISPLAY_CHANNEL = CAM_CHN_ID_1

SENSOR_HMIRROR = False
SENSOR_VFLIP = False

# ---------------------------------------------------------------------------
# 管道矩形：动态白色长矩形锁定
# ---------------------------------------------------------------------------

PIPE_LENGTH_CM = 25.0

# 动态管道识别失败时的初始/回退矩形。
PIPE_INITIAL_LEFT = 51
PIPE_INITIAL_RIGHT = 612       # exclusive
PIPE_INITIAL_TOP = 252
PIPE_INITIAL_BOTTOM = 295      # exclusive

# 只在该范围搜索白色管道。
PIPE_SEARCH_X = 15
PIPE_SEARCH_Y = 152
PIPE_SEARCH_W = 610
PIPE_SEARCH_H = 255

PIPE_UPDATE_EVERY_N = 8
PIPE_WHITE_THRESHOLD_DEFAULT = 150
PIPE_WHITE_THRESHOLD_MIN = 140
PIPE_WHITE_THRESHOLD_MAX = 205
PIPE_WHITE_OFFSET_FROM_UQ = 55
PIPE_MIN_WIDTH = 360
PIPE_MAX_WIDTH = 625
PIPE_MIN_HEIGHT = 16
PIPE_MAX_HEIGHT = 175
PIPE_MIN_PIXELS = 1200
PIPE_SMOOTH_ALPHA = 0.60

# 管道物理长度不会在相邻两次更新间突然缩短。
# 底板遮挡或粘连造成某一端向内跳变时，保留该端上一次可信坐标。
PIPE_ENDPOINT_MIN_TRUSTED_SPAN = 500
PIPE_ENDPOINT_MAX_INWARD_ERROR = 28
PIPE_ENDPOINT_MAX_STEP = 10.0

# 长度方向仍由白色长矩形动态给出；横向只接受“没有被底板拉宽”的可信测量。
# 当前安装的可靠初值为 y=252~295。正常管道厚度约40~50像素，
# 因此检测框一旦高于72像素，就只更新左右端点，不允许它拉偏横向中心。
PIPE_TRANSVERSE_HALF_HEIGHT = 32
PIPE_TRANSVERSE_TRUST_MAX_HEIGHT = 72
PIPE_TRANSVERSE_CENTER_MIN = 232
PIPE_TRANSVERSE_CENTER_MAX = 317
PIPE_TRANSVERSE_TRUST_MAX_CENTER_ERROR = 20
PIPE_TRANSVERSE_MAX_STEP = 6.0

# 连续若干次管道更新失败时继续短时使用缓存矩形；
# 超过该次数后清除pipe_valid，避免长期发送错误位置。
PIPE_MAX_STALE_UPDATES = 6

# ---------------------------------------------------------------------------
# 钢球：完整保留精准版本的识别参数
# ---------------------------------------------------------------------------

BALL_THRESHOLD_DEFAULT = 125
BALL_THRESHOLD_MIN = 80
BALL_THRESHOLD_MAX = 145
BALL_DARK_OFFSET_FROM_PIPE_MEDIAN = 25

BALL_MIN_W = 3
BALL_MIN_H = 3
BALL_MAX_W = 48
BALL_MAX_H = 48
BALL_MIN_ASPECT = 0.36
BALL_MAX_ASPECT = 2.75
BALL_MIN_DENSITY = 0.16

# 锁定后只扫描小矩形。丢失1~2帧时扩大一次。
BALL_LOCAL_HALF_W = 52
BALL_LOCAL_HALF_H = 36
BALL_EXPANDED_HALF_W = 105
BALL_EXPANDED_HALF_H = 62
BALL_GLOBAL_AFTER_MISSES = 2
BALL_GLOBAL_EVERY_N = 1
# 根因是暗色背景与钢球被错误合并，不是阈值不足。
# 全局与局部均使用同一个标定阈值，避免高阈值把球粘到管内阴影。
BALL_GLOBAL_THRESHOLD_STEPS = (0,)
BALL_GLOBAL_THRESHOLD_CAP = BALL_THRESHOLD_MAX
BALL_SMOOTH_ALPHA = 0.76
BALL_PIPE_RECT_PAD = 5
BALL_HOLD_MISSES = 1
# 钢球只在管道内槽中心搜索。当前管道外框约65像素宽，
# 中心40像素足以覆盖钢球，同时排除两侧底板孔、线缆和金属反光。
BALL_PIPE_CENTER_HALF_H = 20
BALL_CENTER_MAX_OFFSET = 18
BALL_CENTER_OFFSET_SCORE_PENALTY = 1.5

# ---------------------------------------------------------------------------
# 显示与UART
# ---------------------------------------------------------------------------

ENABLE_DISPLAY = True
DISPLAY_TO_IDE = False
DISPLAY_EVERY_N = 10

ENABLE_UART = True
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_ID = 3
UART_BAUD = 460800
UART_SEND_INTERVAL_MS = 20
ENABLE_TERMINAL_LOG = False
TERMINAL_STATUS_INTERVAL_MS = 1000

GC_INTERVAL_FRAMES = 300


def terminal_print(*args):
    if ENABLE_TERMINAL_LOG:
        print(*args)


def ticks_diff(now_ms, old_ms):
    try:
        return time.ticks_diff(now_ms, old_ms)
    except Exception:
        return int(now_ms) - int(old_ms)


def ticks_add(base_ms, delta_ms):
    try:
        return time.ticks_add(base_ms, delta_ms)
    except Exception:
        return int(base_ms) + int(delta_ms)


def write_le16(buffer, offset, value):
    value = int(value) & 0xFFFF
    buffer[offset] = value & 0xFF
    buffer[offset + 1] = (value >> 8) & 0xFF


def crc16_ccitt_false(data, length):
    crc = 0xFFFF
    for index in range(int(length)):
        crc ^= (int(data[index]) & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class PipeRect:
    def __init__(self):
        self.left = float(PIPE_INITIAL_LEFT)
        self.right = float(PIPE_INITIAL_RIGHT)
        self.top = float(PIPE_INITIAL_TOP)
        self.bottom = float(PIPE_INITIAL_BOTTOM)
        self.valid = True
        self.fresh = False
        self.reason = "INITIAL"

    def x(self):
        return int(round(self.left))

    def y(self):
        return int(round(self.top))

    def w(self):
        return max(1, int(round(self.right - self.left)))

    def h(self):
        return max(1, int(round(self.bottom - self.top)))


class BallResult:
    def __init__(self):
        self.valid = False
        self.x = -1.0
        self.y = -1.0
        self.position_cm = 0.0
        self.confidence = 0
        self.reason = "INIT"
        self.rect = None
        self.search_mode = "NONE"


class PipeRectTracker:
    def __init__(self):
        self.rect = PipeRect()
        self.white_threshold = PIPE_WHITE_THRESHOLD_DEFAULT
        self.misses = 0

    def set_threshold(self, value):
        self.white_threshold = max(
            PIPE_WHITE_THRESHOLD_MIN,
            min(PIPE_WHITE_THRESHOLD_MAX, int(value)),
        )

    def update(self, image, force=False):
        self.rect.fresh = False
        try:
            blobs = image.find_blobs(
                [(self.white_threshold, 255)],
                roi=(
                    PIPE_SEARCH_X,
                    PIPE_SEARCH_Y,
                    PIPE_SEARCH_W,
                    PIPE_SEARCH_H,
                ),
                x_stride=8,
                y_stride=4,
                area_threshold=900,
                pixels_threshold=PIPE_MIN_PIXELS,
                merge=True,
                margin=4,
            )
        except Exception as error:
            self.misses += 1
            self.rect.valid = self.misses <= PIPE_MAX_STALE_UPDATES
            self.rect.reason = "PIPE_API"
            terminal_print("[PIPE] find_blobs error:", repr(error))
            return self.rect

        best = None
        best_score = -1.0
        for blob in blobs:
            width = int(blob.w())
            height = int(blob.h())
            pixels = int(blob.pixels())

            if width < PIPE_MIN_WIDTH or width > PIPE_MAX_WIDTH:
                continue
            if height < PIPE_MIN_HEIGHT or height > PIPE_MAX_HEIGHT:
                continue
            if width < height * 3:
                continue

            cx = float(blob.x()) + (width - 1) * 0.5
            old_cx = (self.rect.left + self.rect.right - 1.0) * 0.5

            # 采用本次提供代码的长白矩形评分：
            # 白色像素越多、长度越大、与历史中心越接近越优先。
            score = float(pixels) + width * 4.0 - abs(cx - old_cx)
            if score > best_score:
                best_score = score
                best = (
                    float(blob.x()),
                    float(blob.x() + width),
                    float(blob.y()),
                    float(blob.y() + height),
                )

        if best is None:
            self.misses += 1
            self.rect.valid = self.misses <= PIPE_MAX_STALE_UPDATES
            self.rect.reason = "PIPE_NOT_FOUND"
            return self.rect

        self.misses = 0
        left, right, top, bottom = best
        alpha = 1.0 if force else PIPE_SMOOTH_ALPHA

        old_left = self.rect.left
        old_right = self.rect.right
        measured_span = right - left

        # 分别判断两个端口。检测端点突然向管道内部缩进，说明该端被
        # 底板/线缆截断，只保留上一帧；正常变化则限速更新，避免跳变。
        if (
            measured_span >= PIPE_ENDPOINT_MIN_TRUSTED_SPAN
            and left <= old_left + PIPE_ENDPOINT_MAX_INWARD_ERROR
        ):
            left_delta = left - old_left
            left_delta = max(
                -PIPE_ENDPOINT_MAX_STEP,
                min(PIPE_ENDPOINT_MAX_STEP, left_delta),
            )
            target_left = old_left + left_delta
            self.rect.left = (
                alpha * target_left + (1.0 - alpha) * old_left
            )

        if (
            measured_span >= PIPE_ENDPOINT_MIN_TRUSTED_SPAN
            and right >= old_right - PIPE_ENDPOINT_MAX_INWARD_ERROR
        ):
            right_delta = right - old_right
            right_delta = max(
                -PIPE_ENDPOINT_MAX_STEP,
                min(PIPE_ENDPOINT_MAX_STEP, right_delta),
            )
            target_right = old_right + right_delta
            self.rect.right = (
                alpha * target_right + (1.0 - alpha) * old_right
            )

        # 不能直接采用合并blob的top/bottom：车体底板与管道灰度接近时，
        # merge=True会把它们连成很高的区域，导致矩形横向中心被拉偏，
        # 随后的钢球中心带也会一起偏离真实管道。
        old_center_y = (self.rect.top + self.rect.bottom - 1.0) * 0.5
        measured_height = bottom - top
        transverse_trusted = False
        if measured_height <= PIPE_TRANSVERSE_TRUST_MAX_HEIGHT:
            measured_center_y = (top + bottom - 1.0) * 0.5
            measured_center_y = max(
                PIPE_TRANSVERSE_CENTER_MIN,
                min(PIPE_TRANSVERSE_CENTER_MAX, measured_center_y),
            )
            if (
                abs(measured_center_y - old_center_y)
                <= PIPE_TRANSVERSE_TRUST_MAX_CENTER_ERROR
            ):
                center_delta = measured_center_y - old_center_y
                center_delta = max(
                    -PIPE_TRANSVERSE_MAX_STEP,
                    min(PIPE_TRANSVERSE_MAX_STEP, center_delta),
                )
                target_center_y = old_center_y + center_delta
                transverse_trusted = True

        if not transverse_trusted:
            # 异常宽框或中心跳变框只贡献左右端点，横向中心保持可信值。
            target_center_y = old_center_y

        target_top = target_center_y - PIPE_TRANSVERSE_HALF_HEIGHT
        target_bottom = target_center_y + PIPE_TRANSVERSE_HALF_HEIGHT + 1.0
        self.rect.top = target_top
        self.rect.bottom = target_bottom
        self.rect.valid = True
        self.rect.fresh = True
        if transverse_trusted:
            self.rect.reason = "OK"
        else:
            self.rect.reason = "PIPE_X_ONLY"
        return self.rect


class FastBallTracker:
    def __init__(self):
        self.gray_threshold = BALL_THRESHOLD_DEFAULT
        self.filtered_x = None
        self.filtered_y = None
        self.last_x = None
        self.last_y = None
        self.last_rect = None
        self.misses = BALL_GLOBAL_AFTER_MISSES
        self.frame_index = 0

    def set_threshold(self, value):
        self.gray_threshold = max(
            BALL_THRESHOLD_MIN,
            min(BALL_THRESHOLD_MAX, int(value)),
        )

    def _clip_roi(self, x0, y0, x1, y1, pipe):
        # 长度方向跟随动态端点；宽度方向仅保留管道中心带。
        pipe_center_y = pipe.y() + pipe.h() // 2
        pipe_half_h = min(
            BALL_PIPE_CENTER_HALF_H,
            max(4, pipe.h() // 2 + BALL_PIPE_RECT_PAD),
        )
        pipe_ball_top = pipe_center_y - pipe_half_h
        pipe_ball_bottom = pipe_center_y + pipe_half_h + 1
        left = max(
            0,
            pipe.x() - BALL_PIPE_RECT_PAD,
            int(x0),
        )
        top = max(
            0,
            pipe_ball_top,
            int(y0),
        )
        right = min(
            DETECT_WIDTH,
            pipe.x() + pipe.w() + BALL_PIPE_RECT_PAD,
            int(x1),
        )
        bottom = min(
            DETECT_HEIGHT,
            pipe_ball_bottom,
            int(y1),
        )
        if right - left < 4 or bottom - top < 4:
            return None
        return (left, top, right - left, bottom - top)

    def _choose_roi(self, pipe):
        self.frame_index += 1
        if self.last_x is not None and self.misses == 0:
            roi = self._clip_roi(
                self.last_x - BALL_LOCAL_HALF_W,
                self.last_y - BALL_LOCAL_HALF_H,
                self.last_x + BALL_LOCAL_HALF_W + 1,
                self.last_y + BALL_LOCAL_HALF_H + 1,
                pipe,
            )
            return roi, "LOCAL", 1, 1

        if self.last_x is not None and self.misses < BALL_GLOBAL_AFTER_MISSES:
            roi = self._clip_roi(
                self.last_x - BALL_EXPANDED_HALF_W,
                self.last_y - BALL_EXPANDED_HALF_H,
                self.last_x + BALL_EXPANDED_HALF_W + 1,
                self.last_y + BALL_EXPANDED_HALF_H + 1,
                pipe,
            )
            return roi, "EXPAND", 1, 1

        if self.frame_index % BALL_GLOBAL_EVERY_N != 0:
            return None, "WAIT_GLOBAL", 1, 1
        return (
            self._clip_roi(0, 0, DETECT_WIDTH, DETECT_HEIGHT, pipe),
            "GLOBAL",
            # 冷启动时远距离球可能只有几颗暗像素，2x2步长可能完全跳过。
            # 全局阶段使用1x1；锁定后仍回到小型LOCAL ROI，正常帧率不受影响。
            1,
            1,
        )

    def detect(self, image, pipe):
        result = BallResult()
        roi, mode, x_stride, y_stride = self._choose_roi(pipe)
        result.search_mode = mode
        if roi is None:
            result.reason = mode
            return result

        search_threshold = self.gray_threshold
        if mode == "GLOBAL":
            step_index = self.frame_index % len(
                BALL_GLOBAL_THRESHOLD_STEPS
            )
            search_threshold = min(
                BALL_GLOBAL_THRESHOLD_CAP,
                self.gray_threshold
                + BALL_GLOBAL_THRESHOLD_STEPS[step_index],
            )

        try:
            blobs = image.find_blobs(
                [(0, search_threshold)],
                roi=roi,
                x_stride=x_stride,
                y_stride=y_stride,
                area_threshold=3,
                pixels_threshold=3,
                # 不能合并：倾斜管道矩形中的蓝背景是巨大暗blob，
                # 其包围框覆盖钢球；merge=True会把两者合成一个大框。
                merge=False,
                margin=0,
            )
        except Exception as error:
            result.reason = "BALL_API"
            terminal_print("[BALL] find_blobs error:", repr(error))
            return result

        best = None
        best_score = -1.0
        pipe_center_y = pipe.y() + pipe.h() * 0.5
        for blob in blobs:
            width = int(blob.w())
            height = int(blob.h())
            if (
                width < BALL_MIN_W
                or height < BALL_MIN_H
                or width > BALL_MAX_W
                or height > BALL_MAX_H
            ):
                continue
            aspect = float(width) / float(max(1, height))
            if aspect < BALL_MIN_ASPECT or aspect > BALL_MAX_ASPECT:
                continue
            density = float(blob.density())
            if density < BALL_MIN_DENSITY:
                continue

            cx = float(blob.x()) + (width - 1) * 0.5
            cy = float(blob.y()) + (height - 1) * 0.5
            center_offset = abs(cy - pipe_center_y)
            if center_offset > BALL_CENTER_MAX_OFFSET:
                continue
            pixels = int(blob.pixels())
            roundness = min(width, height) / float(max(width, height))
            score = float(pixels) * (0.50 + 0.50 * roundness)
            score *= 0.72 + 0.28 * density
            score -= center_offset * BALL_CENTER_OFFSET_SCORE_PENALTY
            if self.last_x is not None:
                distance = abs(cx - self.last_x) + abs(cy - self.last_y)
                score += max(0.0, 14.0 - distance * 0.10)

            if score > best_score:
                best_score = score
                best = (cx, cy, width, height, density, pixels)

        if best is None:
            self.misses += 1
            if (
                self.filtered_x is not None
                and self.misses <= BALL_HOLD_MISSES
            ):
                axis_length = max(
                    1.0, pipe.right - pipe.left - 1.0
                )
                ratio = (self.filtered_x - pipe.left) / axis_length
                ratio = max(0.0, min(1.0, ratio))
                result.valid = True
                result.x = self.filtered_x
                result.y = self.filtered_y
                result.position_cm = ratio * PIPE_LENGTH_CM
                result.confidence = 20
                result.reason = "HOLD_1"
                result.search_mode = "HOLD"
                result.rect = self.last_rect
                return result
            result.reason = "NO_BLOB_%s_T%d" % (
                mode, search_threshold
            )
            return result

        raw_x, raw_y, width, height, density, pixels = best
        misses_before_detection = self.misses
        self.last_x = raw_x
        self.last_y = raw_y
        self.misses = 0
        self.last_rect = (
            int(round(raw_x - (width - 1) * 0.5)),
            int(round(raw_y - (height - 1) * 0.5)),
            width,
            height,
        )

        if (
            self.filtered_x is None
            or misses_before_detection >= BALL_GLOBAL_AFTER_MISSES
        ):
            self.filtered_x = raw_x
            self.filtered_y = raw_y
        else:
            alpha = BALL_SMOOTH_ALPHA
            self.filtered_x = (
                alpha * raw_x + (1.0 - alpha) * self.filtered_x
            )
            self.filtered_y = (
                alpha * raw_y + (1.0 - alpha) * self.filtered_y
            )

        axis_length = max(1.0, pipe.right - pipe.left - 1.0)
        ratio = (self.filtered_x - pipe.left) / axis_length
        ratio = max(0.0, min(1.0, ratio))

        result.valid = True
        result.x = self.filtered_x
        result.y = self.filtered_y
        result.position_cm = ratio * PIPE_LENGTH_CM
        result.confidence = max(
            1,
            min(100, int(round(45.0 + 35.0 * density + pixels * 0.15))),
        )
        result.reason = "OK"
        result.rect = self.last_rect
        return result


class PacketBuilder:
    """保持 BALL_PACKET_V1 14字节协议不变。"""

    FRAME_SIZE = 14

    def __init__(self):
        self.sequence = 0
        self.buffer = bytearray(self.FRAME_SIZE)

    def build(self, pipe_valid, result):
        buffer = self.buffer
        buffer[0] = 0xAA
        buffer[1] = 0x55
        buffer[2] = 0x01
        buffer[3] = 0x21
        write_le16(buffer, 4, self.sequence)

        flags = 0
        if result.valid:
            flags |= 0x01
        if pipe_valid:
            flags |= 0x02
        buffer[6] = flags
        buffer[7] = (
            max(0, min(100, int(result.confidence)))
            if result.valid else 0
        )

        if result.valid:
            write_le16(buffer, 8, int(round(result.x)))
            write_le16(
                buffer,
                10,
                max(0, min(2500, int(round(result.position_cm * 100.0)))),
            )
        else:
            write_le16(buffer, 8, 0xFFFF)
            write_le16(buffer, 10, 0xFFFF)

        write_le16(buffer, 12, crc16_ccitt_false(buffer, 12))
        self.sequence = (self.sequence + 1) & 0xFFFF
        return buffer


class UARTLink:
    def __init__(self):
        self.uart = None
        if not ENABLE_UART:
            terminal_print("[UART] disabled")
            return
        fpioa = FPIOA()
        fpioa.set_function(
            UART_TX_PIN, FPIOA.UART3_TXD, ie=0, oe=1
        )
        fpioa.set_function(
            UART_RX_PIN, FPIOA.UART3_RXD, ie=1, oe=0
        )
        uart_id = getattr(UART, "UART3", UART_ID)
        try:
            self.uart = UART(
                uart_id,
                baudrate=UART_BAUD,
                bits=8,
                parity=None,
                stop=1,
                timeout=0,
            )
        except TypeError:
            self.uart = UART(uart_id, baudrate=UART_BAUD)
        terminal_print(
            "[UART] GPIO%d/%d UART3 @ %d"
            % (UART_TX_PIN, UART_RX_PIN, UART_BAUD)
        )

    def send(self, data):
        if self.uart is not None:
            self.uart.write(data)

    def deinit(self):
        if self.uart is not None:
            try:
                self.uart.deinit()
            except Exception:
                pass


def draw_debug(image, pipe, ball, fps):
    # 显示通道是RGB565，因此可以使用真正的彩色叠加。
    # 管道矩形固定使用绿色。
    pipe_color = (0, 255, 0)
    ball_color = (255, 0, 0)

    image.draw_rectangle(
        pipe.x(),
        pipe.y(),
        pipe.w(),
        pipe.h(),
        color=pipe_color,
        thickness=3,
    )

    center_y = pipe.y() + pipe.h() // 2
    image.draw_circle(
        pipe.x(),
        center_y,
        5,
        color=pipe_color,
        thickness=2,
    )
    image.draw_circle(
        pipe.x() + pipe.w() - 1,
        center_y,
        5,
        color=pipe_color,
        thickness=2,
    )

    if ball.valid:
        image.draw_circle(
            int(round(ball.x)),
            int(round(ball.y)),
            7,
            color=ball_color,
            thickness=3,
        )
        image.draw_cross(
            int(round(ball.x)),
            int(round(ball.y)),
            color=ball_color,
            size=10,
            thickness=2,
        )

    if ball.valid:
        position_text = "POS:%.2fcm" % ball.position_cm
        position_color = (255, 255, 0)
    else:
        position_text = "POS:LOST"
        position_color = ball_color

    image.draw_string_advanced(
        4,
        4,
        24,
        position_text,
        color=position_color,
    )
    image.draw_string_advanced(
        4,
        32,
        24,
        "FPS:%.1f" % fps,
        color=(0, 255, 255),
    )


def main():
    sensor = None
    uart = None
    display_ok = False
    media_initialized = False

    pipe_tracker = PipeRectTracker()
    ball_tracker = FastBallTracker()
    packet_builder = PacketBuilder()
    pipe = pipe_tracker.rect
    ball = BallResult()

    frame_count = 0
    fps = 0.0
    fps_start_ms = 0
    fps_start_frames = 0
    next_send_ms = 0
    last_status_ms = 0

    try:
        terminal_print("=" * 64)
        terminal_print("BUILD:", BUILD_ID)
        terminal_print("MODE: LOW-RATE PIPE RECT + LOCAL BALL ROI")
        terminal_print("DISPLAY:", 1 if ENABLE_DISPLAY else 0)
        terminal_print("=" * 64)

        uart = UARTLink()
        sensor = Sensor(
            id=SENSOR_ID,
            width=CAM_INPUT_WIDTH,
            height=CAM_INPUT_HEIGHT,
            fps=CAM_FPS,
        )
        sensor.reset()
        # CH0：灰度检测，保留精准钢球识别。
        sensor.set_framesize(
            width=DETECT_WIDTH,
            height=DETECT_HEIGHT,
            chn=DETECT_CHANNEL,
        )
        sensor.set_pixformat(
            Sensor.GRAYSCALE,
            chn=DETECT_CHANNEL,
        )

        # CH1：RGB565彩色显示，专门用于绿色管道框和红色钢球标记。
        sensor.set_framesize(
            width=DETECT_WIDTH,
            height=DETECT_HEIGHT,
            chn=DISPLAY_CHANNEL,
        )
        sensor.set_pixformat(
            Sensor.RGB565,
            chn=DISPLAY_CHANNEL,
        )
        if SENSOR_HMIRROR:
            sensor.set_hmirror(True)
        if SENSOR_VFLIP:
            sensor.set_vflip(True)

        if ENABLE_DISPLAY:
            Display.init(
                Display.ST7701,
                width=640,
                height=480,
                to_ide=DISPLAY_TO_IDE,
                quality=35,
            )
            display_ok = True
            terminal_print("[DISPLAY] ST7701 ready")

        MediaManager.init()
        media_initialized = True
        sensor.run()
        time.sleep_ms(300)

        image = None
        for _ in range(12):
            os.exitpoint()
            image = sensor.snapshot(chn=DETECT_CHANNEL)

        # 启动时先用宽松阈值找管道矩形。
        pipe = pipe_tracker.update(image, force=True)
        try:
            stats = image.get_statistics(
                roi=(pipe.x(), pipe.y(), pipe.w(), pipe.h())
            )
            pipe_uq = int(stats.uq())
            pipe_tracker.set_threshold(
                pipe_uq - PIPE_WHITE_OFFSET_FROM_UQ
            )
            pipe_median = int(stats.median())
            ball_tracker.set_threshold(
                pipe_median - BALL_DARK_OFFSET_FROM_PIPE_MEDIAN
            )
            terminal_print(
                "[CAL] pipe_uq=%d pipe_med=%d pipe_T=%d ball_T=%d"
                % (
                    pipe_uq,
                    pipe_median,
                    pipe_tracker.white_threshold,
                    ball_tracker.gray_threshold,
                )
            )
        except Exception as error:
            terminal_print("[CAL] defaults used:", repr(error))

        # 使用更新后的阈值再精确锁定一次。
        pipe = pipe_tracker.update(image, force=True)
        terminal_print(
            "[PIPE] lock x=%d y=%d w=%d h=%d"
            % (pipe.x(), pipe.y(), pipe.w(), pipe.h())
        )
        terminal_print("[RUN] main loop")

        fps_start_ms = time.ticks_ms()
        last_status_ms = fps_start_ms
        next_send_ms = ticks_add(
            fps_start_ms,
            UART_SEND_INTERVAL_MS,
        )

        while True:
            os.exitpoint()
            image = sensor.snapshot(chn=DETECT_CHANNEL)
            frame_count += 1

            if frame_count % PIPE_UPDATE_EVERY_N == 0:
                pipe = pipe_tracker.update(image)
            else:
                pipe.fresh = False

            if pipe.valid:
                ball = ball_tracker.detect(image, pipe)
            else:
                ball = BallResult()
                ball.reason = "PIPE_INVALID"

            now_ms = time.ticks_ms()

            if ticks_diff(now_ms, next_send_ms) >= 0:
                uart.send(packet_builder.build(pipe.valid, ball))
                next_send_ms = ticks_add(
                    next_send_ms,
                    UART_SEND_INTERVAL_MS,
                )

                # 如果主循环偶尔卡顿，只跳过已错过的发送时刻。
                # 不连续补发旧坐标，下一包仍保持在固定 20 ms 时间轴上。
                while ticks_diff(now_ms, next_send_ms) >= 0:
                    next_send_ms = ticks_add(
                        next_send_ms,
                        UART_SEND_INTERVAL_MS,
                    )

            elapsed = ticks_diff(now_ms, fps_start_ms)
            if elapsed >= 1000:
                frames = frame_count - fps_start_frames
                fps = frames * 1000.0 / float(max(1, elapsed))
                fps_start_ms = now_ms
                fps_start_frames = frame_count

            if display_ok and frame_count % DISPLAY_EVERY_N == 0:
                # 仅显示时抓取一次RGB565帧；检测仍使用灰度CH0。
                display_image = sensor.snapshot(chn=DISPLAY_CHANNEL)
                draw_debug(display_image, pipe, ball, fps)
                Display.show_image(display_image, x=0, y=0)

            if (
                ENABLE_TERMINAL_LOG
                and ticks_diff(now_ms, last_status_ms)
                >= TERMINAL_STATUS_INTERVAL_MS
            ):
                last_status_ms = now_ms
                if ball.valid:
                    terminal_print(
                        "[STAT] fps=%.1f pipe=%s x=%d..%d "
                        "ball=OK x=%.1f pos=%.2fcm mode=%s"
                        % (
                            fps,
                            pipe.reason,
                            pipe.x(),
                            pipe.x() + pipe.w() - 1,
                            ball.x,
                            ball.position_cm,
                            ball.search_mode,
                        )
                    )
                else:
                    terminal_print(
                        "[STAT] fps=%.1f pipe=%s ball=LOST reason=%s"
                        % (fps, pipe.reason, ball.reason)
                    )

            if frame_count % GC_INTERVAL_FRAMES == 0:
                gc.collect()

    except KeyboardInterrupt:
        terminal_print("[STOP] user interrupted")
    except BaseException as error:
        terminal_print("[FATAL]", repr(error))
        if ENABLE_TERMINAL_LOG:
            try:
                sys.print_exception(error)
            except Exception:
                pass
    finally:
        terminal_print("[CLEANUP] start")
        try:
            if uart is not None:
                uart.send(packet_builder.build(False, BallResult()))
        except Exception:
            pass
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
        if display_ok:
            try:
                Display.deinit()
            except Exception:
                pass
        try:
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        except Exception:
            pass
        time.sleep_ms(100)
        if media_initialized:
            try:
                MediaManager.deinit()
            except Exception:
                pass
        if uart is not None:
            uart.deinit()
        terminal_print("[CLEANUP] done")


if __name__ == "__main__":
    main()
