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


BUILD_ID = "K230_BALL_RECT_LOCAL_TRACK_20260729_07"

# ---------------------------------------------------------------------------
# 摄像头
# ---------------------------------------------------------------------------

SENSOR_ID = 2
CAM_INPUT_WIDTH = 1280
CAM_INPUT_HEIGHT = 960
CAM_FPS = 90
DETECT_WIDTH = 640
DETECT_HEIGHT = 480
SENSOR_HMIRROR = False
SENSOR_VFLIP = False

# ---------------------------------------------------------------------------
# 管道矩形
# ---------------------------------------------------------------------------

PIPE_LENGTH_CM = 25.0

# 由板端 480x360 的 left=38 right=458 top=124 bottom=155 按 4/3 换算。
PIPE_INITIAL_LEFT = 51
PIPE_INITIAL_RIGHT = 612       # exclusive
PIPE_INITIAL_TOP = 165
PIPE_INITIAL_BOTTOM = 208      # exclusive

# 只在该范围搜索白色管道，排除画面下方机构。
PIPE_SEARCH_X = 15
PIPE_SEARCH_Y = 65
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

# ---------------------------------------------------------------------------
# 钢球
# ---------------------------------------------------------------------------

BALL_THRESHOLD_DEFAULT = 170
BALL_THRESHOLD_MIN = 100
BALL_THRESHOLD_MAX = 185
BALL_DARK_OFFSET_FROM_PIPE_UQ = 48

BALL_MIN_W = 3
BALL_MIN_H = 3
BALL_MAX_W = 42
BALL_MAX_H = 42
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

# ---------------------------------------------------------------------------
# 显示与UART
# ---------------------------------------------------------------------------

ENABLE_DISPLAY = True
DISPLAY_TO_IDE = False
DISPLAY_EVERY_N = 30

ENABLE_UART = True
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_ID = 3
UART_BAUD = 460800
UART_SEND_INTERVAL_MS = 20
ENABLE_TERMINAL_LOG = False

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

            # 长、白色像素多的目标优先；小幅偏好与旧矩形接近者。
            cx = float(blob.x()) + (width - 1) * 0.5
            old_cx = (self.rect.left + self.rect.right - 1.0) * 0.5
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
            self.rect.reason = "PIPE_NOT_FOUND"
            return self.rect

        left, right, top, bottom = best
        if force:
            alpha = 1.0
        else:
            alpha = PIPE_SMOOTH_ALPHA
        self.rect.left = alpha * left + (1.0 - alpha) * self.rect.left
        self.rect.right = alpha * right + (1.0 - alpha) * self.rect.right
        self.rect.top = alpha * top + (1.0 - alpha) * self.rect.top
        self.rect.bottom = alpha * bottom + (1.0 - alpha) * self.rect.bottom
        self.rect.valid = True
        self.rect.fresh = True
        self.rect.reason = "OK"
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
        # 球只可能出现在缓存管道矩形内。
        left = max(
            0, pipe.x() - BALL_PIPE_RECT_PAD, int(x0)
        )
        top = max(
            0, pipe.y() - BALL_PIPE_RECT_PAD, int(y0)
        )
        right = min(
            DETECT_WIDTH,
            pipe.x() + pipe.w() + BALL_PIPE_RECT_PAD,
            int(x1),
        )
        bottom = min(
            DETECT_HEIGHT,
            pipe.y() + pipe.h() + BALL_PIPE_RECT_PAD,
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
            pixels = int(blob.pixels())
            roundness = min(width, height) / float(max(width, height))
            score = float(pixels) * (0.50 + 0.50 * roundness)
            score *= 0.72 + 0.28 * density
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


def draw_debug(image, pipe, ball):
    image.draw_rectangle(
        pipe.x(),
        pipe.y(),
        pipe.w(),
        pipe.h(),
        color=255,
        thickness=2,
    )
    center_y = pipe.y() + pipe.h() // 2
    image.draw_circle(
        pipe.x(), center_y, 4, color=255, thickness=2
    )
    image.draw_circle(
        pipe.x() + pipe.w() - 1,
        center_y,
        4,
        color=255,
        thickness=2,
    )
    if ball.valid:
        image.draw_circle(
            int(round(ball.x)),
            int(round(ball.y)),
            6,
            color=255,
            thickness=2,
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
        sensor.set_framesize(
            width=DETECT_WIDTH,
            height=DETECT_HEIGHT,
            chn=CAM_CHN_ID_0,
        )
        sensor.set_pixformat(Sensor.GRAYSCALE, chn=CAM_CHN_ID_0)
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
            image = sensor.snapshot(chn=CAM_CHN_ID_0)

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
            ball_tracker.set_threshold(
                pipe_uq - BALL_DARK_OFFSET_FROM_PIPE_UQ
            )
            terminal_print(
                "[CAL] pipe_uq=%d pipe_T=%d ball_T=%d"
                % (
                    pipe_uq,
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
        next_send_ms = ticks_add(
            fps_start_ms,
            UART_SEND_INTERVAL_MS,
        )

        while True:
            os.exitpoint()
            image = sensor.snapshot(chn=CAM_CHN_ID_0)
            frame_count += 1

            if frame_count % PIPE_UPDATE_EVERY_N == 0:
                pipe = pipe_tracker.update(image)
            else:
                pipe.fresh = False

            ball = ball_tracker.detect(image, pipe)
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

            if ENABLE_TERMINAL_LOG:
                elapsed = ticks_diff(now_ms, fps_start_ms)
                if elapsed >= 1000:
                    frames = frame_count - fps_start_frames
                    fps = frames * 1000.0 / float(max(1, elapsed))
                    fps_start_ms = now_ms
                    fps_start_frames = frame_count

            if display_ok and frame_count % DISPLAY_EVERY_N == 0:
                draw_debug(image, pipe, ball)
                Display.show_image(image, x=0, y=0)

            if ENABLE_TERMINAL_LOG and frame_count % 120 == 0:
                terminal_print(
                    "[PIPE] x=%d y=%d w=%d h=%d fresh=%d %s"
                    % (
                        pipe.x(),
                        pipe.y(),
                        pipe.w(),
                        pipe.h(),
                        1 if pipe.fresh else 0,
                        pipe.reason,
                    )
                )
                if ball.valid:
                    terminal_print(
                        "[BALL] x=%.1f pos=%.2fcm %s w=%d h=%d fps=%.1f"
                        % (
                            ball.x,
                            ball.position_cm,
                            ball.search_mode,
                            ball.rect[2],
                            ball.rect[3],
                            fps,
                        )
                    )
                else:
                    terminal_print(
                        "[BALL] LOST %s fps=%.1f"
                        % (ball.reason, fps)
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
