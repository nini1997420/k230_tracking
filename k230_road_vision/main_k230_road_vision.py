# -*- coding: utf-8 -*-
"""
=============================================================================
K230 道路循迹视觉程序 (Road Vision) — 板端主程序
=============================================================================

基于 K230 / CanMV MicroPython v1.4.3，实现车载道路视觉感知。

职责边界：
  K230 只负责：摄像头采集 → 道路边界提取 → 几何偏差计算 → 路口结构识别
            → 可信度评估 → 二进制帧发送给 MSPM0
  K230 不负责：电机PWM、车轮速度闭环、路线决策、转弯控制

MSPM0 接收端未确认：当前通信协议为版本化草案，接收端源码不在本仓库。
                    联调前需与接收端逐字节核对帧格式。

参考：
  - 我chovy.txt（任务职责最高优先级）
  - K230_现有项目硬件与编程使用手册.md（生命周期约束）
  - main_k230_vision_uart_plan_b_v1.py（UART/CRC 可复用写法）

部署：
  本文件可独立部署，也可用 main.py（合并版）部署。
  模块化源码保留在 k230_road_vision/ 目录用于版本管理。
=============================================================================
"""

import time
import os
import gc
import sys

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

# ---- 项目模块 ----
from road_config import *
from road_geometry import RoadBoundaryExtractor, RoadGeometry, VisionState
from road_structure import RoadStructureDetector, StructureResult, JunctionStage
from vision_protocol import (
    RoadFrameBuilder, RoadFrameDecoder, crc16_ccitt_false,
    FRAME_SIZE, CRC_DATA_SIZE,
    MODE_IDLE, MODE_TRACK, MODE_TURNING, MODE_REACQUIRE, MODE_FAULT, MODE_NUMBER,
    FLAG_VISION_VALID, FLAG_DEGRADED, FLAG_LEFT_VALID, FLAG_RIGHT_VALID,
    FLAG_LEFT_BRANCH, FLAG_RIGHT_BRANCH, FLAG_INTERSECTION,
    ANOMALY_BLUR, ANOMALY_OVEREXPOSED, ANOMALY_UNDEREXPOSED, ANOMALY_EDGE_NOISE,
    INVALID_S16, INVALID_U16,
)


# ============================================================================
# 图像预处理工具（板端实现）
# ============================================================================

class ImagePreprocessor:
    """
    板端图像预处理：RGB888 → 灰度 + 异常检测。

    使用 CanMV to_grayscale() + to_numpy_ref() 高效获取灰度 ndarray，
    仅读取采样行，绝不复制全图，绝不逐像素 get_pixel 双重循环。
    """

    _shape_printed = False

    def __init__(self, sample_y_list):
        self.sample_y = sample_y_list  # 12 个采样行 Y 坐标（由 RoadBoundaryExtractor 预计算）
        self.roi_left = ROI_LEFT
        self.roi_right = ROI_RIGHT
        self.detect_w = DETECT_WIDTH

    def extract_rows(self, img):
        """
        从 CanMV image 提取灰度 ndarray 引用 + 异常检测标志。

        返回 (gray_np, anomaly_flags)
          gray_np: 灰度 numpy ref，shape [H, W] 或 [H, W, C]，可直接下标访问
          anomaly_flags: u8 异常标志
        失败时返回 (None, 0)，打印清晰警告。
        """
        # 1. 转换为灰度（在副本上操作，不修改原图）
        try:
            gray = img.copy()
            gray.to_grayscale()
        except Exception:
            print("[PREPROC] img.copy() or to_grayscale() failed")
            return None, 0

        # 2. 获取 numpy 引用
        try:
            gray_np = gray.to_numpy_ref()
        except Exception:
            print("[PREPROC] WARNING: to_numpy_ref() not available or failed — "
                  "cannot extract grayscale data")
            return None, 0

        # 3. 打印 shape 一次
        if not ImagePreprocessor._shape_printed:
            try:
                print("[PREPROC] ndarray shape:", gray_np.shape)
            except Exception:
                pass
            ImagePreprocessor._shape_printed = True

        # 4. 检测异常（从采样行中再子采样，避免遍历全图）
        flags = 0
        ndim = len(gray_np.shape)
        sample_vals = []

        if ndim == 2:
            # [y, x] 格式
            for y in self.sample_y:
                if y < gray_np.shape[0]:
                    row = gray_np[y, self.roi_left:self.roi_right]
                    step = max(1, len(row) // 20)
                    for j in range(0, len(row), step):
                        sample_vals.append(int(row[j]))
        elif ndim == 3:
            # [y, x, c] 格式 — 取 channel 0
            h = gray_np.shape[0]
            for y in self.sample_y:
                if y >= h:
                    continue
                row = gray_np[y, self.roi_left:self.roi_right, 0]
                step = max(1, len(row) // 20)
                for j in range(0, len(row), step):
                    sample_vals.append(int(row[j]))
        else:
            print("[PREPROC] unexpected ndim:", ndim)
            # 仍然返回 gray_np 引用（调用方可能仍用它做边界提取）
            return gray_np, 0

        if sample_vals:
            avg = sum(sample_vals) / len(sample_vals)
            if avg < 25:
                flags |= ANOMALY_UNDEREXPOSED
            elif avg > 230:
                flags |= ANOMALY_OVEREXPOSED

        return gray_np, flags


# ============================================================================
# UART 通信管理（板端）
# ============================================================================

class RoadUART:
    """K230 板端 UART3 通信管理。"""

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
        """发送原始字节。"""
        try:
            if self.uart and raw_bytes:
                self.uart.write(raw_bytes)
        except Exception:
            pass

    def recv(self, max_pkts=4):
        """非阻塞接收 ASCII 命令包，格式 [cmd,arg,...]"""
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


# ============================================================================
# 模式状态机
# ============================================================================

class ModeStateMachine:
    """
    K230 运行模式状态机。

    模式：
      IDLE       — 等待 MSPM0 命令，低频或暂停分析
      TRACK      — 持续道路边界、偏差和路口检测
      TURNING    — MSPM0 正在转弯，结果标记暂不可用
      REACQUIRE  — 转弯后重捕获，需连续多帧稳定才恢复
      FAULT      — 异常，发送视觉无效
      NUMBER     — 数字识别模式（第一阶段保留接口，默认关闭）
    """

    def __init__(self):
        self.mode = MODE_IDLE
        self.prev_mode = MODE_IDLE
        self.mode_changed = False
        self.reacquire_count = 0
        # 回调引用（由 K230RoadVision 在 init 时注入）
        self._on_turning_to_reacquire_cb = None

    def set_mode(self, new_mode):
        if new_mode != self.mode:
            prev = self.mode
            self.prev_mode = prev
            self.mode = new_mode
            self.mode_changed = True
            self.reacquire_count = 0

            # 从 TURNING 进入 REACQUIRE 时触发重置回调
            if prev == MODE_TURNING and new_mode == MODE_REACQUIRE:
                if self._on_turning_to_reacquire_cb is not None:
                    try:
                        self._on_turning_to_reacquire_cb()
                    except Exception:
                        pass

    def update(self, road_geom, struct_result):
        """根据视觉结果更新 REACQUIRE 状态。"""
        if self.mode == MODE_REACQUIRE:
            if road_geom.vision_state == VisionState.NORMAL and \
               road_geom.confidence >= CONF_HIGH_THRESH:
                self.reacquire_count += 1
            else:
                self.reacquire_count = max(0, self.reacquire_count - 1)

            if self.reacquire_count >= REACQUIRE_CONFIRM_FRAMES:
                self.set_mode(MODE_TRACK)


# ============================================================================
# 主程序
# ============================================================================

class K230RoadVision:
    """K230 道路循迹主程序。"""

    def __init__(self):
        self.mode_fsm = ModeStateMachine()

        # 先构建 BoundaryExtractor，以便获取 sample_y 用于 Preprocessor
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
            scale_x=2.0, scale_y=2.0,   # 320→640, 240→480
        )
        self.structure_detector = RoadStructureDetector()
        self.frame_builder = RoadFrameBuilder()
        self.preprocessor = ImagePreprocessor(self.boundary_extractor.sample_y)

        # 注入 TURNING → REACQUIRE 回调
        self.mode_fsm._on_turning_to_reacquire_cb = self._on_turning_to_reacquire

        self.uart = None       # 延迟初始化，保证安全状态先建立
        self.frame_count = 0
        self.fps = 0.0
        self.last_send_ms = 0
        self.last_status_ms = 0
        self.last_fps_calc_ms = 0
        self.last_fps_count = 0

        # 显示状态
        self.display_ok = False
        self.display_x = (640 - DETECT_WIDTH) // 2
        self.display_y = (480 - DETECT_HEIGHT) // 2

    def _on_turning_to_reacquire(self):
        """TURNING → REACQUIRE 转换时重置几何历史与结构检测器。"""
        self.road_geom.reset_history()
        self.structure_detector.on_turning_complete()

    def _should_output_valid(self):
        """判断当前模式是否应该输出有效的几何数据。"""
        m = self.mode_fsm.mode
        if m in (MODE_IDLE, MODE_FAULT, MODE_TURNING, MODE_NUMBER):
            return False
        if m == MODE_REACQUIRE:
            return self.mode_fsm.reacquire_count >= REACQUIRE_CONFIRM_FRAMES
        return True  # MODE_TRACK

    def run(self):
        sensor = None

        try:
            # ============================================================
            # 安全状态的通信对象先创建
            # ============================================================
            self.uart = RoadUART(UART_TX_PIN, UART_RX_PIN, UART_ID, UART_BAUD)

            # ============================================================
            # Sensor 创建与配置（单通道，不启用双 CAM_CHN）
            # ============================================================
            print("=" * 55)
            print("K230 Road Vision v1.0")
            print("  DETECT: %dx%d  ROI: [%d:%d, %d:%d]" % (
                DETECT_WIDTH, DETECT_HEIGHT, ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT))
            print("  UART3: GPIO%d/GPIO%d @ %d baud" % (
                UART_TX_PIN, UART_RX_PIN, UART_BAUD))
            print("  MSPM0 receiver: UNCONFIRMED (protocol v0x%02X)" % (
                vision_protocol.PROTOCOL_VERSION if "vision_protocol" in dir() else 0x02))
            print("=" * 55)

            try:
                sensor = Sensor(
                    id=SENSOR_ID,
                    width=CAM_INPUT_WIDTH,
                    height=CAM_INPUT_HEIGHT,
                    fps=CAM_FPS,
                )
            except Exception as e:
                print("[FATAL] Sensor init failed:", e)
                raise
            sensor.reset()

            # 单通道配置：CH0 用于道路检测
            sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT,
                                 chn=CAM_CHN_ID_0)
            sensor.set_pixformat(Sensor.RGB888, chn=CAM_CHN_ID_0)

            # ============================================================
            # Display → MediaManager → sensor.run
            # ============================================================
            if ENABLE_DISPLAY:
                Display.init(Display.ST7701, width=640, height=480,
                             to_ide=DISPLAY_TO_IDE, quality=DISPLAY_QUALITY)
                self.display_ok = True

            MediaManager.init()
            sensor.run()

            # ============================================================
            # 预热
            # ============================================================
            print("warming up camera...")
            time.sleep_ms(300)
            for i in range(15):
                os.exitpoint()
                try:
                    img = sensor.snapshot(chn=CAM_CHN_ID_0)
                except Exception as e:
                    print("[FATAL] warmup snapshot %d failed:" % i, e)
                    raise
                time.sleep_ms(10)
            print("[CAM] warmup completed")

            # ============================================================
            # 一次性摄像头诊断信息
            # ============================================================
            print("[CAM] SENSOR_ID=%d  input=%dx%d@%dfps" % (
                SENSOR_ID, CAM_INPUT_WIDTH, CAM_INPUT_HEIGHT, CAM_FPS))
            print("[CAM] CAM_CHN_ID_0  detect=%dx%d  pixformat=RGB888" % (
                DETECT_WIDTH, DETECT_HEIGHT))
            try:
                print("[CAM] first image: width=%d height=%d" % (img.width(), img.height()))
            except Exception:
                pass

            # 初始模式设为 TRACK，也可等待 MSPM0 命令切换
            self.mode_fsm.set_mode(MODE_TRACK)
            self.uart.send_raw(self._build_status_frame())

            now_ms = time.ticks_ms()
            self.last_send_ms = now_ms
            self.last_status_ms = now_ms
            self.last_fps_calc_ms = now_ms

            print("entering main loop...")
            print("commands: [mode,idle] [mode,track] [mode,turning] [mode,fault]")
            print("-" * 55)

            # ============================================================
            # 主循环
            # ============================================================
            while True:
                os.exitpoint()
                loop_start = time.ticks_ms()

                # ---- 取帧 ----
                img = sensor.snapshot(chn=CAM_CHN_ID_0)

                # ---- 处理 UART 命令 ----
                self._handle_commands()

                # ---- 按模式分支 ----
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

                # ---- 图像预处理（所有非 IDLE/FAULT 模式都执行） ----
                gray_np, anomaly = self.preprocessor.extract_rows(img)
                if gray_np is None:
                    # 预处理失败，发送无效帧
                    if self.uart:
                        self.uart.send_raw(self._build_invalid_frame())
                    self._update_fps(loop_start)
                    continue

                # ---- 边界提取 ----
                boundary = self.boundary_extractor.extract(gray_np)

                # ---- TURNING / NUMBER 模式：跳过几何计算，仅维持图像管线 ----
                if self.mode_fsm.mode in (MODE_TURNING, MODE_NUMBER):
                    # 结构检测仍运行（保持状态机内部状态）
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

                # ---- 道路几何 ----
                geom = self.road_geom.compute(boundary, now_ms=loop_start)

                # ---- 结构识别 ----
                struct = self.structure_detector.detect(boundary, geom)

                # ---- 模式状态机更新 ----
                self.mode_fsm.update(geom, struct)

                # ---- 构建二进制帧 ----
                elapsed = ticks_ms_diff(loop_start, now_ms)
                if ticks_ms_diff(loop_start, self.last_send_ms) >= FRAME_SEND_MIN_MS:
                    self.last_send_ms = loop_start
                    frame = self._build_road_frame(geom, struct, anomaly, loop_start)
                    if self.uart:
                        self.uart.send_raw(frame)

                # ---- 状态打印（低频） ----
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

                # ---- 显示（低频） ----
                if self.display_ok and self.frame_count % DISPLAY_EVERY_N == 0:
                    self._draw_overlay(img, boundary, geom)
                    try:
                        Display.show_image(img, x=self.display_x, y=self.display_y)
                    except Exception:
                        pass

                # ---- FPS 和 GC ----
                self._update_fps(loop_start)

                if self.frame_count % GC_INTERVAL_FRAMES == 0:
                    try:
                        if gc.mem_free() < GC_FREE_THRESH:
                            gc.collect()
                    except Exception:
                        pass

                # 帧间休眠
                elapsed = ticks_ms_diff(time.ticks_ms(), loop_start)
                if elapsed < 8:
                    time.sleep_ms(8 - elapsed)

        except KeyboardInterrupt:
            print("\nuser stop")
        except BaseException as e:
            print("Exception:", e)
            try:
                sys.print_exception(e)
            except Exception:
                pass
        finally:
            print("cleanup...")
            # 发送视觉无效帧通知下位机
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
        """构建二进制帧。"""
        # ---- 模式语义：某些模式下始终输出无效数据 ----
        valid = self._should_output_valid()

        if valid and geom.vision_state != VisionState.INVALID:
            flags = FLAG_VISION_VALID
            if geom.degraded:
                flags |= FLAG_DEGRADED
            if geom.left_valid:
                flags |= FLAG_LEFT_VALID
            if geom.right_valid:
                flags |= FLAG_RIGHT_VALID
            if struct.left_branch:
                flags |= FLAG_LEFT_BRANCH
            if struct.right_branch:
                flags |= FLAG_RIGHT_BRANCH
            if struct.intersection_candidate:
                flags |= FLAG_INTERSECTION

            # 转换单位
            # lateral_error: 检测像素 → 0.1mm
            #   假设 1px ≈ 1.5mm（取决于摄像头高度，需标定）
            px_to_mm = 1.5  # 待实车标定
            lateral_raw = int(round(geom.lateral_error * px_to_mm * 10))
            heading_raw = int(round(geom.heading_error * 180.0 / 3.14159 * 100))
            width_raw = int(round(geom.road_width * px_to_mm * 10))
        else:
            # 模式要求输出无效 / 视觉状态 INVALID
            flags = 0
            lateral_raw = INVALID_S16
            heading_raw = INVALID_S16
            width_raw = INVALID_U16

        # 钳位
        lateral_raw = max(-32767, min(32767, lateral_raw))
        heading_raw = max(-32767, min(32767, heading_raw))
        width_raw = max(0, min(65534, width_raw))

        confidence = geom.confidence if valid else 0

        return self.frame_builder.build(
            timestamp_ms=timestamp_ms,
            mode=self.mode_fsm.mode,
            flags=flags,
            lateral_error_raw=lateral_raw,
            heading_error_raw=heading_raw,
            road_width_raw=width_raw,
            junction_stage=struct.junction_stage,
            junction_distance=struct.junction_distance,
            confidence=confidence,
            anomaly_flags=anomaly,
        )

    def _build_status_frame(self):
        """构建状态帧（模式切换等不包含道路数据时使用）。"""
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
        """构建完全无效帧。"""
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

    def _draw_overlay(self, img, boundary, geom):
        """在图像上画边界点、中心线和辅助信息。"""
        try:
            # 画边界点
            for (lx, ly) in boundary.left_points:
                img.draw_circle(int(lx), int(ly), 2, color=(0, 0, 255), fill=True)
            for (rx, ry) in boundary.right_points:
                img.draw_circle(int(rx), int(ry), 2, color=(255, 0, 0), fill=True)
            # 画中心点
            for (cx, cy) in boundary.center_points:
                img.draw_circle(int(cx), int(cy), 1, color=(0, 255, 0), fill=True)
            # 画中心线
            if geom.left_valid and geom.right_valid:
                cx_line = int(DETECT_WIDTH / 2)
                cy_near = int(ROI_BOTTOM * 0.9)
                cy_far = int(ROI_TOP + 10)
                img.draw_line(cx_line, cy_near, cx_line, cy_far, color=(0, 200, 0), thickness=1)
            # 画 ROI 框
            img.draw_rectangle(ROI_LEFT, ROI_TOP,
                               ROI_RIGHT - ROI_LEFT, ROI_BOTTOM - ROI_TOP,
                               color=(100, 100, 100), thickness=1)
        except Exception:
            pass


# ============================================================================
# 辅助函数
# ============================================================================

def ticks_ms_diff(now, old):
    """ticks_ms 差值，自动处理回绕。"""
    diff = now - old
    if diff < -0x40000000:
        diff += 0x100000000
    return diff & 0x7FFFFFFF


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    if ON_K230 or True:  # 允许尝试运行
        K230RoadVision().run()
    else:
        print("This program requires K230/CanMV MicroPython hardware.")
        print("Run the PC tests instead: python tests/test_road_geometry.py")
