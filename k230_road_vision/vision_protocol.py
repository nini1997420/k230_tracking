# -*- coding: utf-8 -*-
"""
K230 道路循迹 — 通信协议层

定义 K230 → MSPM0 的道路视觉二进制帧格式（版本化协议）。

警告：本协议为草案版本。MSPM0 接收端源码 (app_aim_protocol.c/.h) 
      不在本仓库中，接收端未确认。在此标记为 "接收端未确认"。

      现有 Plan B 22 字节协议 (rect/laser 坐标) 的字段不包含道路宽度、
      横向偏差、航向偏差和支路信息，因此不兼容，需要此新协议。

帧格式 (ROAD_VISION_FRAME, 24 字节):
┌───────┬──────┬──────────────────────────────────────────┐
│Offset │ Size │ Field                                    │
├───────┼──────┼──────────────────────────────────────────┤
│   0   │   2  │ Header: 0xAA 0x55                        │
│   2   │   1  │ Protocol Version: 0x02                    │
│   3   │   1  │ Message Type: 0x10 (ROAD_VISION)          │
│   4   │   2  │ Sequence number (u16, little-endian)      │
│   6   │   4  │ Timestamp ms (u32, little-endian)         │
│  10   │   1  │ Mode (u8): 0=IDLE 1=TRACK 2=TURNING       │
│      │      │           3=REACQUIRE 4=FAULT 5=NUMBER     │
│  11   │   1  │ Vision Flags (u8):                        │
│      │      │   bit0 = vision_valid                     │
│      │      │   bit1 = degraded                         │
│      │      │   bit2 = left_valid                       │
│      │      │   bit3 = right_valid                      │
│      │      │   bit4 = left_branch                      │
│      │      │   bit5 = right_branch                     │
│      │      │   bit6 = intersection_candidate            │
│      │      │   bit7 = reserved (must be 0)              │
│  12   │   2  │ lateral_error (s16 LE) [0.1mm]            │
│      │      │   >0 = road center to the right            │
│      │      │   0x8000 = INVALID                        │
│  14   │   2  │ heading_error (s16 LE) [0.01 deg]         │
│      │      │   >0 = road bends right                   │
│      │      │   0x8000 = INVALID                        │
│  16   │   2  │ road_width (u16 LE) [0.1mm]               │
│      │      │   0xFFFF = INVALID                        │
│  18   │   1  │ junction_stage (u8)                       │
│      │      │   0=NONE 1=APPROACHING 2=NEAR 3=AT 4=PASSED│
│  19   │   1  │ junction_distance (u8)                    │
│      │      │   0=far 1=mid 2=near 3=at                 │
│  20   │   1  │ confidence (u8) [0-100]                   │
│  21   │   1  │ anomaly_flags (u8)                        │
│      │      │   bit0 = blur                             │
│      │      │   bit1 = overexposed                      │
│      │      │   bit2 = underexposed                     │
│      │      │   bit3 = edge_noise                       │
│      │      │   bit4-7 = reserved                       │
│  22   │   2  │ CRC16-CCITT-FALSE (u16 LE)                │
│      │      │   covers bytes 0..21                      │
└───────┴──────┴──────────────────────────────────────────┘
Total: 24 bytes

CRC 参数:
  - 算法: CRC16-CCITT-FALSE
  - 多项式: 0x1021
  - 初始值: 0xFFFF
  - 不取反最终值
  - 覆盖范围: byte 0 ~ byte 21 (frame_data_size = 22)

MSPM0 接收端验证:
  接收端必须拒绝以下帧:
    - 帧长 != 24
    - 协议版本不匹配
    - 消息类型未知
    - CRC 错误
    - 序号重复或倒序（允许丢帧后的跳变）
    - 非法字段值（如 lateral_error 为 0x8000 时 bit0 应为 0）
  建议计数器: validFrames / crcErrors / lengthErrors / fieldErrors / sequenceErrors
"""


# 协议常量
PROTOCOL_VERSION = 0x02
MSG_TYPE_ROAD_VISION = 0x10
FRAME_SIZE = 24
CRC_DATA_SIZE = 22  # CRC 覆盖字节数

HEADER0 = 0xAA
HEADER1 = 0x55

INVALID_S16 = 0x8000
INVALID_U16 = 0xFFFF

# 模式
MODE_IDLE = 0
MODE_TRACK = 1
MODE_TURNING = 2
MODE_REACQUIRE = 3
MODE_FAULT = 4
MODE_NUMBER = 5

# 标志位
FLAG_VISION_VALID = 0x01
FLAG_DEGRADED = 0x02
FLAG_LEFT_VALID = 0x04
FLAG_RIGHT_VALID = 0x08
FLAG_LEFT_BRANCH = 0x10
FLAG_RIGHT_BRANCH = 0x20
FLAG_INTERSECTION = 0x40

# 非法值定义
VALID_MODES = {MODE_IDLE, MODE_TRACK, MODE_TURNING, MODE_REACQUIRE, MODE_FAULT, MODE_NUMBER}
VALID_STAGES = {0, 1, 2, 3, 4}  # NONE, APPROACHING, NEAR, AT, PASSED
VALID_DISTANCES = {0, 1, 2, 3}  # far, mid, near, at
VALID_FLAGS_MASK = 0x7F  # bit7 reserved, must be 0
VALID_ANOMALY_MASK = 0x0F  # bit4-7 reserved, must be 0

# 异常标志位
ANOMALY_BLUR = 0x01
ANOMALY_OVEREXPOSED = 0x02
ANOMALY_UNDEREXPOSED = 0x04
ANOMALY_EDGE_NOISE = 0x08


# ============================================================================
# CRC16-CCITT-FALSE
# ============================================================================

# 查找表（预计算）
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
    """
    计算 CRC16-CCITT-FALSE。
    data: bytes 或 bytearray 或 list-of-int
    length: 计算覆盖字节数
    """
    crc = 0xFFFF
    for index in range(int(length)):
        crc = ((crc << 8) ^ CRC16_TABLE[((crc >> 8) ^ data[index]) & 0xFF]) & 0xFFFF
    return crc


# ============================================================================
# Little-endian 写入辅助
# ============================================================================

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


# ============================================================================
# 读辅助（用于测试解码器）
# ============================================================================

def _read_le16(buf, offset):
    return (buf[offset] | (buf[offset + 1] << 8)) & 0xFFFF


def _read_le32(buf, offset):
    return (buf[offset] | (buf[offset + 1] << 8) |
            (buf[offset + 2] << 16) | (buf[offset + 3] << 24)) & 0xFFFFFFFF


# ============================================================================
# 帧构建器
# ============================================================================

class RoadFrameBuilder:
    """
    构建 24 字节道路视觉帧。

    使用预分配 bytearray，避免每帧字符串格式化。
    """

    def __init__(self):
        self.sequence = 0
        self.buf = bytearray(FRAME_SIZE)

    def build(self, timestamp_ms, mode, flags, lateral_error_raw, heading_error_raw,
              road_width_raw, junction_stage, junction_distance,
              confidence, anomaly_flags) -> bytes:
        """
        构建一帧完整数据。

        lateral_error_raw: 横向偏差 [0.1mm]，INVALID_S16 表示无效
        heading_error_raw: 航向偏差 [0.01deg]，INVALID_S16 表示无效
        road_width_raw: 道路宽度 [0.1mm]，INVALID_U16 表示无效
        """
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


# ============================================================================
# 帧解码器（用于 PC 测试和 MSPM0 参考实现）
# ============================================================================

class RoadFrameDecoder:
    """
    解码 24 字节道路视觉帧。用于 PC 测试验证编码/解码一致性。

    MSPM0 接收端可参考此逻辑实现 C 版本。
    """

    def __init__(self):
        # 有状态序号跟踪
        self.last_seq = -1             # 上一帧序号，-1 表示未初始化
        self.seq_wrap_count = 0        # 回绕计数

    def decode(self, frame: bytes):
        """
        解码帧，返回 (dict, None) 或 (None, error_reason)。

        增强验证：
          - 字段合法性（mode, stage, distance, confidence, 保留位）
          - vision_valid 与 INVALID 字段一致性
          - 有状态序号检查（倒序、回绕）
        接收端未确认。
        """
        # 长度检查
        if len(frame) != FRAME_SIZE:
            return None, "length_error: expected %d, got %d" % (FRAME_SIZE, len(frame))

        # CRC 验证
        crc_rx = _read_le16(frame, 22)
        crc_calc = crc16_ccitt_false(frame, CRC_DATA_SIZE)
        if crc_rx != crc_calc:
            return None, "crc_error: rx=0x%04X calc=0x%04X" % (crc_rx, crc_calc)

        # 帧头验证
        if frame[0] != HEADER0 or frame[1] != HEADER1:
            return None, "header_error"

        # 协议版本
        if frame[2] != PROTOCOL_VERSION:
            return None, "version_error: 0x%02X" % frame[2]

        # 消息类型
        if frame[3] != MSG_TYPE_ROAD_VISION:
            return None, "msg_type_error: 0x%02X" % frame[3]

        # ---- 字段合法性验证 ----
        mode = frame[10]
        flags = frame[11]
        lateral_raw = _read_le16(frame, 12)
        heading_raw = _read_le16(frame, 14)
        width_raw = _read_le16(frame, 16)
        stage = frame[18]
        distance = frame[19]
        confidence = frame[20]
        anomaly = frame[21]

        # 非法 mode
        if mode not in VALID_MODES:
            return None, "field_error: illegal mode 0x%02X" % mode

        # 非法 stage
        if stage not in VALID_STAGES:
            return None, "field_error: illegal junction_stage %d" % stage

        # 非法 distance
        if distance not in VALID_DISTANCES:
            return None, "field_error: illegal junction_distance %d" % distance

        # 非法 confidence (>100)
        if confidence > 100:
            return None, "field_error: confidence %d > 100" % confidence

        # 保留位非零
        if flags & ~VALID_FLAGS_MASK:
            return None, "field_error: flags reserved bits set (0x%02X)" % flags
        if anomaly & ~VALID_ANOMALY_MASK:
            return None, "field_error: anomaly reserved bits set (0x%02X)" % anomaly

        # ---- vision_valid 与 INVALID 字段一致性 ----
        vision_valid = bool(flags & FLAG_VISION_VALID)
        if vision_valid:
            # 如果宣称有效，则几何字段不能是 INVALID
            if lateral_raw == INVALID_S16:
                return None, "field_error: vision_valid=1 but lateral_error=INVALID"
            if heading_raw == INVALID_S16:
                return None, "field_error: vision_valid=1 but heading_error=INVALID"
            if width_raw == INVALID_U16:
                return None, "field_error: vision_valid=1 but road_width=INVALID"
            if confidence == 0:
                return None, "field_error: vision_valid=1 but confidence=0"

        # ---- 有状态序号检查 ----
        seq = _read_le16(frame, 4)
        seq_ok, seq_err = self._check_sequence(seq)
        if not seq_ok:
            return None, seq_err

        result = {
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
        }
        # 仅当所有校验通过后才更新 last_seq
        self.last_seq = seq
        return result, None

    def _check_sequence(self, seq):
        """有状态序号检查。返回 (ok, error_reason)。"""
        if self.last_seq < 0:
            # 首帧，允许任意序号
            return True, None

        delta = (seq - self.last_seq) & 0xFFFF

        # 回绕：从接近 65535 跳到接近 0
        if self.last_seq > 0xF000 and seq < 0x0FFF:
            self.seq_wrap_count += 1
            return True, None

        # 倒序：序号小于上一帧（非回绕）
        if delta > 0x8000:
            return False, "sequence_error: rewind last=%d current=%d" % (
                self.last_seq, seq)

        # 重复序号
        if delta == 0:
            return False, "sequence_error: duplicate seq=%d" % seq

        # 跳变太大（>100），可能是丢帧或攻击
        if delta > 100 and self.last_seq > 0:
            # 允许大跳变但记录
            pass

        return True, None

    def reset_sequence(self):
        """重置序号状态（用于测试或连接重建）。"""
        self.last_seq = -1
        self.seq_wrap_count = 0

    @staticmethod
    def decode_validate(frame: bytes) -> bool:
        """仅验证帧是否合法（CRC + 帧头 + 长度），不做字段和序号检查。"""
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
        """将 u16 转为 s16。"""
        val = val & 0xFFFF
        if val >= 0x8000:
            return val - 0x10000
        return val
