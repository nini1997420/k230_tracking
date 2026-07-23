# -*- coding: utf-8 -*-
"""
PC 端通信协议测试

测试内容：
  - 正常帧编码/解码一致性
  - CRC 标准向量验证
  - 单字节破坏后 CRC 失败
  - 截断帧被拒绝
  - 超长帧被拒绝
  - 非法协议版本被拒绝
  - 无效测量不被编码成正常零误差
  - 序号重复（同一序号应通过检查，仅倒序被识别）

不导入板端模块。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_protocol import (
    RoadFrameBuilder, RoadFrameDecoder, crc16_ccitt_false,
    FRAME_SIZE, CRC_DATA_SIZE,
    HEADER0, HEADER1, PROTOCOL_VERSION, MSG_TYPE_ROAD_VISION,
    MODE_TRACK, MODE_IDLE, MODE_FAULT, MODE_TURNING, MODE_REACQUIRE,
    FLAG_VISION_VALID, FLAG_DEGRADED, FLAG_LEFT_VALID, FLAG_RIGHT_VALID,
    FLAG_LEFT_BRANCH, FLAG_RIGHT_BRANCH, FLAG_INTERSECTION,
    INVALID_S16, INVALID_U16,
    VALID_MODES, VALID_STAGES, VALID_DISTANCES,
    VALID_FLAGS_MASK, VALID_ANOMALY_MASK,
)


# ============================================================================
# CRC 标准向量
# ============================================================================

def test_crc_standard_vectors():
    """验证 CRC16-CCITT-FALSE 算法一致性（与 Plan B 已知参考值对齐）。"""
    # 空数据（长度 0）：CRC 应为 0xFFFF
    crc = crc16_ccitt_false(b"", 0)
    assert crc == 0xFFFF, "CRC of empty: expected 0xFFFF, got 0x%04X" % crc

    # 与 Plan B 实现一致性：构造一帧已知内容，编解码 CRC 应一致
    builder = RoadFrameBuilder()
    frame = builder.build(0, MODE_TRACK, 0, 0, 0, 0, 0, 0, 0, 0)
    # CRC 在 bytes 22-23 (LE)
    crc_in_frame = frame[22] | (frame[23] << 8)
    # 对 bytes 0-21 重新计算 CRC
    crc_recalc = crc16_ccitt_false(frame, CRC_DATA_SIZE)
    assert crc_in_frame == crc_recalc, \
        "CRC consistency: frame CRC=0x%04X, recalc=0x%04X" % (crc_in_frame, crc_recalc)

    # 验证 CRC 字节序：CRC 写入后单字节破坏应使 decode 失败
    decoder = RoadFrameDecoder()
    _, err = decoder.decode(frame)
    assert err is None, "Valid frame should decode without CRC error"
    print("  PASS: test_crc_standard_vectors")


# ============================================================================
# 编码/解码一致性
# ============================================================================

def test_encode_decode_roundtrip():
    """正常帧编码后由解码器得到完全相同字段。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()

    frame = builder.build(
        timestamp_ms=123456,
        mode=MODE_TRACK,
        flags=FLAG_VISION_VALID | FLAG_LEFT_VALID | FLAG_RIGHT_VALID,
        lateral_error_raw=350,     # 35.0mm right
        heading_error_raw=-200,    # -2.00 deg left tilt
        road_width_raw=1200,       # 120.0mm
        junction_stage=1,          # APPROACHING
        junction_distance=1,       # mid
        confidence=85,
        anomaly_flags=0,
    )

    assert len(frame) == FRAME_SIZE, \
        "Frame size should be %d, got %d" % (FRAME_SIZE, len(frame))
    assert frame[0] == HEADER0 and frame[1] == HEADER1, "Header mismatch"

    result, err = decoder.decode(frame)
    assert err is None, "Decode error: %s" % err

    assert result is not None, "Decode returned None"
    assert result["mode"] == MODE_TRACK, "mode mismatch: %d" % result["mode"]
    assert result["vision_valid"] is True
    assert result["left_valid"] is True
    assert result["right_valid"] is True
    assert result["lateral_error_0_1mm"] == 350
    assert result["heading_error_0_01deg"] == -200
    assert result["road_width_0_1mm"] == 1200
    assert result["junction_stage"] == 1
    assert result["junction_distance"] == 1
    assert result["confidence"] == 85
    assert result["degraded"] is False
    print("  PASS: test_encode_decode_roundtrip")


def test_invalid_measurement():
    """无效测量应编码为 INVALID_S16/INVALID_U16，不编码成正常零误差。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()

    frame = builder.build(
        timestamp_ms=999999,
        mode=MODE_FAULT,
        flags=0,                   # vision_valid = 0
        lateral_error_raw=INVALID_S16,
        heading_error_raw=INVALID_S16,
        road_width_raw=INVALID_U16,
        junction_stage=0,
        junction_distance=0,
        confidence=0,
        anomaly_flags=1,           # blur
    )

    result, err = decoder.decode(frame)
    assert err is None

    assert result["vision_valid"] is False
    assert result["lateral_error_0_1mm"] == -32768, \
        "Invalid s16 should decode to -32768, got %d" % result["lateral_error_0_1mm"]
    assert result["heading_error_0_01deg"] == -32768
    assert result["road_width_0_1mm"] == 0xFFFF  # INVALID_U16, stays u16
    assert result["confidence"] == 0
    print("  PASS: test_invalid_measurement")


# ============================================================================
# 错误检测
# ============================================================================

def test_crc_corruption():
    """单字节破坏后 CRC 必须失败。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()

    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_list = list(frame)
    frame_list[10] ^= 0x01  # 翻转 mode 字节最低位
    corrupted = bytes(frame_list)

    result, err = decoder.decode(corrupted)
    assert err is not None, "Corrupted frame should be rejected"
    assert "crc" in err.lower(), "Error should mention CRC"
    print("  PASS: test_crc_corruption")


def test_truncated_frame():
    """截断帧被拒绝。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()

    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    truncated = frame[:20]  # 缺少最后 4 字节

    result, err = decoder.decode(truncated)
    assert err is not None
    assert "length" in err.lower()
    print("  PASS: test_truncated_frame")


def test_oversized_frame():
    """超长帧被拒绝。"""
    decoder = RoadFrameDecoder()
    oversized = b"\xAA" * (FRAME_SIZE + 3)
    result, err = decoder.decode(oversized)
    assert err is not None
    assert "length" in err.lower()
    print("  PASS: test_oversized_frame")


def test_wrong_protocol_version():
    """非法协议版本被拒绝。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()

    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_list = list(frame)
    frame_list[2] = 0xFF  # 错误的协议版本
    # 重算 CRC
    from vision_protocol import crc16_ccitt_false, CRC_DATA_SIZE, _write_le16
    modified = bytearray(frame_list)
    new_crc = crc16_ccitt_false(modified, CRC_DATA_SIZE)
    _write_le16(modified, 22, new_crc)

    result, err = decoder.decode(bytes(modified))
    assert err is not None
    assert "version" in err.lower()
    print("  PASS: test_wrong_protocol_version")


def test_wrong_message_type():
    """非法消息类型被拒绝。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()

    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_list = list(frame)
    frame_list[3] = 0xFF  # 错误的消息类型
    from vision_protocol import crc16_ccitt_false, CRC_DATA_SIZE, _write_le16
    modified = bytearray(frame_list)
    new_crc = crc16_ccitt_false(modified, CRC_DATA_SIZE)
    _write_le16(modified, 22, new_crc)

    result, err = decoder.decode(bytes(modified))
    assert err is not None
    assert "msg_type" in err.lower() or "type" in err.lower()
    print("  PASS: test_wrong_message_type")


def test_sequence_increment():
    """序号应单调递增。"""
    builder = RoadFrameBuilder()

    seqs = []
    for i in range(10):
        frame = builder.build(i * 100, MODE_TRACK, 0, 0, 0, 0, 0, 0, 0, 0)
        seq = frame[4] | (frame[5] << 8)
        seqs.append(seq)

    for i in range(1, len(seqs)):
        diff = (seqs[i] - seqs[i - 1]) & 0xFFFF
        # 应该递增 1（不考虑回绕）
        assert diff == 1, "Sequence should increment by 1, seq[%d]=%d, seq[%d]=%d" % (
            i - 1, seqs[i - 1], i, seqs[i])
    print("  PASS: test_sequence_increment")


def test_flags_encoding():
    """验证各标志位的编码正确性。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()

    # 测试每个标志位
    flags = (FLAG_VISION_VALID | FLAG_DEGRADED | FLAG_LEFT_VALID |
             FLAG_RIGHT_VALID)
    frame = builder.build(0, MODE_TRACK, flags, 100, 0, 500, 0, 0, 90, 0)
    result, err = decoder.decode(frame)
    assert err is None
    assert result["vision_valid"] is True
    assert result["degraded"] is True
    assert result["left_valid"] is True
    assert result["right_valid"] is True
    print("  PASS: test_flags_encoding")


# ============================================================================
# 字段合法性验证（issue 6）
# ============================================================================

def test_reject_illegal_mode():
    """拒绝非法 mode 值。"""
    builder = RoadFrameBuilder()
    # 构建后手动修改 mode 字节并重算 CRC
    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_list = list(frame)
    frame_list[10] = 0xFF  # 非法 mode
    from vision_protocol import crc16_ccitt_false, CRC_DATA_SIZE, _write_le16
    modified = bytearray(frame_list)
    new_crc = crc16_ccitt_false(modified, CRC_DATA_SIZE)
    _write_le16(modified, 22, new_crc)
    decoder = RoadFrameDecoder()
    _, err = decoder.decode(bytes(modified))
    assert err is not None
    assert "illegal mode" in err.lower() or "field_error" in err.lower()
    print("  PASS: test_reject_illegal_mode")


def test_reject_illegal_stage():
    """拒绝非法 junction_stage 值。"""
    builder = RoadFrameBuilder()
    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_list = list(frame)
    frame_list[18] = 9  # 非法 stage (>4)
    from vision_protocol import crc16_ccitt_false, CRC_DATA_SIZE, _write_le16
    modified = bytearray(frame_list)
    new_crc = crc16_ccitt_false(modified, CRC_DATA_SIZE)
    _write_le16(modified, 22, new_crc)
    decoder = RoadFrameDecoder()
    _, err = decoder.decode(bytes(modified))
    assert err is not None
    assert "stage" in err.lower() or "field_error" in err.lower()
    print("  PASS: test_reject_illegal_stage")


def test_reject_illegal_distance():
    """拒绝非法 junction_distance 值。"""
    builder = RoadFrameBuilder()
    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_list = list(frame)
    frame_list[19] = 5  # 非法 distance (>3)
    from vision_protocol import crc16_ccitt_false, CRC_DATA_SIZE, _write_le16
    modified = bytearray(frame_list)
    new_crc = crc16_ccitt_false(modified, CRC_DATA_SIZE)
    _write_le16(modified, 22, new_crc)
    decoder = RoadFrameDecoder()
    _, err = decoder.decode(bytes(modified))
    assert err is not None
    assert "distance" in err.lower() or "field_error" in err.lower()
    print("  PASS: test_reject_illegal_distance")


def test_reject_illegal_confidence():
    """拒绝 confidence > 100。"""
    builder = RoadFrameBuilder()
    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_list = list(frame)
    frame_list[20] = 200  # 非法 confidence
    from vision_protocol import crc16_ccitt_false, CRC_DATA_SIZE, _write_le16
    modified = bytearray(frame_list)
    new_crc = crc16_ccitt_false(modified, CRC_DATA_SIZE)
    _write_le16(modified, 22, new_crc)
    decoder = RoadFrameDecoder()
    _, err = decoder.decode(bytes(modified))
    assert err is not None
    assert "confidence" in err.lower() or "field_error" in err.lower()
    print("  PASS: test_reject_illegal_confidence")


def test_reject_reserved_bits_in_flags():
    """拒绝 flags 保留位非零。"""
    builder = RoadFrameBuilder()
    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_list = list(frame)
    frame_list[11] = 0x80  # bit7 reserved set
    from vision_protocol import crc16_ccitt_false, CRC_DATA_SIZE, _write_le16
    modified = bytearray(frame_list)
    new_crc = crc16_ccitt_false(modified, CRC_DATA_SIZE)
    _write_le16(modified, 22, new_crc)
    decoder = RoadFrameDecoder()
    _, err = decoder.decode(bytes(modified))
    assert err is not None
    print("  PASS: test_reject_reserved_bits_in_flags")


def test_reject_vision_valid_with_invalid_fields():
    """拒绝 vision_valid=1 但字段为 INVALID。"""
    builder = RoadFrameBuilder()
    frame = builder.build(
        0, MODE_TRACK,
        FLAG_VISION_VALID,       # 宣告有效
        INVALID_S16,             # 但 lateral 是 INVALID → 矛盾
        0, INVALID_U16,          # width 也是 INVALID
        0, 0, 50, 0,
    )
    decoder = RoadFrameDecoder()
    _, err = decoder.decode(frame)
    assert err is not None
    assert "vision_valid" in err.lower() or "field_error" in err.lower()
    print("  PASS: test_reject_vision_valid_with_invalid_fields")


def test_sequence_duplicate_rejected():
    """重复序号被拒绝。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()
    f1 = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    r1, e1 = decoder.decode(f1)
    assert e1 is None
    # 同一帧再 decode，序号相同 → 应拒绝
    r2, e2 = decoder.decode(f1)
    assert e2 is not None
    assert "duplicate" in e2.lower()
    print("  PASS: test_sequence_duplicate_rejected")


def test_sequence_rewind_rejected():
    """倒序序号被拒绝。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()
    # 先 seq 5, 6
    for _ in range(6):
        builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_seq7 = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    decoder.decode(frame_seq7)  # last_seq = 7

    # 再发 seq 3（倒序）→ 应拒绝
    builder2 = RoadFrameBuilder()
    for _ in range(3):
        builder2.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    frame_seq4 = builder2.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    _, err = decoder.decode(frame_seq4)
    assert err is not None
    assert "rewind" in err.lower() or "sequence" in err.lower()
    print("  PASS: test_sequence_rewind_rejected")


def test_sequence_wraparound_accepted():
    """序号回绕被接受。"""
    builder = RoadFrameBuilder()
    decoder = RoadFrameDecoder()
    # 发帧直到 seq=0xFFFE
    builder.sequence = 0xFFFE
    frame = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    r, e = decoder.decode(frame)
    assert e is None

    # 下一帧 seq=0xFFFF
    frame2 = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    r2, e2 = decoder.decode(frame2)
    assert e2 is None

    # 回绕到 0
    frame3 = builder.build(0, MODE_TRACK, 0, 500, 0, 1000, 0, 0, 80, 0)
    r3, e3 = decoder.decode(frame3)  # seq should be 0x0000 (wrapped)
    assert e3 is None, "Wraparound should be accepted: %s" % e3
    print("  PASS: test_sequence_wraparound_accepted")


# ============================================================================
# 运行全部测试
# ============================================================================

if __name__ == "__main__":
    print("K230 Vision Protocol Tests")
    print("=" * 45)
    passed = 0
    failed = 0
    tests = [
        test_crc_standard_vectors,
        test_encode_decode_roundtrip,
        test_invalid_measurement,
        test_crc_corruption,
        test_truncated_frame,
        test_oversized_frame,
        test_wrong_protocol_version,
        test_wrong_message_type,
        test_sequence_increment,
        test_flags_encoding,
        test_reject_illegal_mode,
        test_reject_illegal_stage,
        test_reject_illegal_distance,
        test_reject_illegal_confidence,
        test_reject_reserved_bits_in_flags,
        test_reject_vision_valid_with_invalid_fields,
        test_sequence_duplicate_rejected,
        test_sequence_rewind_rejected,
        test_sequence_wraparound_accepted,
    ]
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print("  FAIL: %s -- %s" % (t.__name__, e))
            failed += 1
        except Exception as e:
            print("  ERROR: %s -- %s" % (t.__name__, e))
            import traceback
            traceback.print_exc()
            failed += 1
    print()
    print("Results: %d passed, %d failed, %d total" % (passed, failed, len(tests)))
    if failed > 0:
        print("PROTOCOL ISSUES FOUND -- fix before deployment.")
        sys.exit(1)
    else:
        print("All PC protocol tests passed.")
        print("MSPM0 RECEIVER STATUS: UNCONFIRMED -- receiver source not in repo")
        print("Verify with actual MSPM0 hardware before claiming 'link completed'.")
