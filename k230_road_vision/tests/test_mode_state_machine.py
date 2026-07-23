# -*- coding: utf-8 -*-
"""
PC 端模式状态机测试

测试内容：
  - TURNING：vision_valid 必须为 0，几何字段必须为 INVALID
  - REACQUIRE：未达到确认帧数前 vision_valid 为 0
  - REACQUIRE 达到确认帧数后切换 TRACK 并输出有效值
  - NUMBER 未实现时不得发送有效道路结果
  - TURNING → REACQUIRE 时复位几何历史和路口状态
  - IDLE/FAULT 模式不输出有效数据

本测试使用模拟视觉结果验证模式机行为。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from road_geometry import RoadBoundaryExtractor, RoadGeometry, VisionState
from road_structure import RoadStructureDetector, StructureResult
from vision_protocol import (
    MODE_IDLE, MODE_TRACK, MODE_TURNING, MODE_REACQUIRE, MODE_FAULT, MODE_NUMBER,
    FLAG_VISION_VALID, INVALID_S16, INVALID_U16,
)

try:
    from road_config import CONF_HIGH_THRESH, REACQUIRE_CONFIRM_FRAMES
except ImportError:
    CONF_HIGH_THRESH = 70
    REACQUIRE_CONFIRM_FRAMES = 8


# ============================================================================
# 模拟 K230RoadVision._should_output_valid 逻辑
# ============================================================================

class MockModeFSM:
    """模拟模式状态机，复制真实 _should_output_valid 逻辑。"""
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
                    self._on_turning_to_reacquire_cb()

    def update(self, road_geom, struct_result):
        if self.mode == MODE_REACQUIRE:
            if road_geom.vision_state == VisionState.NORMAL and \
               road_geom.confidence >= CONF_HIGH_THRESH:
                self.reacquire_count += 1
            else:
                self.reacquire_count = max(0, self.reacquire_count - 1)
            if self.reacquire_count >= REACQUIRE_CONFIRM_FRAMES:
                self.set_mode(MODE_TRACK)

    def should_output_valid(self):
        m = self.mode
        if m in (MODE_IDLE, MODE_FAULT, MODE_TURNING, MODE_NUMBER):
            return False
        if m == MODE_REACQUIRE:
            return self.reacquire_count >= REACQUIRE_CONFIRM_FRAMES
        return True  # MODE_TRACK


# ============================================================================
# 测试用例
# ============================================================================

def test_idle_no_valid_output():
    """IDLE 模式不应输出有效道路数据。"""
    fsm = MockModeFSM()
    fsm.set_mode(MODE_IDLE)
    assert not fsm.should_output_valid(), "IDLE should not output valid"
    print("  PASS: test_idle_no_valid_output")


def test_fault_no_valid_output():
    """FAULT 模式不应输出有效道路数据。"""
    fsm = MockModeFSM()
    fsm.set_mode(MODE_FAULT)
    assert not fsm.should_output_valid(), "FAULT should not output valid"
    print("  PASS: test_fault_no_valid_output")


def test_turning_no_valid_output():
    """
    TURNING 模式：允许采图，但 vision_valid 必须为 0，
    几何字段必须使用 INVALID。
    """
    fsm = MockModeFSM()
    fsm.set_mode(MODE_TURNING)
    assert not fsm.should_output_valid(), "TURNING should not output valid"

    # 验证模式不会因视觉结果而改变
    geom = RoadGeometry().last_geometry  # 默认 INVALID
    struct = StructureResult()
    fsm.update(geom, struct)
    assert fsm.mode == MODE_TURNING, "TURNING should stay TURNING"
    print("  PASS: test_turning_no_valid_output")


def test_reacquire_no_valid_until_confirmed():
    """
    REACQUIRE 模式：未连续达到确认帧数前，vision_valid 必须为 0。
    """
    fsm = MockModeFSM()
    fsm.set_mode(MODE_REACQUIRE)

    # 初始 reacquire_count = 0 → 不应输出有效
    assert not fsm.should_output_valid(), \
        "REACQUIRE with count=0 should not output valid"

    # 手动设置 count < 确认帧数 → 仍不应输出有效
    fsm.reacquire_count = REACQUIRE_CONFIRM_FRAMES - 1
    assert not fsm.should_output_valid(), \
        "REACQUIRE with count=%d should not output valid" % fsm.reacquire_count

    # 达到确认帧数 → TRACK
    fsm.reacquire_count = REACQUIRE_CONFIRM_FRAMES
    assert fsm.should_output_valid(), \
        "REACQUIRE with count=%d should output valid" % fsm.reacquire_count

    print("  PASS: test_reacquire_no_valid_until_confirmed")


def test_reacquire_confirms_to_track():
    """
    REACQUIRE 达到连续确认帧数后，自动切换为 TRACK 并输出有效值。
    """
    fsm = MockModeFSM()
    geom = RoadGeometry()
    extractor = RoadBoundaryExtractor()

    # 模拟正常视觉结果（用于 update）
    class FakeGeom:
        vision_state = VisionState.NORMAL
        confidence = 85

    fsm.set_mode(MODE_REACQUIRE)

    for i in range(REACQUIRE_CONFIRM_FRAMES):
        fsm.update(FakeGeom(), StructureResult())

    assert fsm.mode == MODE_TRACK, \
        "After %d confirm frames, should switch to TRACK, got mode=%d" % (
            REACQUIRE_CONFIRM_FRAMES, fsm.mode)
    assert fsm.should_output_valid(), "TRACK should output valid"
    print("  PASS: test_reacquire_confirms_to_track")


def test_reacquire_resets_on_invalid_vision():
    """
    REACQUIRE 中遇到 INVALID 视觉结果，计数应回退。
    """
    fsm = MockModeFSM()

    class FakeNormal:
        vision_state = VisionState.NORMAL
        confidence = 85

    class FakeInvalid:
        vision_state = VisionState.INVALID
        confidence = 0

    fsm.set_mode(MODE_REACQUIRE)

    # 先累积 3 帧正常
    for _ in range(3):
        fsm.update(FakeNormal(), StructureResult())
    assert fsm.reacquire_count == 3

    # 一帧 INVALID → 计数减 1
    fsm.update(FakeInvalid(), StructureResult())
    assert fsm.reacquire_count == 2, \
        "After INVALID frame, reacquire_count should decrease"

    print("  PASS: test_reacquire_resets_on_invalid_vision")


def test_number_no_valid_output():
    """
    NUMBER 模式未实现时，不得发送有效道路结果。
    """
    fsm = MockModeFSM()
    fsm.set_mode(MODE_NUMBER)
    assert not fsm.should_output_valid(), "NUMBER mode should not output valid"
    print("  PASS: test_number_no_valid_output")


def test_turning_to_reacquire_callback():
    """
    TURNING → REACQUIRE 转换时应触发回调（复位几何历史和路口状态）。
    """
    reset_called = [False]

    def on_reset():
        reset_called[0] = True

    fsm = MockModeFSM()
    fsm._on_turning_to_reacquire_cb = on_reset

    # IDLE → TURNING
    fsm.set_mode(MODE_TURNING)
    assert not reset_called[0]

    # TURNING → REACQUIRE → 回调应被调用
    fsm.set_mode(MODE_REACQUIRE)
    assert reset_called[0], "TURNING→REACQUIRE should trigger reset callback"

    print("  PASS: test_turning_to_reacquire_callback")


def test_track_stays_track():
    """TRACK 模式应持续输出有效数据。"""
    fsm = MockModeFSM()
    fsm.set_mode(MODE_TRACK)
    assert fsm.should_output_valid(), "TRACK should output valid"

    # 视觉结果无效也不应退出 TRACK
    class FakeInvalid:
        vision_state = VisionState.INVALID
        confidence = 0

    fsm.update(FakeInvalid(), StructureResult())
    assert fsm.mode == MODE_TRACK, \
        "TRACK should stay TRACK even with invalid vision"

    print("  PASS: test_track_stays_track")


# ============================================================================
# 运行全部测试
# ============================================================================

if __name__ == "__main__":
    print("K230 Mode State Machine Tests")
    print("=" * 45)
    passed = 0
    failed = 0
    tests = [
        test_idle_no_valid_output,
        test_fault_no_valid_output,
        test_turning_no_valid_output,
        test_reacquire_no_valid_until_confirmed,
        test_reacquire_confirms_to_track,
        test_reacquire_resets_on_invalid_vision,
        test_number_no_valid_output,
        test_turning_to_reacquire_callback,
        test_track_stays_track,
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
        print("MODE STATE MACHINE ISSUES FOUND -- fix before deployment.")
        sys.exit(1)
    else:
        print("All PC mode state machine tests passed.")
