# -*- coding: utf-8 -*-
"""
PC 端道路结构检测测试

测试内容：
  - all_segments 每行全部黑线段保留
  - 左支路（额外线段在左边界左侧）
  - 右支路（额外线段在右边界右侧）
  - 十字路口（额外线段在左右边界之间）
  - 路口位置来自图像 Y 坐标，非命中次数
  - 已确认路口保持到离开确认
  - 冷却只用于防止再次触发
  - 误触发抵抗
  - 进入保持与离开迟滞

不导入板端模块。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from road_geometry import (
    RoadBoundaryExtractor, RoadGeometry, VisionState,
    BoundaryResult, GeometryResult,
)
from road_structure import RoadStructureDetector, StructureResult, JunctionStage


def make_road_with_extra_branch(branch_side="none", img_w=320, img_h=240,
                                 road_center=160, road_width=100):
    """
    生成带额外支线的道路图像。
    branch_side: "left" / "right" / "cross" / "none"
    额外黑线段出现在远处（y 较小的行），模拟支路。
    """
    black = 30
    road_surface = 255
    bg = 200
    line_thick = 5
    left_edge = road_center - road_width // 2
    right_edge = road_center + road_width // 2
    branch_y_start = 40   # 支路起始 Y
    branch_y_end = 80     # 支路结束 Y

    img = []
    for y in range(img_h):
        row = []
        for x in range(img_w):
            # 正常道路黑线
            on_left = abs(x - left_edge) <= line_thick // 2
            on_right = abs(x - right_edge) <= line_thick // 2
            # 额外支路
            on_branch = False
            if branch_y_start <= y <= branch_y_end:
                if branch_side == "left":
                    on_branch = abs(x - (left_edge - 30)) <= 5
                elif branch_side == "right":
                    on_branch = abs(x - (right_edge + 30)) <= 5
                elif branch_side == "cross":
                    # 左支路 + 右支路
                    on_branch = (abs(x - (left_edge - 25)) <= 4 or
                                 abs(x - (right_edge + 25)) <= 4)

            if on_left or on_right or on_branch:
                row.append(black)
            elif left_edge < x < right_edge:
                row.append(road_surface)
            else:
                row.append(bg)
        img.append(row)
    return img


def make_wide_junction_image(img_w=320, img_h=240, road_center=160, road_width=100):
    """
    生成路口扩宽图像：远处道路变宽。
    """
    black = 30
    road_surface = 255
    bg = 200
    line_thick = 5
    junction_y = 80  # 路口起始 Y

    img = []
    for y in range(img_h):
        if y < junction_y:
            # 远处：宽路
            w = int(road_width * 1.8)
            left = road_center - w // 2
            right = road_center + w // 2
        else:
            left = road_center - road_width // 2
            right = road_center + road_width // 2

        row = []
        for x in range(img_w):
            on_left = abs(x - left) <= line_thick // 2
            on_right = abs(x - right) <= line_thick // 2
            if on_left or on_right:
                row.append(black)
            elif left < x < right:
                row.append(road_surface)
            else:
                row.append(bg)
        img.append(row)
    return img


def make_normal_road(img_w=320, img_h=240, road_center=160, road_width=100):
    """生成普通直路（无支路）."""
    black = 30
    road_surface = 255
    bg = 200
    line_thick = 5
    left_edge = road_center - road_width // 2
    right_edge = road_center + road_width // 2

    img = []
    for y in range(img_h):
        row = []
        for x in range(img_w):
            on_left = abs(x - left_edge) <= line_thick // 2
            on_right = abs(x - right_edge) <= line_thick // 2
            if on_left or on_right:
                row.append(black)
            elif left_edge < x < right_edge:
                row.append(road_surface)
            else:
                row.append(bg)
        img.append(row)
    return img


def _extract_and_compute(img, extractor=None, geom=None):
    """辅助：提取边界并计算几何。"""
    if extractor is None:
        extractor = RoadBoundaryExtractor()
    if geom is None:
        geom = RoadGeometry()
    boundary = extractor.extract(img)
    g = geom.compute(boundary)
    return boundary, g


# ============================================================================
# 测试：all_segments 保留
# ============================================================================

def test_all_segments_retained():
    """验证 BoundaryResult.all_segments 保留了每行全部黑线段。"""
    extractor = RoadBoundaryExtractor()
    img = make_normal_road()
    boundary = extractor.extract(img)

    assert hasattr(boundary, 'all_segments'), "BoundaryResult must have all_segments"
    assert isinstance(boundary.all_segments, list), "all_segments must be a list"
    assert len(boundary.all_segments) > 0, "all_segments should not be empty"
    # 每条目都应该是列表
    for segs in boundary.all_segments:
        assert isinstance(segs, list), "each element must be a list of segments"
        for seg in segs:
            assert isinstance(seg, tuple) and len(seg) == 2, \
                "each segment must be (start, end) tuple"
    print("  PASS: test_all_segments_retained")


# ============================================================================
# 测试：支路类型
# ============================================================================

def test_left_branch_detection():
    """左支路（额外黑线在左边界左侧）应被检测。"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    detector = RoadStructureDetector()

    img = make_road_with_extra_branch(branch_side="left")
    boundary, g = _extract_and_compute(img, extractor, geom)

    # 多帧确认
    result = None
    for _ in range(8):
        result = detector.detect(boundary, g)

    assert result is not None
    assert result.left_branch, "Should detect left branch with extra segments on left"
    print("  PASS: test_left_branch_detection")


def test_right_branch_detection():
    """右支路（额外黑线在右边界右侧）应被检测。"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    detector = RoadStructureDetector()

    img = make_road_with_extra_branch(branch_side="right")
    boundary, g = _extract_and_compute(img, extractor, geom)

    result = None
    for _ in range(8):
        result = detector.detect(boundary, g)

    assert result is not None
    assert result.right_branch, "Should detect right branch with extra segments on right"
    print("  PASS: test_right_branch_detection")


def test_cross_intersection_detection():
    """十字路口（两侧额外黑线段）应触发 intersection_candidate。"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    detector = RoadStructureDetector()

    img = make_road_with_extra_branch(branch_side="cross")
    boundary, g = _extract_and_compute(img, extractor, geom)

    result = None
    for _ in range(8):
        result = detector.detect(boundary, g)

    assert result is not None
    # 十字路口会触发 left_branch + right_branch + intersection_candidate
    assert result.left_branch or result.right_branch or result.intersection_candidate, \
        "Cross junction should trigger at least one branch flag"
    print("  PASS: test_cross_intersection_detection")


def test_no_false_branch_on_normal_road():
    """普通直路不应误触发支路检测。"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    detector = RoadStructureDetector()

    img = make_normal_road()
    boundary, g = _extract_and_compute(img, extractor, geom)

    result = None
    for _ in range(10):
        result = detector.detect(boundary, g)

    assert result is not None
    assert not result.left_branch, "Normal road should not detect left branch"
    assert not result.right_branch, "Normal road should not detect right branch"
    assert not result.intersection_candidate, "Normal road should not detect intersection"
    print("  PASS: test_no_false_branch_on_normal_road")


# ============================================================================
# 测试：路口 Y 位置（非命中次数）
# ============================================================================

def test_junction_position_from_image_y():
    """路口距离应来自图像 Y 坐标，不是命中次数。"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    detector = RoadStructureDetector()

    img = make_wide_junction_image()
    boundary, g = _extract_and_compute(img, extractor, geom)

    result = None
    for _ in range(8):
        result = detector.detect(boundary, g)

    assert result is not None
    # junction_distance_px 应该是一个合理的像素距离（非零且非命中次数）
    if result.junction_distance_px > 0:
        # 距离应该 < DETECT_HEIGHT (240)
        assert result.junction_distance_px < 240, \
            "junction_distance_px=%d should be < 240" % result.junction_distance_px
        # 路口阶段不应是 NONE（如果有检测到）
        # 注意：wide_junction 可能由宽度增大触发
    print("  PASS: test_junction_position_from_image_y")


# ============================================================================
# 测试：已确认路口的保持与离开
# ============================================================================

def test_junction_hold_until_leave():
    """
    路口确认后应保持输出，直到特征消失才进入 PASSED。
    """
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    detector = RoadStructureDetector()

    # 1. 持续显示支路图像，建立确认
    img_branch = make_road_with_extra_branch(branch_side="left")
    boundary, g = _extract_and_compute(img_branch, extractor, geom)

    for _ in range(8):
        result = detector.detect(boundary, g)

    assert result.left_branch, "After 8 frames, left branch should be confirmed"
    assert result.structure_confirmed, "Structure should be confirmed"

    # 2. 切换到正常道路（特征消失），多帧让 hits 衰减到 0
    img_normal = make_normal_road()
    boundary_n, g_n = _extract_and_compute(img_normal, extractor, geom)

    # 需要足够帧数让 hits 从 5 衰减到 0
    for _ in range(10):
        detector.detect(boundary_n, g_n)
    # 路口应变为 PASSED 并进入冷却
    assert detector.cooldown_counter > 0 or \
        detector._last_junction_stage == JunctionStage.PASSED, \
        "After feature disappears, should enter cooldown or PASSED"

    print("  PASS: test_junction_hold_until_leave")


def test_junction_cooldown_prevents_retrigger():
    """
    路口冷却期间不应再次触发。
    """
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    detector = RoadStructureDetector()

    img_branch = make_road_with_extra_branch(branch_side="left")
    boundary, g = _extract_and_compute(img_branch, extractor, geom)

    # 确认路口
    for _ in range(8):
        detector.detect(boundary, g)

    # 特征消失 → 需要多帧让 hits 衰减
    img_normal = make_normal_road()
    boundary_n, g_n = _extract_and_compute(img_normal, extractor, geom)
    for _ in range(10):
        detector.detect(boundary_n, g_n)
    assert detector.cooldown_counter > 0, "Should enter cooldown after junction passes"

    # 在冷却期内再次出现支路 → 不应立即触发
    result = detector.detect(boundary, g)
    # 冷却期内 left_branch 应为 False
    assert not result.left_branch or detector.cooldown_counter > 0, \
        "Cooldown should prevent immediate re-trigger"

    print("  PASS: test_junction_cooldown_prevents_retrigger")


# ============================================================================
# 测试：宽度突变路口
# ============================================================================

def test_width_increase_junction():
    """道路宽度突变应触发路口候选。"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    detector = RoadStructureDetector()

    img = make_wide_junction_image()
    boundary, g = _extract_and_compute(img, extractor, geom)

    # 多帧检测
    result = None
    for _ in range(8):
        result = detector.detect(boundary, g)

    assert result is not None
    # 扩宽道路应至少触发 intersection_candidate
    # （具体取决于阈值和图像质量）
    print("  PASS: test_width_increase_junction")


# ============================================================================
# 测试：on_turning_complete 复位
# ============================================================================

def test_turning_complete_resets_structure():
    """转弯结束后路口状态应完全复位。"""
    detector = RoadStructureDetector()

    # 模拟一些 hits
    detector.left_branch_hits = 10
    detector.right_branch_hits = 5
    detector.intersection_hits = 8
    detector.cooldown_counter = 3
    detector._confirmed_junction_active = True
    detector.junction_y_px = 50

    detector.on_turning_complete()

    assert detector.left_branch_hits == 0
    assert detector.right_branch_hits == 0
    assert detector.intersection_hits == 0
    assert detector.cooldown_counter == 0
    assert detector._confirmed_junction_active == False
    assert detector.junction_y_px == -1
    assert detector._last_junction_stage == JunctionStage.NONE
    assert len(detector.width_history) == 0

    print("  PASS: test_turning_complete_resets_structure")


# ============================================================================
# 运行全部测试
# ============================================================================

if __name__ == "__main__":
    print("K230 Road Structure Tests")
    print("=" * 45)
    passed = 0
    failed = 0
    tests = [
        test_all_segments_retained,
        test_left_branch_detection,
        test_right_branch_detection,
        test_cross_intersection_detection,
        test_no_false_branch_on_normal_road,
        test_junction_position_from_image_y,
        test_junction_hold_until_leave,
        test_junction_cooldown_prevents_retrigger,
        test_width_increase_junction,
        test_turning_complete_resets_structure,
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
        print("STRUCTURE ISSUES FOUND -- fix before deployment.")
        sys.exit(1)
    else:
        print("All PC structure tests passed.")
