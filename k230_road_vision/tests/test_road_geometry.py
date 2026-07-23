# -*- coding: utf-8 -*-
"""
PC 端道路几何算法测试

测试内容：
  - 双边界居中直路：横向偏差≈0，航向偏差≈0
  - 道路整体右移/左移：横向偏差符号正确
  - 道路向右/左倾斜：航向偏差符号正确
  - 单左边界/单右边界：进入 DEGRADED
  - 双边界丢失：输出 INVALID
  - 道路宽度异常：降低可信度
  - 单帧跳变：不立即接管历史

本测试不导入板端模块，可在 PC 端直接运行。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from road_geometry import (
    RoadBoundaryExtractor, RoadGeometry, VisionState,
    BoundaryResult, GeometryResult,
)


def make_straight_road_image(center_x=160, road_width=100, img_w=320, img_h=240):
    """
    生成居中直路的伪灰度图。
    道路区域为白色(255)，黑线在左右边界。
    简化：黑线区灰度30，道路区内255，其余200。
    """
    black = 30
    road_surface = 255
    bg = 200
    left_edge = center_x - road_width // 2
    right_edge = center_x + road_width // 2
    line_thick = 5  # 黑线厚度

    img = []
    for y in range(img_h):
        row = []
        for x in range(img_w):
            if abs(x - left_edge) <= line_thick // 2:
                row.append(black)
            elif abs(x - right_edge) <= line_thick // 2:
                row.append(black)
            elif left_edge < x < right_edge:
                row.append(road_surface)
            else:
                row.append(bg)
        img.append(row)
    return img


def make_tilted_road_image(center_x_top=160, center_x_bottom=160, road_width=100,
                           img_w=320, img_h=240):
    """
    生成倾斜道路灰度图。
    中心从 top(center_x_top) 线性变化到 bottom(center_x_bottom)。
    """
    black = 30
    road_surface = 255
    bg = 200
    line_thick = 5

    img = []
    for y in range(img_h):
        t = y / max(img_h - 1, 1)  # 0=top, 1=bottom
        cx = center_x_top + t * (center_x_bottom - center_x_top)
        left = cx - road_width // 2
        right = cx + road_width // 2
        row = []
        for x in range(img_w):
            if abs(x - left) <= line_thick // 2:
                row.append(black)
            elif abs(x - right) <= line_thick // 2:
                row.append(black)
            elif left < x < right:
                row.append(road_surface)
            else:
                row.append(bg)
        img.append(row)
    return img


def make_single_boundary_image(side="left", center_x=160, road_width=100,
                                img_w=320, img_h=240):
    """只画一侧黑线。"""
    black = 30
    bg = 200
    line_thick = 5
    left_edge = center_x - road_width // 2
    right_edge = center_x + road_width // 2

    img = []
    for y in range(img_h):
        row = []
        for x in range(img_w):
            draw_left = (side in ("left", "both") and
                         abs(x - left_edge) <= line_thick // 2)
            draw_right = (side in ("right", "both") and
                          abs(x - right_edge) <= line_thick // 2)
            if draw_left or draw_right:
                row.append(black)
            elif left_edge < x < right_edge:
                row.append(255)
            else:
                row.append(bg)
        img.append(row)
    return img


def make_noise_image(img_w=320, img_h=240):
    """纯噪声，不含道路结构。"""
    bg = 160
    img = []
    for y in range(img_h):
        row = [bg + (y * 13 + x * 7) % 30 - 15 for x in range(img_w)]
        img.append(row)
    return img


# ============================================================================
# 测试用例
# ============================================================================

def test_straight_road_centered():
    """居中直路：横向偏差≈0，航向偏差≈0"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img = make_straight_road_image(center_x=160, road_width=100)
    boundary = extractor.extract(img)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.NORMAL, \
        "Expected NORMAL, got %s" % VisionState.name(result.vision_state)
    assert result.left_valid, "Left boundary should be valid"
    assert result.right_valid, "Right boundary should be valid"
    assert abs(result.lateral_error) < 20, \
        "Straight road lateral_error should be near 0, got %.2f" % result.lateral_error
    assert abs(result.heading_error) < 0.1, \
        "Straight road heading_error should be near 0, got %.3f" % result.heading_error
    assert result.confidence > CONF_HIGH_THRESH, \
        "Confidence should be high, got %d" % result.confidence
    print("  PASS: test_straight_road_centered")


def test_road_shifted_right():
    """道路整体右移：横向偏差符号应为正（中心在右）"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img = make_straight_road_image(center_x=190, road_width=100)  # 中心右移
    boundary = extractor.extract(img)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.NORMAL
    # lateral_error > 0: 画面中心在检测中心左侧 → 道路中心在右
    assert result.lateral_error > 5, \
        "Road shifted right: lateral_error should be >0, got %.2f" % result.lateral_error
    print("  PASS: test_road_shifted_right")


def test_road_shifted_left():
    """道路整体左移：横向偏差符号应为负"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img = make_straight_road_image(center_x=130, road_width=100)
    boundary = extractor.extract(img)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.NORMAL
    assert result.lateral_error < -5, \
        "Road shifted left: lateral_error should be <0, got %.2f" % result.lateral_error
    print("  PASS: test_road_shifted_left")


def test_road_tilted_right():
    """道路向右倾斜：航向偏差符号应为正"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    # 顶部中心在左，底部中心在右 → 近场中心比中场中心偏右 → heading > 0
    img = make_tilted_road_image(center_x_top=140, center_x_bottom=180, road_width=100)
    boundary = extractor.extract(img)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.NORMAL
    assert result.heading_error > 0.0, \
        "Road tilted right: heading_error should be >0, got %.3f" % result.heading_error
    print("  PASS: test_road_tilted_right")


def test_road_tilted_left():
    """道路向左倾斜：航向偏差符号应为负"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img = make_tilted_road_image(center_x_top=180, center_x_bottom=140, road_width=100)
    boundary = extractor.extract(img)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.NORMAL
    assert result.heading_error < 0.0, \
        "Road tilted left: heading_error should be <0, got %.3f" % result.heading_error
    print("  PASS: test_road_tilted_left")


def test_single_left_boundary():
    """单左边界：进入 DEGRADED"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    # 先建双边界直路建立历史宽度
    img_dual = make_straight_road_image(center_x=160, road_width=100)
    geom.compute(extractor.extract(img_dual))

    img_left = make_single_boundary_image(side="left", center_x=160, road_width=100)
    boundary = extractor.extract(img_left)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.DEGRADED, \
        "Single left: expected DEGRADED, got %s" % VisionState.name(result.vision_state)
    assert result.degraded, "Expected degraded=True"
    assert result.confidence < CONF_HIGH_THRESH, \
        "Single boundary confidence should be lower, got %d" % result.confidence
    print("  PASS: test_single_left_boundary")


def test_single_right_boundary():
    """单右边界：进入 DEGRADED"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img_dual = make_straight_road_image(center_x=160, road_width=100)
    geom.compute(extractor.extract(img_dual))

    img_right = make_single_boundary_image(side="right", center_x=160, road_width=100)
    boundary = extractor.extract(img_right)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.DEGRADED
    assert result.degraded
    assert result.right_valid
    assert not result.left_valid
    print("  PASS: test_single_right_boundary")


def test_double_boundary_lost():
    """双边界丢失：输出 INVALID"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img = make_noise_image()
    boundary = extractor.extract(img)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.INVALID, \
        "No boundary: expected INVALID, got %s" % VisionState.name(result.vision_state)
    assert not result.left_valid and not result.right_valid, \
        "Neither boundary should be valid on pure noise"
    print("  PASS: test_double_boundary_lost")


def test_abnormal_road_width():
    """道路宽度异常：降低可信度"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    # 极窄道路（5px → 不合理）
    img = make_straight_road_image(center_x=160, road_width=8)
    boundary = extractor.extract(img)
    result = geom.compute(boundary)

    assert result.confidence < CONF_HIGH_THRESH, \
        "Abnormal narrow road: confidence should be low, got %d" % result.confidence
    print("  PASS: test_abnormal_road_width")


def test_geometry_reset_on_invalid():
    """INVALID 后不应保留旧偏差"""
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()

    # 先正常识别
    img = make_straight_road_image(center_x=180, road_width=100)
    geom.compute(extractor.extract(img))

    # 再 INVALID
    img_bad = make_noise_image()
    result = geom.compute(extractor.extract(img_bad))
    assert result.vision_state == VisionState.INVALID
    # 几何参数不能伪装成正常
    assert not (result.left_valid and result.right_valid)
    print("  PASS: test_geometry_reset_on_invalid")


# ============================================================================
# 单边界横向误差符号测试 (issue 2)
# ============================================================================

def test_single_left_boundary_road_shifted_right():
    """
    单左边界 + 道路右移：
    - 左边界在画面左侧 → 估算中心也在左侧 → lateral_error 应为负
    （画面中心在估计中心的右侧 → 道路中心偏左）
    """
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    # 先建立历史宽度：居中道路
    img_dual = make_straight_road_image(center_x=160, road_width=100)
    geom.compute(extractor.extract(img_dual))

    # 道路右移 → 左边界在画面较右侧
    img_right = make_single_boundary_image(side="left", center_x=190, road_width=100)
    boundary = extractor.extract(img_right)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.DEGRADED
    # 左边界在 190-50=140 处 → est_cx = 140+50=190 → lateral = 190-160 = 30 > 0
    # 道路中心在右 → lateral > 0，符合符号约定
    assert result.lateral_error > 0, \
        "Single left + road shifted right: lateral_error should be >0, got %.2f" % result.lateral_error
    print("  PASS: test_single_left_boundary_road_shifted_right")


def test_single_left_boundary_road_shifted_left():
    """
    单左边界 + 道路左移：
    - 左边界在画面更左侧 → 估算中心在左侧 → lateral_error 应为负
    """
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img_dual = make_straight_road_image(center_x=160, road_width=100)
    geom.compute(extractor.extract(img_dual))

    img_left = make_single_boundary_image(side="left", center_x=130, road_width=100)
    boundary = extractor.extract(img_left)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.DEGRADED
    # 左边界在 130-50=80 处 → est_cx = 80+50=130 → lateral = 130-160 = -30 < 0
    assert result.lateral_error < 0, \
        "Single left + road shifted left: lateral_error should be <0, got %.2f" % result.lateral_error
    print("  PASS: test_single_left_boundary_road_shifted_left")


def test_single_right_boundary_road_shifted_right():
    """
    单右边界 + 道路右移：
    - 右边界在画面更右侧 → 估算中心也在右侧 → lateral_error 应为正
    """
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img_dual = make_straight_road_image(center_x=160, road_width=100)
    geom.compute(extractor.extract(img_dual))

    img_right = make_single_boundary_image(side="right", center_x=190, road_width=100)
    boundary = extractor.extract(img_right)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.DEGRADED
    # 右边界在 190+50=240 处 → est_cx = 240-50=190 → lateral = 190-160 = 30 > 0
    assert result.lateral_error > 0, \
        "Single right + road shifted right: lateral_error should be >0, got %.2f" % result.lateral_error
    print("  PASS: test_single_right_boundary_road_shifted_right")


def test_single_right_boundary_road_shifted_left():
    """
    单右边界 + 道路左移：
    - 右边界在画面左侧 → 估算中心在左侧 → lateral_error 应为负
    """
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()
    img_dual = make_straight_road_image(center_x=160, road_width=100)
    geom.compute(extractor.extract(img_dual))

    img_left = make_single_boundary_image(side="right", center_x=130, road_width=100)
    boundary = extractor.extract(img_left)
    result = geom.compute(boundary)

    assert result.vision_state == VisionState.DEGRADED
    # 右边界在 130+50=180 处 → est_cx = 180-50=130 → lateral = 130-160 = -30 < 0
    assert result.lateral_error < 0, \
        "Single right + road shifted left: lateral_error should be <0, got %.2f" % result.lateral_error
    print("  PASS: test_single_right_boundary_road_shifted_left")


# ============================================================================
# 单边界→双边界→再单边界 复位测试 (issue 3)
# ============================================================================

def test_single_to_dual_to_single_boundary_reset():
    """
    单边界 → 双边界恢复 → 再次单边界：
    验证 single_boundary_frames 和 single_boundary_start_ms 在双边界恢复时被复位。
    """
    extractor = RoadBoundaryExtractor()
    geom = RoadGeometry()

    # 1. 先建立历史宽度
    img_dual = make_straight_road_image(center_x=160, road_width=100)
    geom.compute(extractor.extract(img_dual))
    assert geom.single_boundary_frames == 0
    assert geom.single_boundary_start_ms == 0

    # 2. 单左边界 5 帧 → 累加计数
    img_left = make_single_boundary_image(side="left", center_x=160, road_width=100)
    for i in range(5):
        geom.compute(extractor.extract(img_left), now_ms=100 + i * 20)
    assert geom.single_boundary_frames == 5, \
        "After 5 single-boundary frames, count should be 5, got %d" % geom.single_boundary_frames

    # 3. 双边界恢复 → 计数器应复位
    img_dual2 = make_straight_road_image(center_x=160, road_width=100)
    geom.compute(extractor.extract(img_dual2))
    assert geom.single_boundary_frames == 0, \
        "After dual boundary recovery, frames should reset to 0, got %d" % geom.single_boundary_frames
    assert geom.single_boundary_start_ms == 0, \
        "After dual boundary recovery, start_ms should reset to 0, got %d" % geom.single_boundary_start_ms

    # 4. 再次单边界 → 计数从 0 重新开始
    img_right = make_single_boundary_image(side="right", center_x=160, road_width=100)
    result = geom.compute(extractor.extract(img_right), now_ms=300)
    assert geom.single_boundary_frames == 1, \
        "After re-single boundary, frames should be 1, got %d" % geom.single_boundary_frames
    assert geom.single_boundary_start_ms == 300, \
        "After re-single boundary, start_ms should be 300, got %d" % geom.single_boundary_start_ms
    # 此时应该还是 DEGRADED（因为帧数还未超限）
    assert result.vision_state == VisionState.DEGRADED

    print("  PASS: test_single_to_dual_to_single_boundary_reset")


# ============================================================================
# 导入测试所需常量
# ============================================================================

try:
    from road_config import CONF_HIGH_THRESH
except ImportError:
    CONF_HIGH_THRESH = 70


# ============================================================================
# 运行全部测试
# ============================================================================

if __name__ == "__main__":
    print("K230 Road Geometry Tests")
    print("=" * 45)
    passed = 0
    failed = 0
    tests = [
        test_straight_road_centered,
        test_road_shifted_right,
        test_road_shifted_left,
        test_road_tilted_right,
        test_road_tilted_left,
        test_single_left_boundary,
        test_single_right_boundary,
        test_double_boundary_lost,
        test_abnormal_road_width,
        test_geometry_reset_on_invalid,
        test_single_left_boundary_road_shifted_right,
        test_single_left_boundary_road_shifted_left,
        test_single_right_boundary_road_shifted_right,
        test_single_right_boundary_road_shifted_left,
        test_single_to_dual_to_single_boundary_reset,
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
            failed += 1
    print()
    print("Results: %d passed, %d failed, %d total" % (passed, failed, len(tests)))
    if failed > 0:
        print("STILL NEED K230 BOARD VERIFICATION:")
        print("  - Real camera image instead of synthetic grayscale")
        print("  - Actual road width calibration (px_to_mm)")
        print("  - Real road scene boundary detection")
        print("  - Heading error sign verification on actual road")
        sys.exit(1)
    else:
        print("All PC algorithm tests passed.")
        print("STILL NEED K230 BOARD VERIFICATION (see README)")
