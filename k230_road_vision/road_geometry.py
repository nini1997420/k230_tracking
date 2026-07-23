# -*- coding: utf-8 -*-
"""
K230 道路循迹 — 道路几何层

负责：
  - 沿采样行提取左右黑色边界点
  - 计算道路中心线
  - 计算横向偏差 (lateral_error)
  - 计算航向偏差 (heading_error)
  - 道路宽度估算
  - 单边界降级处理
  - 可信度评估

本模块可在 PC 端独立运行测试，不依赖 K230 MicroPython 板端模块。
"""

import math

# 延迟导入配置，以支持 PC 测试时 mock
try:
    from road_config import (
        NUM_SAMPLE_ROWS, MIN_LINE_WIDTH, MAX_LINE_WIDTH,
        EXPECTED_ROAD_WIDTH_MIN, EXPECTED_ROAD_WIDTH_MAX,
        BOUNDARY_SEARCH_MARGIN, HEADING_NEAR_ROWS, HEADING_FAR_ROWS,
        SINGLE_BOUNDARY_MAX_FRAMES, SINGLE_BOUNDARY_MAX_MS,
        ROAD_WIDTH_SMOOTH_ALPHA,
        CONF_WEIGHT_VALID_POINTS, CONF_WEIGHT_FIT_RESIDUAL,
        CONF_WEIGHT_WIDTH, CONF_WEIGHT_CONTINUITY, CONF_WEIGHT_STABILITY,
        CONF_HIGH_THRESH, CONF_LOW_THRESH,
    )
except ImportError:
    # PC 测试时的默认值
    NUM_SAMPLE_ROWS = 12
    MIN_LINE_WIDTH = 4
    MAX_LINE_WIDTH = 40
    EXPECTED_ROAD_WIDTH_MIN = 30
    EXPECTED_ROAD_WIDTH_MAX = 200
    BOUNDARY_SEARCH_MARGIN = 10
    HEADING_NEAR_ROWS = [0, 1, 2, 3]
    HEADING_FAR_ROWS = [8, 9, 10, 11]
    SINGLE_BOUNDARY_MAX_FRAMES = 15
    SINGLE_BOUNDARY_MAX_MS = 500
    ROAD_WIDTH_SMOOTH_ALPHA = 0.3
    CONF_WEIGHT_VALID_POINTS = 0.30
    CONF_WEIGHT_FIT_RESIDUAL = 0.25
    CONF_WEIGHT_WIDTH = 0.20
    CONF_WEIGHT_CONTINUITY = 0.15
    CONF_WEIGHT_STABILITY = 0.10
    CONF_HIGH_THRESH = 70
    CONF_LOW_THRESH = 30

# 导出检查：确保所有符号在 PC 测试时可被 mock 替换
__all__ = [
    "RoadBoundaryExtractor",
    "RoadGeometry",
    "VisionState",
    "BoundaryResult",
    "GeometryResult",
]


class VisionState:
    """视觉状态枚举。"""
    NORMAL = 0
    DEGRADED = 1
    INVALID = 2

    @staticmethod
    def name(state):
        return {0: "NORMAL", 1: "DEGRADED", 2: "INVALID"}.get(state, "???")


class BoundaryResult:
    """单帧边界提取结果。"""
    def __init__(self):
        self.left_points = []       # [(x, y), ...] 左边界点（检测坐标）
        self.right_points = []      # [(x, y), ...] 右边界点
        self.left_valid = False     # 左边界有效
        self.right_valid = False    # 右边界有效
        self.center_points = []     # [(x, y), ...] 各行道路中心
        self.num_valid_rows = 0     # 有效采样行数
        self.all_segments = []     # 每行所有黑线段 [(start, end), ...] per row


class GeometryResult:
    """道路几何计算结果。"""
    def __init__(self):
        self.lateral_error = 0.0      # 横向偏差 [检测像素]，>0=中心在右
        self.heading_error = 0.0      # 航向偏差 [rad]，>0=道路向右倾斜
        self.road_width = 0.0         # 道路宽度 [检测像素]
        self.left_valid = False
        self.right_valid = False
        self.vision_state = VisionState.NORMAL
        self.confidence = 0           # 0-100
        self.degraded = False
        self.num_valid_rows = 0
        self.fit_residual = 0.0       # 中心线拟合残差


# ============================================================================
# 边界提取器
# ============================================================================

class RoadBoundaryExtractor:
    """
    从检测图像中沿多行采样提取左右黑色边界。

    输入：320x240 灰度图（numpy array 或 list-of-lists）
    输出：BoundaryResult
    """

    def __init__(self, detect_w=320, detect_h=240,
                 roi_top=30, roi_bottom=220, roi_left=20, roi_right=300,
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

        # 预计算采样行 Y 坐标（从底部到顶部均匀分布）
        self.sample_y = []
        roi_h = roi_bottom - roi_top
        for i in range(num_rows):
            y = roi_bottom - 1 - i * roi_h // max(1, num_rows - 1)
            self.sample_y.append(max(roi_top, min(roi_bottom - 1, y)))

    def extract(self, gray_img) -> BoundaryResult:
        """
        从灰度图中提取边界。

        gray_img: 2D array，gray_img[y][x] = 灰度值 0-255
        """
        result = BoundaryResult()

        for idx, y in enumerate(self.sample_y):
            if y < 0 or y >= self.detect_h:
                continue
            row = gray_img[y] if y < len(gray_img) else None
            if row is None:
                continue

            # 扫描当前行，找黑线区域
            segments = self._find_black_segments(row)
            result.all_segments.append(segments)

            # 在左右半幅分别搜索
            mid_x = (self.roi_left + self.roi_right) // 2
            left_x, right_x = self._match_boundary_pair(segments, mid_x)

            if left_x is not None:
                result.left_points.append((left_x, y))
            if right_x is not None:
                result.right_points.append((right_x, y))
            if left_x is not None and right_x is not None:
                cx = (left_x + right_x) // 2
                result.center_points.append((cx, y))
                result.num_valid_rows += 1

        result.left_valid = len(result.left_points) >= self.num_rows // 3
        result.right_valid = len(result.right_points) >= self.num_rows // 3
        return result

    def _find_black_segments(self, row):
        """在像素行中找黑线连续段 [(start, end), ...]"""
        segments = []
        i = self.roi_left
        while i < min(self.roi_right, len(row)):
            if row[i] < self.gray_thresh:
                start = i
                while i < min(self.roi_right, len(row)) and row[i] < self.gray_thresh:
                    i += 1
                end = i - 1
                w = end - start + 1
                if self.min_line_w <= w <= self.max_line_w:
                    segments.append((start, end))
            else:
                i += 1
        return segments

    def _match_boundary_pair(self, segments, mid_x):
        """从黑线段中匹配左右边界对，返回 (left_x, right_x) 或 None。"""
        left_best = None
        right_best = None
        left_dist = float("inf")
        right_dist = float("inf")

        for seg in segments:
            cx = (seg[0] + seg[1]) // 2
            w = seg[1] - seg[0] + 1
            if cx < mid_x:
                d = mid_x - cx
                # 优先选择距离中线较近且宽度合理的
                if d < left_dist and w < self.max_line_w * 2:
                    left_dist = d
                    left_best = seg
            else:
                d = cx - mid_x
                if d < right_dist and w < self.max_line_w * 2:
                    right_dist = d
                    right_best = seg

        lx = (left_best[0] + left_best[1]) // 2 if left_best else None
        rx = (right_best[0] + right_best[1]) // 2 if right_best else None

        # 宽度合理性检查
        if lx is not None and rx is not None:
            road_w = rx - lx
            if road_w < EXPECTED_ROAD_WIDTH_MIN or road_w > EXPECTED_ROAD_WIDTH_MAX:
                return lx, None  # 右边界不可信

        return lx, rx


# ============================================================================
# 道路几何计算器
# ============================================================================

class RoadGeometry:
    """
    从边界点计算道路几何参数：横向偏差、航向偏差、道路宽度。

    坐标约定：
      - 检测坐标系：320x240，原点左上角
      - 逻辑坐标系：640x480（用于与外部沟通）
      - 内部计算全部使用检测坐标，输出时乘以 scale 转换
    """

    def __init__(self, detect_w=320, detect_h=240,
                 scale_x=640.0 / 320.0, scale_y=480.0 / 240.0):
        self.detect_w = detect_w
        self.detect_h = detect_h
        self.scale_x = scale_x
        self.scale_y = scale_y
        # 画面中心（检测坐标）
        self.cx_det = detect_w / 2.0
        self.cy_det = detect_h / 2.0

        # 历史状态
        self.hist_road_width = 0.0
        self.hist_width_ready = False
        self.single_boundary_frames = 0
        self.single_boundary_start_ms = 0
        self.last_geometry = GeometryResult()

    def compute(self, boundary: BoundaryResult, now_ms=0) -> GeometryResult:
        """
        从 BoundaryResult 计算 GeometryResult。
        """
        g = GeometryResult()
        g.left_valid = boundary.left_valid
        g.right_valid = boundary.right_valid
        g.num_valid_rows = boundary.num_valid_rows

        # ---- 情况1：双边界有效 ----
        if boundary.left_valid and boundary.right_valid:
            lp = boundary.left_points
            rp = boundary.right_points

            # 计算各匹配行的中心与宽度
            centers = []
            widths = []
            for (lx, ly) in lp:
                rx = self._find_right_at_y(rp, ly, 10)
                if rx is not None:
                    centers.append(((lx + rx) / 2.0, ly))
                    widths.append(abs(rx - lx))

            if len(centers) >= 3:
                g.road_width = float(sum(widths)) / max(len(widths), 1)
                self.hist_road_width = g.road_width
                self.hist_width_ready = True

                # 横向偏差：近场中心 vs 画面中心的水平偏移
                near_pts = sorted(centers, key=lambda p: p[1], reverse=True)[:4]
                avg_near_x = sum(p[0] for p in near_pts) / max(len(near_pts), 1)
                # lateral_error > 0: 道路中心在画面中心右侧
                g.lateral_error = avg_near_x - self.cx_det

                # 航向偏差：中场-近场角度
                g.heading_error = self._compute_heading(centers)

                # 拟合残差
                g.fit_residual = self._compute_residual(centers)

                g.vision_state = VisionState.NORMAL
                g.degraded = False
                self.single_boundary_frames = 0
                self.single_boundary_start_ms = 0

            else:
                g = self._degrade_result(g, "insufficient_center_points")

        # ---- 情况2：单边界有效 ----
        elif boundary.left_valid and not boundary.right_valid:
            g = self._single_boundary_estimate(
                boundary, side="left", now_ms=now_ms)
            self.single_boundary_frames += 1
        elif boundary.right_valid and not boundary.left_valid:
            g = self._single_boundary_estimate(
                boundary, side="right", now_ms=now_ms)
            self.single_boundary_frames += 1

        # ---- 情况3：双边界丢失 ----
        else:
            g.vision_state = VisionState.INVALID
            g.degraded = False
            g.confidence = 0
            self.single_boundary_frames += 1

        # ---- 可信度计算 ----
        g.confidence = self._compute_confidence(boundary, g)
        if g.confidence < CONF_LOW_THRESH:
            g.vision_state = VisionState.INVALID

        self.last_geometry = g
        return g

    def _single_boundary_estimate(self, boundary, side, now_ms):
        """单边界时根据历史宽度估算另一侧。"""
        g = GeometryResult()
        g.left_valid = boundary.left_valid
        g.right_valid = boundary.right_valid
        g.degraded = True

        if not self.hist_width_ready:
            g.vision_state = VisionState.INVALID
            g.confidence = 0
            return g

        # 检查单边界持续时间
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

        # 用历史宽度估算
        near_pts = sorted(pts, key=lambda p: p[1], reverse=True)[:4]
        avg_x = sum(p[0] for p in near_pts) / max(len(near_pts), 1)
        if side == "left":
            est_cx = avg_x + self.hist_road_width / 2.0
        else:
            est_cx = avg_x - self.hist_road_width / 2.0

        g.lateral_error = est_cx - self.cx_det
        g.road_width = self.hist_road_width
        g.heading_error = 0.0
        g.vision_state = VisionState.DEGRADED
        g.confidence = int(CONF_LOW_THRESH + 10)
        g.num_valid_rows = boundary.num_valid_rows
        return g

    def _find_right_at_y(self, right_points, y, tolerance):
        """在右侧点集中找与给定 y 最近的点。"""
        best = None
        best_d = tolerance + 1
        for (rx, ry) in right_points:
            d = abs(ry - y)
            if d < best_d:
                best_d = d
                best = rx
        return best

    def _compute_heading(self, centers):
        """从中心点集计算航向偏差（道路倾斜角度）。"""
        if len(centers) < 4:
            return 0.0
        sorted_pts = sorted(centers, key=lambda p: p[1])  # 从远到近

        # 近场：底部 1/3 的点
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
        # heading_error > 0: 近场中心在中场中心右侧 → 道路向右倾斜
        return math.atan2(dx, dy)

    def _compute_residual(self, centers):
        """计算中心点拟合残差（RMS）。"""
        if len(centers) < 4:
            return 0.0
        xs = [p[0] for p in centers]
        ys = [p[1] for p in centers]
        n = len(xs)
        if n < 2:
            return 0.0
        # 简单线性拟合
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((ys[i] - mean_y) ** 2 for i in range(n))
        if abs(den) < 1e-6:
            return 0.0
        slope = num / den
        intercept = mean_x - slope * mean_y
        # RMS 残差
        residual_sum = sum((xs[i] - (slope * ys[i] + intercept)) ** 2
                           for i in range(n))
        return math.sqrt(residual_sum / n)

    def _compute_confidence(self, boundary, geom):
        """综合多因素计算可信度 0-100。"""
        score = 0.0
        total_w = 0.0

        # 有效点数权重
        if boundary.num_valid_rows > 0:
            point_ratio = boundary.num_valid_rows / float(NUM_SAMPLE_ROWS)
            score += CONF_WEIGHT_VALID_POINTS * point_ratio * 100
        total_w += CONF_WEIGHT_VALID_POINTS

        # 拟合残差权重（残差越小越高）
        if geom.fit_residual >= 0:
            residual_score = max(0, 100 - geom.fit_residual * 20)
            score += CONF_WEIGHT_FIT_RESIDUAL * residual_score
        total_w += CONF_WEIGHT_FIT_RESIDUAL

        # 道路宽度合理性
        if geom.road_width > 0:
            if EXPECTED_ROAD_WIDTH_MIN <= geom.road_width <= EXPECTED_ROAD_WIDTH_MAX:
                width_score = 100
            else:
                dist = min(abs(geom.road_width - EXPECTED_ROAD_WIDTH_MIN),
                          abs(geom.road_width - EXPECTED_ROAD_WIDTH_MAX))
                width_score = max(0, 100 - dist * 2)
            score += CONF_WEIGHT_WIDTH * width_score
        total_w += CONF_WEIGHT_WIDTH

        # 边界连续性
        cont_score = 50
        if boundary.left_valid and boundary.right_valid:
            cont_score = 100
        elif boundary.left_valid or boundary.right_valid:
            cont_score = 50
        score += CONF_WEIGHT_CONTINUITY * cont_score
        total_w += CONF_WEIGHT_CONTINUITY

        # 帧间稳定性（与上一帧比较）
        if self.last_geometry and self.last_geometry.road_width > 0:
            if geom.road_width > 0:
                change = abs(geom.road_width - self.last_geometry.road_width)
                stability_score = max(0, 100 - change * 3)
            else:
                stability_score = 0
        else:
            stability_score = 80  # 首帧默认中等
        score += CONF_WEIGHT_STABILITY * stability_score
        total_w += CONF_WEIGHT_STABILITY

        if total_w <= 0:
            return 0
        return int(min(100, max(0, score / total_w)))

    def _degrade_result(self, g, reason):
        """降级处理辅助。"""
        g.vision_state = VisionState.DEGRADED
        g.degraded = True
        g.confidence = CONF_LOW_THRESH + 5
        return g

    def reset_history(self):
        """重置历史状态（转弯后重捕获时调用）。"""
        self.hist_width_ready = False
        self.hist_road_width = 0.0
        self.single_boundary_frames = 0
        self.single_boundary_start_ms = 0

    def to_logical(self, det_val, axis="x"):
        """将检测坐标值转换为逻辑坐标。"""
        return det_val * (self.scale_x if axis == "x" else self.scale_y)
