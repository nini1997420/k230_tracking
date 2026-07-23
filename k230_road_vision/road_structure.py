# -*- coding: utf-8 -*-
"""
K230 道路循迹 — 道路结构识别层

负责：
  - 左/右支路检测
  - 十字/扩宽区域检测
  - 路口阶段估计（APPROACHING、NEAR、AT、PASSED）
  - 多帧确认与进入/离开迟滞
  - 转弯中结果标记不可用于正常循迹

注意：本模块只报告"看到什么"，不输出 TURN_LEFT/TURN_RIGHT。
     路线决策属于 MSPM0。
"""

try:
    from road_config import (
        JUNCTION_WIDTH_INCREASE_RATIO,
        JUNCTION_CONFIRM_FRAMES,
        JUNCTION_COOLDOWN_FRAMES,
        JUNCTION_DISTANCE_NEAR_PX,
        JUNCTION_DISTANCE_MID_PX,
    )
except ImportError:
    JUNCTION_WIDTH_INCREASE_RATIO = 1.5
    JUNCTION_CONFIRM_FRAMES = 5
    JUNCTION_COOLDOWN_FRAMES = 20
    JUNCTION_DISTANCE_NEAR_PX = 40
    JUNCTION_DISTANCE_MID_PX = 80

# 检测图像高度（用于 junction_y_px 归一化）
DETECT_HEIGHT = 240


class JunctionStage:
    """路口阶段枚举。"""
    NONE = 0
    APPROACHING = 1    # 远处发现路口
    NEAR = 2           # 路口接近
    AT = 3             # 到达转弯参考位置
    PASSED = 4         # 已通过

    @staticmethod
    def name(stage):
        return {0: "NONE", 1: "APPROACHING", 2: "NEAR",
                3: "AT", 4: "PASSED"}.get(stage, "???")


class StructureResult:
    """道路结构识别结果。"""
    def __init__(self):
        self.left_branch = False           # 左支路存在
        self.right_branch = False          # 右支路存在
        self.intersection_candidate = False  # 十字/扩宽候选
        self.junction_stage = JunctionStage.NONE
        self.junction_distance = 0         # 0=far(>80px) 1=mid 2=near 3=at
        self.junction_distance_px = 0      # 路口距离 [检测像素]
        self.structure_confirmed = False   # 结构多帧确认


class RoadStructureDetector:
    """
    道路结构检测器。

    综合多特征检测路口，包括：
      - 道路宽度突变
      - 左/右边界外扩
      - all_segments 中的额外黑线段（支路）
      - 多帧一致性
      - 位置随车辆运动
    """

    def __init__(self):
        # 支路候选计数器
        self.left_branch_hits = 0
        self.right_branch_hits = 0
        self.intersection_hits = 0

        # 历史
        self.prev_left_branch = False
        self.prev_right_branch = False
        self.prev_intersection = False
        self.cooldown_counter = 0

        # 宽度历史（用于检测宽度突变）
        self.width_history = []
        self.max_width_history = 10

        # 路口确认与位置追踪
        self._confirmed_junction_active = False  # 路口已确认且未离开
        self.junction_y_px = -1                  # 路口特征最强的 Y 坐标（图像坐标）
        self._last_junction_stage = JunctionStage.NONE
        self._last_junction_distance = 0
        self._last_junction_distance_px = 0

    # ------------------------------------------------------------------
    # 主检测入口
    # ------------------------------------------------------------------

    def detect(self, boundary, geom) -> StructureResult:
        """
        从 BoundaryResult 和 GeometryResult 检测道路结构。

        boundary: BoundaryResult（当前帧边界点，包含 all_segments）
        geom: GeometryResult（当前帧几何参数）
        """
        result = StructureResult()
        self.cooldown_counter = max(0, self.cooldown_counter - 1)

        # ---- 1. INVALID → 重置并返回空结果 ----
        if geom.vision_state == 2:
            self._decay_counts()
            # 若之前已确认路口，INVALID 视为特征消失，进入 PASSED/冷却
            if self._confirmed_junction_active:
                self._transition_to_passed()
            return result

        # ---- 2. 记录宽度历史 ----
        if geom.road_width > 0:
            self.width_history.append(geom.road_width)
            if len(self.width_history) > self.max_width_history:
                self.width_history.pop(0)

        # ---- 3. 特征1：道路宽度增大 ----
        width_increase = False
        if len(self.width_history) >= 3 and geom.road_width > 0:
            avg_old = sum(self.width_history[:-1]) / max(len(self.width_history) - 1, 1)
            if avg_old > 0:
                ratio = geom.road_width / avg_old
                if ratio > JUNCTION_WIDTH_INCREASE_RATIO:
                    width_increase = True

        # ---- 4. 特征2：左右边界外扩 ----
        left_expanded = self._check_boundary_expansion(
            boundary.left_points, side="left")
        right_expanded = self._check_boundary_expansion(
            boundary.right_points, side="right")

        # ---- 5. 特征3：all_segments 中的额外黑线段 ----
        extra_left, extra_right, extra_segments_y = self._check_extra_segments(boundary)

        # ---- 6. 计数各支路类型 hits ----
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

        if width_increase and (has_left_feature or has_right_feature):
            self.intersection_hits += 1
        else:
            self.intersection_hits = max(0, self.intersection_hits - 1)

        # 当前帧是否有任何活跃特征
        any_hit = (self.left_branch_hits > 0 or self.right_branch_hits > 0 or
                   self.intersection_hits > 0)

        # ---- 7. 路口确认：hits 达到确认帧数 ----
        if not self._confirmed_junction_active and self.cooldown_counter <= 0:
            if (self.left_branch_hits >= JUNCTION_CONFIRM_FRAMES or
                    self.right_branch_hits >= JUNCTION_CONFIRM_FRAMES or
                    self.intersection_hits >= JUNCTION_CONFIRM_FRAMES):
                self._confirmed_junction_active = True
                # 从当前帧特征位置记录 junction_y_px
                self.junction_y_px = self._compute_junction_y(
                    extra_segments_y, boundary, width_increase)

        # ---- 8. 路口已确认：持续输出路口信息 ----
        if self._confirmed_junction_active:

            # 更新 junction_y_px（特征仍存在时，跟踪路口位置下移）
            if any_hit:
                current_y = self._compute_junction_y(
                    extra_segments_y, boundary, width_increase)
                if current_y > 0 and current_y > self.junction_y_px:
                    self.junction_y_px = current_y

            # 设置输出标志
            result.left_branch = True
            result.right_branch = True
            result.intersection_candidate = True
            result.structure_confirmed = True

            # ---- 9. 特征消失：路口已通过 ----
            if not any_hit:
                self._transition_to_passed()

            # 从 junction_y_px 计算阶段与距离
            if self.junction_y_px > 0:
                stage, distance, dist_px = self._estimate_junction_stage(
                    self.junction_y_px)
            else:
                stage, distance, dist_px = JunctionStage.APPROACHING, 1, JUNCTION_DISTANCE_MID_PX

            # 若处于 PASSED 阶段（_transition_to_passed 已设置），沿用 PASSED
            if self._last_junction_stage == JunctionStage.PASSED:
                stage = JunctionStage.PASSED
                distance = self._last_junction_distance
                dist_px = self._last_junction_distance_px

            result.junction_stage = stage
            result.junction_distance = distance
            result.junction_distance_px = dist_px

            # 保存最后输出值
            self._last_junction_stage = stage
            self._last_junction_distance = distance
            self._last_junction_distance_px = dist_px

            return result

        # ---- 路口未确认时的正常输出 ----
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

    # ------------------------------------------------------------------
    # 特征检测方法
    # ------------------------------------------------------------------

    def _check_boundary_expansion(self, points, side):
        """检查边界点是否存在明显的外扩（远离画面中心）。"""
        if not points or len(points) < 3:
            return False
        # 比较最近点和较远点的 X 坐标
        sorted_pts = sorted(points, key=lambda p: p[1], reverse=True)
        near = sorted_pts[:3]
        far = sorted_pts[-3:]
        if len(near) < 2 or len(far) < 2:
            return False
        near_avg_x = sum(p[0] for p in near) / len(near)
        far_avg_x = sum(p[0] for p in far) / len(far)
        diff = far_avg_x - near_avg_x
        if side == "left":
            return diff < -8  # 左边界向左外扩
        else:
            return diff > 8   # 右边界向右外扩

    def _check_extra_segments(self, boundary):
        """
        检查 all_segments 中是否存在额外黑线段（支路候选）。

        all_segments 是列表的列表：每个采样行一条目，
        每个条目是该行所有黑线段的 [(start, end), ...] 列表
        （在边界匹配过滤之前的原始段）。

        Returns:
            (extra_left: bool, extra_right: bool, extra_segments_y: int)
            extra_segments_y 是发现额外段的最低 Y 坐标，-1 表示无。
        """
        if not hasattr(boundary, 'all_segments') or not boundary.all_segments:
            return False, False, -1

        extra_left = False
        extra_right = False
        extra_y = -1

        # 构建 Y→边界X 的快速查找
        left_by_y = {}
        right_by_y = {}
        for x, y in boundary.left_points:
            left_by_y[int(y)] = x
        for x, y in boundary.right_points:
            right_by_y[int(y)] = x

        num_rows = len(boundary.all_segments)
        if num_rows == 0:
            return False, False, -1

        roi_top = 30
        roi_bottom = 220

        for i, segments in enumerate(boundary.all_segments):
            if not segments:
                continue

            # 估计该行的 Y 坐标（与 RoadBoundaryExtractor 采样行分布一致）
            y = roi_bottom - 1 - i * (roi_bottom - roi_top) // max(1, num_rows - 1)

            lx = left_by_y.get(y)
            rx = right_by_y.get(y)
            if lx is None or rx is None:
                continue

            for seg_start, seg_end in segments:
                seg_cx = (seg_start + seg_end) // 2
                seg_w = seg_end - seg_start + 1

                # 跳过极窄噪声
                if seg_w < 3:
                    continue

                # 段在左右边界之间 → 双支路候选
                if lx + 5 < seg_cx < rx - 5:
                    extra_left = True
                    extra_right = True
                    if y > extra_y:
                        extra_y = y

                # 段显著在左边界左侧
                elif seg_cx < lx - 12:
                    extra_left = True
                    if y > extra_y:
                        extra_y = y

                # 段显著在右边界右侧
                elif seg_cx > rx + 12:
                    extra_right = True
                    if y > extra_y:
                        extra_y = y

        return extra_left, extra_right, extra_y

    # ------------------------------------------------------------------
    # 路口阶段估计
    # ------------------------------------------------------------------

    def _estimate_junction_stage(self, junction_y_px):
        """
        根据路口在图像中的 Y 坐标估计路口阶段。

        junction_y_px: 路口特征最强的图像 Y 坐标（从上往下）。
        距离 = DETECT_HEIGHT - junction_y_px（从底部算起的距离）。
        """
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

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _compute_junction_y(self, extra_segments_y, boundary, width_increase):
        """
        综合多特征计算当前帧的路口特征 Y 坐标。
        优先使用 extra_segments_y（最直接），其次使用边界外扩位置。
        """
        y = extra_segments_y
        if y <= 0:
            # 退而求其次：用边界最低点来估计
            all_pts = boundary.left_points + boundary.right_points
            if all_pts:
                y = max(p[1] for p in all_pts)
        return y

    def _transition_to_passed(self):
        """路口特征消失，转入 PASSED 阶段并启动冷却。"""
        self._confirmed_junction_active = False
        self.cooldown_counter = JUNCTION_COOLDOWN_FRAMES
        self._last_junction_stage = JunctionStage.PASSED
        self._last_junction_distance = 4
        self._last_junction_distance_px = 0
        self.junction_y_px = -1

    def _decay_counts(self):
        """INVALID 时缓慢衰减计数器。"""
        self.left_branch_hits = max(0, self.left_branch_hits - 1)
        self.right_branch_hits = max(0, self.right_branch_hits - 1)
        self.intersection_hits = max(0, self.intersection_hits - 1)

    def _reset_counts(self):
        """完全重置计数器（转弯结束后调用）。"""
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
        """转弯结束后重置路口状态。"""
        self._reset_counts()
