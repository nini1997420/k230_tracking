# K230 道路循迹 标定指南

## 1. 摄像头安装

- 摄像头朝车辆前进方向
- 位置接近车体纵向中心
- 角度调整至能同时看到左右道路黑线
- 近场（车体前方 5-15cm）、中场（15-40cm）、远场（>40cm）

安装固定后记录：

```
摄像头安装高度（地面到镜头）：____ mm
摄像头俯角（相对水平面）：____ 度
摄像头水平偏移（左/右）：____ mm
```

## 2. 道路 ROI 标定

`road_config.py` 中的 ROI 参数定义检测区域：

```python
ROI_TOP = 30      # 排除远处
ROI_BOTTOM = 220  # 排除车体
ROI_LEFT = 20     # 左侧边界
ROI_RIGHT = 300   # 右侧边界
```

**步骤**：
1. 将车放上直路
2. 开启 DISPLAY_TO_IDE 查看实时画面
3. 调整 ROI 使道路区域在 ROI 内、车体部件在 ROI 外
4. ROI_BOTTOM 应高于车体黑色结构

## 3. 灰度阈值标定

```python
LINE_GRAY_THRESH = 80  # 0-255，越小越严格
```

- 太高（>120）：地面污渍被误判为黑线
- 太低（<40）：黑线检测不到
- 推荐：在 IDE 中打印采样行的灰度值，取黑线平均值 + 20

## 4. 道路宽度标定

```python
EXPECTED_ROAD_WIDTH_MIN = 30   # [px]
EXPECTED_ROAD_WIDTH_MAX = 200  # [px]
```

**步骤**：
1. 车辆放在直路正中
2. 记录 `road_width` 输出值（打印机状态信息）
3. 设置 MIN 为记录值 × 0.7，MAX 为记录值 × 2.0

## 5. px_to_mm 像素-毫米转换

```python
px_to_mm = 1.5  # 主程序中待标定
```

**步骤**：
1. 在道路平面放已知宽度 L mm 的参照物
2. 测量参照物在 ROI 底部（最近处）的像素宽度 w px
3. px_to_mm = L / w
4. 由于透视，此值只在一段深度范围内有效

## 6. 偏差正负验证

**横向偏差**：
1. 车辆在直路正中 → lateral_error ≈ 0
2. 车辆右移 20mm → lateral_error > 0 且数值约 200 (20.0mm)
3. 车辆左移 20mm → lateral_error < 0

**航向偏差**：
1. 直路 → heading_error ≈ 0
2. 道路右转弯 → heading_error > 0
3. 道路左转弯 → heading_error < 0

若正负相反，修改 `road_geometry.py` 中对应符号，或使用软件取反。

## 7. 可信度阈值标定

```python
CONF_HIGH_THRESH = 70  # NORMAL 阈值
CONF_LOW_THRESH = 30   # 低于此为 INVALID
```

观察实车运行时的 confidence 值分布：
- 正常直路：应稳定在 70-100
- 弯道/远处道路模糊：60-80
- 单边界：30-50
- 无道路：0-30

调整阈值使 NORMAL/DEGRADED/INVALID 切换符合预期。

## 8. 路口检测标定

```python
JUNCTION_WIDTH_INCREASE_RATIO = 1.5  # 宽度放大倍数
JUNCTION_CONFIRM_FRAMES = 5          # 连续确认帧数
JUNCTION_COOLDOWN_FRAMES = 20        # 冷却帧数（防重复）
```

标定时在路口手动确认：
1. 是否在路口前 APPROACHING
2. 是否在通过时触发
3. 是否每帧重复触发（增大 COOLDOWN）
4. 直路是否误报（减小 RATIO 或增大 CONFIRM_FRAMES）
