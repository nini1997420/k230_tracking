# K230 道路循迹视觉程序

## 项目定位

K230 车载道路视觉感知模块。把摄像头图像转换为结构化道路信息（偏差、边界、路口），通过 UART 发送给 MSPM0 进行运动控制。

**K230 只负责"看到什么"，MSPM0 负责"怎么走"。**

```
摄像头 → K230视觉 → UART(二进制帧) → MSPM0决策+运动
```

## 文件结构

```
k230_road_vision/
├── main_k230_road_vision.py   # 板端主程序（可独立部署）
├── road_config.py             # 集中配置（部署前改此处）
├── road_geometry.py           # 边界提取 + 偏差计算
├── road_structure.py          # 支路/路口检测
├── vision_protocol.py         # 24字节二进制帧 + CRC
├── README.md                  # 本文件
├── CALIBRATION.md             # 标定指南
├── main.py                    # 合并单文件部署版
└── tests/
    ├── test_road_geometry.py  # PC 算法测试（10项）
    └── test_vision_protocol.py # PC 协议测试（10项）
```

## 快速开始

### PC 端测试（无需硬件）

```bash
cd k230_road_vision
python tests/test_road_geometry.py   # 道路算法测试
python tests/test_vision_protocol.py  # 通信协议测试
```

### K230 板端部署

1. 将 `k230_road_vision/` 目录拷贝到 K230 `/sdcard/`。
2. 在 CanMV IDE 中打开 `main_k230_road_vision.py`。
3. 按实车修改 `road_config.py` 中的摄像头参数和道路参数。
4. 点击"运行"。

### 合并部署（可选）

如果 CanMV 板端不方便多文件导入，使用 `main.py`（单文件合并版）。

## 数据输出

### 横向偏差 (lateral_error)

- `> 0`：道路中心在车辆右侧
- `< 0`：道路中心在车辆左侧
- 单位：0.1mm（如 350 = 35.0mm 右侧）

### 航向偏差 (heading_error)

- `> 0`：道路向右倾斜（顺时针方向）
- `< 0`：道路向左倾斜
- 单位：0.01°（如 200 = 2.00° 向右）

### 视觉状态

- `NORMAL`：双边界有效，可信度高
- `DEGRADED`：单边界，使用历史宽度估算（降低可信度）
- `INVALID`：双边界丢失，应减速/停车

## 运行模式

通过 UART 发送 `[mode,模式名]` 切换：

| 命令 | 模式 | 说明 |
|---|---|---|
| `[mode,idle]` | IDLE | 空闲，低频分析 |
| `[mode,track]` | TRACK | 全速道路循迹（默认） |
| `[mode,turning]` | TURNING | 转弯中，结果标记暂不可用 |
| `[mode,reacquire]` | REACQUIRE | 转弯后重捕获 |
| `[mode,fault]` | FAULT | 异常，发送无效帧 |

## UART 协议

- 波特率：460800 8N1
- 帧长：24 字节
- CRC：CRC16-CCITT-FALSE（覆盖 22 字节）
- **MSPM0 接收端状态：未确认**（接收端源码不在本仓库）

详细帧格式见 `vision_protocol.py` 文件头注释。

## 板端验证清单

按顺序逐阶段验证：

- [ ] 1. 摄像头画面方向和 ROI
- [ ] 2. 边界点/中心线显示
- [ ] 3. 移动道路板验证偏差正负
- [ ] 4. 遮挡边界验证 NORMAL/DEGRADED/INVALID
- [ ] 5. 支路/路口多帧事件
- [ ] 6. 串口工具检查二进制帧
- [ ] 7. MSPM0 接收计数器（不接电机）
- [ ] 8. 低速直线闭环
- [ ] 9. 低速弯道
- [ ] 10. 路口和任务流程

## 已知限制

1. MSPM0 接收端未确认（`app_aim_protocol.c/.h` 不在本仓库）
2. `px_to_mm` 系数 `1.5` 为估计值，需实车标定
3. 路口检测为简化实现，实车需调优阈值
4. `NUMBER` 模式接口保留但未实现
