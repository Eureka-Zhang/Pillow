# 智能止鼾枕 - 整体运行说明

本仓库为智能止鼾枕的**总控 + 传感器 + 执行器**联合代码。按下面方式可一次性跑起「App 连接 → 睡眠模式 → 打鼾/睡姿融合 → 气囊调控」全流程。

---

## 一、硬件与串口

| 设备         | 典型节点     | 说明                     |
|--------------|--------------|--------------------------|
| Arduino 气囊 | `/dev/ttyACM0` | 充气/放气控制            |
| 压力传感器   | `/dev/ttyUSB0` | 压力矩阵睡姿推理         |
| 麦克风       | 系统默认     | 语音打鼾/睡姿（可选）    |

详见 [CONNECTION_GUIDE.md](CONNECTION_GUIDE.md)。

---

## 二、整体 App 联合运行（推荐）

**一步启动总控**（蓝牙或 TCP），由 App 发指令控制睡眠模式与气囊：

```bash
cd /root/Pillow

# 方式 A：TCP 监听（无蓝牙或调试时用）
python3 smart_pillow_controller.py --tcp-port 8765

# 方式 B：蓝牙 RFCOMM（需先配对并配置 SDP）
python3 smart_pillow_controller.py --bt-channel 1
```

- **TCP 模式**：App 或本机用 `telnet <设备IP> 8765` / `nc <IP> 8765` 连接，按行发送指令。
- 总控会**自动检测** Arduino 与压力传感器串口；收到 **START_SLEEP** 后启动「实时打鼾与睡姿监控」子进程（压力+语音融合，自动止鼾），收到 **STOP_SLEEP** 停止并回传睡眠报告。

**协议（每行一条）**

| 指令            | 说明                         |
|-----------------|------------------------------|
| `START_SLEEP`   | 进入睡眠模式，启动打鼾/睡姿监控与自动调节 |
| `STOP_SLEEP`    | 退出睡眠模式，返回 SLEEP_REPORT        |
| `ALARM:HH:MM`   | 设置闹钟（如 `ALARM:07:00`）          |
| `PILLOW_UP:L1`  | 指定区域充气（L1/L2/R2/R1）           |
| `PILLOW_DOWN:L2`| 指定区域放气                      |
| `PILLOW_RESET`  | 全部放气约 30s                    |

**用 TCP 快速自测**

```bash
# 终端 1：启动总控
python3 smart_pillow_controller.py --tcp-port 8765

# 终端 2：连接并发指令
nc localhost 8765
START_SLEEP
# ... 一段时间后 ...
STOP_SLEEP
```

---

## 三、仅跑「实时打鼾与睡姿」（不连 App）

不通过总控、直接跑融合监控与气囊控制（例如调试或无 App 时）：

```bash
cd /root/Pillow
python3 realtime_snore_posture_control.py \
  --arduino-port /dev/ttyACM0 \
  --pressure-port /dev/ttyUSB0
```

- 会持续读压力推理与语音推理输出，做融合判断，并在终端打印 `[判断]` 与 Arduino 回复。
- 检测到打鼾：先全放气 30s；若仍打鼾则按融合睡姿执行对应充气序列。

---

## 四、一键启动脚本（TCP 总控）

项目根目录提供脚本，默认用 TCP 端口 8765 启动总控，便于联合运行：

```bash
cd /root/Pillow
chmod +x run_app.sh
./run_app.sh
```

之后用 App 或 `nc <本机IP> 8765` 连接并发送 `START_SLEEP` / `STOP_SLEEP` 等即可。日志与 Arduino 输出均在同一终端。

---

## 五、目录结构（与运行相关）

```
Pillow/
├── smart_pillow_controller.py   # 总控入口（蓝牙/TCP + Arduino + 睡眠模式）
├── realtime_snore_posture_control.py  # 打鼾+睡姿融合与自动止鼾（可由总控拉起）
├── run_app.sh                   # 一键启动总控（TCP）
├── Sensor/
│   ├── Pressure/                # 压力推理（串口读数据，输出 [Result] 行）
│   └── Voice/                   # 语音打鼾/睡姿推理
└── Actuator/
    └── airbag_controller/       # Arduino 气囊控制器固件
```

整体 **App 联合代码运行** 即：启动总控 → App 连接 → 发 **START_SLEEP** → 总控拉起 `realtime_snore_posture_control.py` → 压力+语音融合判断 → 通过串口控制 Arduino 气囊。
