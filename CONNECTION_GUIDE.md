# 智能止鼾枕 - 连接指南

适用于 **Orange Pi 5 Plus (RK3588)**，包含 PCIe/USB 蓝牙、Arduino、压力传感器与总控程序的连接与启动步骤。

---

## 一、硬件连接总览

```
                    Orange Pi 5 Plus (RK3588)
    ┌─────────────────────────────────────────────────────────┐
    │                                                           │
    │   PCIe (M.2 或扩展) → 转接板(USB) → [ 蓝牙模块 Realtek ]   │  ← 与手机/App 通信
    │                                                           │
    │   USB 口 ①  ───────────────→ [ Arduino Uno ]              │  ← 气囊控制 (ttyACM0)
    │                                                           │
    │   USB 口 ②  ───────────────→ [ CH340 串口 ]              │  ← 压力传感器 (ttyUSB0)
    │                                                           │
    └─────────────────────────────────────────────────────────┘
```

- **蓝牙**：通过 PCIe 转 USB 的扩展板接 USB 蓝牙棒（系统里显示为 USB 设备，如 Realtek 0bda:b85b）。
- **Arduino**：USB 直连板子，用于气囊充放气控制。
- **压力传感器**：通过 CH340 等 USB 转串口连接板子，用于压力矩阵数据。

---

## 二、PCIe / USB 蓝牙连接

### 2.1 物理连接

1. 将 **M.2 转 USB 转接卡**（或带 USB 的 PCIe 扩展板）插入 Orange Pi 5 Plus 的 **M.2 M key** 插槽（或 PCIe 排针扩展）。
2. 将 **USB 蓝牙适配器**（如 Realtek）插在转接卡/扩展板的 USB 口上。
3. 上电后，在系统里应看到 USB 设备，例如：
   ```bash
   lsusb
   # 例如: Bus 005 Device 003: ID 0bda:b85b Realtek Semiconductor Corp. Bluetooth Radio
   ```

### 2.2 确认蓝牙控制器

```bash
# 查看蓝牙控制器
bluetoothctl list

# 若有多个（板载 + 外接），记下外接 Realtek 的 MAC，后面用 select 选它
bluetoothctl show
```

---

## 三、Arduino 连接（气囊控制）

1. 用 **USB 线** 将 **Arduino Uno R3** 接到 Orange Pi 的 USB 口。
2. 确认设备节点（通常为 `ttyACM0`）：
   ```bash
   ls -l /dev/ttyACM*
   # 或
   dmesg | grep -i "ttyACM\|arduino"
   ```
3. 上传固件：在 PC 上用 Arduino IDE 打开 `Actuator/airbag_controller/airbag_controller.ino`，选择板型与串口后上传。

---

## 四、压力传感器连接（CH340）

1. 将带 **CH340** 的压力传感器板通过 **USB** 接到 Orange Pi。
2. 确认设备节点（通常为 `ttyUSB0`）：
   ```bash
   ls -l /dev/ttyUSB*
   dmesg | grep -i "ch341\|ttyUSB"
   ```

**注意**：若同时插着 Arduino 和 CH340，`ttyACM0` 一般为 Arduino，`ttyUSB0` 一般为 CH340；总控程序会自动扫描串口，必要时可在代码中写死端口。

---

## 五、蓝牙配对（与手机/上位机）

### 5.1 启动蓝牙并进入 bluetoothctl

```bash
sudo systemctl enable --now bluetooth
sudo rfkill unblock bluetooth
bluetoothctl
```

### 5.2 在 bluetoothctl 中操作（若有多控制器先选外接）

```text
list
select <外接控制器 MAC>    # 若有多个控制器，选 Realtek 对应的
power on
agent on
default-agent
discoverable on
pairable on
scan on
```

用手机或上位机搜索并发起配对；在 bluetoothctl 中看到设备后：

```text
pair <手机/设备 MAC>
trust <手机/设备 MAC>
connect <手机/设备 MAC>
```

配对成功后可用 `scan off`，再 `quit` 退出。

### 5.3 若出现 br-connection-create-socket

表示本机 BlueZ/内核在建立经典蓝牙连接时创建 socket 失败，可先：

- 重启蓝牙：`sudo systemctl restart bluetooth`
- 换用 **TCP 模式** 跑总控（见下文），用 WiFi 连接代替蓝牙。

---

## 六、运行总控程序

总控脚本位于项目根目录：`smart_pillow_controller.py`。

### 6.1 依赖

```bash
cd /root/Pillow
pip install -r requirements.txt   # 至少需要 pyserial
```

### 6.2 蓝牙模式（需本机支持 Bluetooth socket）

```bash
cd /root/Pillow
python3 smart_pillow_controller.py --bt-channel 1 --arduino-baud 9600
```

若报错 `Address family not supported` 或 `br-connection-create-socket`，改用 TCP 模式。

### 6.3 TCP 模式（推荐在蓝牙不可用时使用）

```bash
cd /root/Pillow
python3 smart_pillow_controller.py --tcp-port 8765 --arduino-baud 9600
```

程序监听 `0.0.0.0:8765`，协议与蓝牙一致（按行收发）。手机/上位机通过 **WiFi** 连接板子 IP，例如 `192.168.x.x:8765`，发送 `START_SLEEP`、`PILLOW_UP:L1` 等指令。

### 6.4 查看本机 IP（供 TCP 连接）

```bash
ip -4 addr show | grep inet
# 或
hostname -I
```

---

## 七、串口与设备对应（参考）

| 设备         | 常见节点    | 用途           |
|--------------|-------------|----------------|
| Arduino Uno  | `/dev/ttyACM0` | 气囊控制       |
| CH340 串口   | `/dev/ttyUSB0` | 压力传感器数据 |

总控在 `START_SLEEP` 时会自动扫描串口并启动压力监测脚本；若自动选择不准，可在代码中指定端口。

---

## 八、如何确认连接成功

按下面逐项检查，看到“预期结果”即表示该部分连接成功。

### 8.1 USB 设备（蓝牙、Arduino、压力传感器）

在终端执行：

```bash
lsusb
```

**连接成功时** 应看到类似：

- `0bda:b85b` 或 `Realtek ... Bluetooth` → 蓝牙已识别
- `2341:0043` 或 `Arduino` → Arduino 已识别
- `1a86:7523` 或 `CH340` / `QinHeng` → 压力传感器串口已识别

再确认串口节点存在：

```bash
ls -l /dev/ttyACM0 /dev/ttyUSB0
```

**连接成功时**：两个文件都存在（或至少存在你当前插着的那个），且无 “Permission denied”。

---

### 8.2 蓝牙控制器与配对

```bash
bluetoothctl list
```

**连接成功时**：至少有一行控制器，例如 `Controller XX:XX:XX:XX:XX:XX ...`。

查看是否已配对设备：

```bash
bluetoothctl paired-devices
```

**配对成功时**：会列出已配对的设备及其 MAC。

查看某设备是否已连接（把 `<MAC>` 换成实际地址）：

```bash
bluetoothctl info <MAC>
```

**连接成功时**：输出里有 `Connected: yes`。

---

### 8.3 Arduino 串口通信

先确认端口（Arduino 多为 `ttyACM0`），再发一条命令看是否有回复：

```bash
# 安装 cu 或用 Python 发（波特率 9600）
python3 -c "
import serial, time
s = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)
s.write(b'S00')
time.sleep(0.3)
print(s.read(s.in_waiting or 1).decode(errors='ignore'))
s.close()
"
```

**连接成功时**：能打开串口且无报错，可能看到 Arduino 回复如 `OK:All operations stopped` 或类似输出。

---

### 8.4 如何确认香橙派已向 Arduino 发指令

总控在**每次**向 Arduino 发送串口命令时，会在**香橙派终端**打出一行日志，例如：

```text
-> Arduino: P11 (已发送)
-> Arduino: V21 (已发送)
-> Arduino: S00 (已发送)
```

**核查步骤：**

1. **在香橙派上**运行总控（TCP 或蓝牙模式），保持终端开着。
2. **从 PC/手机**发一条会控制气囊的指令，例如：`PILLOW_UP:L1` 或 `PILLOW_DOWN:L2`。
3. **看香橙派终端**：
   - 若出现 `Arduino 已连接: /dev/ttyACM0`（或 ttyUSBx），说明已连上 Arduino。
   - 若再出现 `-> Arduino: P11 (已发送)`（或 V21、S00 等），说明**已经向 Arduino 发出了该指令**。
4. **看 Arduino 行为**：对应气囊应开始充气/放气；若 Arduino 接了串口监视器（另一台 PC），可看到 Arduino 打印的 `OK:Inflating...` 等回复。

若**没有**出现 `-> Arduino: xxx (已发送)`：
- 先看是否出现 `ERR:ARDUINO_CONNECT_FAILED`（未找到/未连上串口）。
- 若连上却仍无该行，可能是发的是不经过 Arduino 的指令（如 `ALARM:07:30`），可改发 `PILLOW_UP:L1` 再试。

总控会**优先使用像 Arduino 的串口**（ttyACM、描述里含 arduino），避免误用压力传感器的 CH340 串口。

---

### 8.5 总控程序（TCP 或蓝牙）是否在“接客”

**TCP 模式**：先启动总控：

```bash
cd /root/Pillow
python3 smart_pillow_controller.py --tcp-port 8765 --arduino-baud 9600
```

**连接成功时**：终端出现类似 `TCP listening on 0.0.0.0:8765`，程序不退出。

再用另一台机器或手机（同 WiFi）测试（把 `板子IP` 换成实际 IP，如 `192.168.1.100`）：

```bash
# 用 nc 测试（在 PC 或另一终端）
echo "PILLOW_UP:L1" | nc -q 1 板子IP 8765
```

**连接成功时**：能收到一行回复，例如 `OK:PILLOW_UP:L1`；总控终端里会打印收到的指令。

**蓝牙模式**：启动总控后，用已配对的手机/上位机连接 RFCOMM channel 1，发送一行 `PILLOW_UP:L1`。

**连接成功时**：总控终端出现 `App connected`，并打印 `<- APP: PILLOW_UP:L1`；手机/上位机收到 `OK:PILLOW_UP:L1`。

---

### 8.6 一句话自检

| 要确认的项     | 命令 / 操作                    | 成功时的表现                         |
|----------------|--------------------------------|--------------------------------------|
| USB 设备       | `lsusb`                        | 能看到蓝牙、Arduino、CH340 对应条目  |
| 串口节点       | `ls /dev/ttyACM0 /dev/ttyUSB0` | 两个或至少一个存在且可读             |
| 蓝牙已配对     | `bluetoothctl paired-devices` | 列表中有你的手机/上位机               |
| 蓝牙已连接     | `bluetoothctl info <MAC>`      | `Connected: yes`                     |
| Arduino 通信   | 用 Python 发 `S00`             | 能打开串口并有回复或无报错            |
| 总控 TCP 监听  | 启动后看终端                   | 出现 `TCP listening on 0.0.0.0:8765` |
| 总控收到指令   | 用 nc 或 App 发一条指令        | 收到 `OK:...` 且总控终端有日志        |

---

## 九、快速检查清单

- [ ] `lsusb` 中能看到 Realtek 蓝牙、Arduino、CH340
- [ ] `bluetoothctl list` 能看到外接蓝牙控制器
- [ ] Arduino 已上传 `airbag_controller.ino`
- [ ] `/dev/ttyACM0` 与 `/dev/ttyUSB0` 存在且权限正常
- [ ] 蓝牙已与手机/上位机配对并信任
- [ ] 总控以 TCP 或蓝牙模式正常启动，能收到指令并回复

**如何逐项确认**：见上一节「八、如何确认连接成功」。

完成以上步骤后，即可通过蓝牙或 TCP 发送指令控制气囊并进入/退出睡眠模式。
