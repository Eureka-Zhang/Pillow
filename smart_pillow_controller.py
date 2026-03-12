#!/usr/bin/env python3
"""
智能止鼾枕总控

职责：
1) 主线程打开蓝牙 RFCOMM 或 TCP，接收并解析 App 指令
2) 通过串口向 Arduino 下发气囊控制命令
3) START_SLEEP 进入自动止鼾模式（启动压力监测，根据姿态自动调节）
4) STOP_SLEEP 停止自动模式，统计睡眠数据并通过连接回传 App

协议（每行一条，回复也按行）：
  START_SLEEP / STOP_SLEEP / ALARM:HH:MM
  PILLOW_UP:L1|L2|R2|R1  /  PILLOW_DOWN:L1|L2|R2|R1  /  PILLOW_RESET
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import serial
import serial.tools.list_ports

# 蓝牙 RFCOMM：优先用标准库；Python < 3.11 在 Linux 上用内核常量
if getattr(socket, "AF_BLUETOOTH", None) is not None:
    _BT_AF = socket.AF_BLUETOOTH
    _BT_PROTO = socket.BTPROTO_RFCOMM
else:
    _BT_AF = 32
    _BT_PROTO = 6

LOG = logging.getLogger("pillow-controller")

# 区域 -> 气囊编号（与 App/调控界面一致）
ZONE_TO_AIRBAG: Dict[str, int] = {
    "L1": 1,
    "L2": 2,
    "R2": 3,
    "R1": 4,
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class SleepStats:
    started_at: str = ""
    ended_at: str = ""
    duration_sec: int = 0
    snore_events: int = 0
    auto_adjust_events: int = 0
    manual_adjust_events: int = 0
    reset_count: int = 0
    command_count: int = 0
    alarms: List[str] = field(default_factory=list)
    # 数据统计（由 realtime 脚本写入文件，STOP_SLEEP 时合并）
    total_sleep_sec: float = 0.0
    snoring_total_sec: float = 0.0
    snoring_posture_ratio: Dict[str, float] = field(default_factory=dict)
    sleep_score: int = 0

    def to_report_line(self) -> str:
        payload = {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_sec": self.duration_sec,
            "snore_events": self.snore_events,
            "auto_adjust_events": self.auto_adjust_events,
            "manual_adjust_events": self.manual_adjust_events,
            "reset_count": self.reset_count,
            "command_count": self.command_count,
            "alarms": self.alarms,
            "total_sleep_sec": round(self.total_sleep_sec, 1),
            "snoring_total_sec": round(self.snoring_total_sec, 1),
            "snoring_posture_ratio": self.snoring_posture_ratio,
            "sleep_score": self.sleep_score,
        }
        return "SLEEP_REPORT:" + json.dumps(payload, ensure_ascii=False)


class ArduinoBridge:
    """通过串口与 Arduino 气囊控制器通信。"""

    KEYWORDS = ("arduino", "ch340", "ttyusb", "ttyacm", "usb serial")

    def __init__(self, baudrate: int = 9600):
        self.baudrate = baudrate
        self._ser: Optional[serial.Serial] = None
        self._port: Optional[str] = None
        self._lock = threading.Lock()
        self._reader_stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def current_port(self) -> Optional[str]:
        return self._port

    def connect_auto(self) -> bool:
        if self._ser and self._ser.is_open:
            return True
        port = self._detect_port()
        if not port:
            LOG.error("未检测到 Arduino 串口")
            return False
        return self.connect(port)

    def _read_serial_to_stdout(self) -> None:
        """后台线程：将 Arduino 串口输出转发到终端，便于查看下游状态。"""
        buffer = ""
        while not self._reader_stop.is_set():
            ser = self._ser
            if not ser or not ser.is_open:
                break
            try:
                raw = ser.readline()
                if not raw:
                    continue
                buffer += raw.decode("utf-8", errors="replace")
                while "\n" in buffer or "\r" in buffer:
                    sep = "\n" if "\n" in buffer else "\r"
                    i = buffer.index(sep)
                    line = buffer[:i].strip()
                    buffer = buffer[i + 1 :].lstrip("\r\n")
                    if line:
                        print("[Arduino]", line, flush=True)
            except (OSError, serial.SerialException, TypeError, AttributeError):
                # 关闭串口时 fd 可能已为 None，readline 会抛 TypeError/AttributeError，属正常退出
                break
            except Exception:
                LOG.exception("Arduino 串口读取异常")

    def connect(self, port: str) -> bool:
        try:
            self._reader_stop.clear()
            self._ser = serial.Serial(port, self.baudrate, timeout=0.2)
            self._port = port
            # 打开串口会触发 Arduino 复位，需等其 boot + setup() 完成后再发指令
            time.sleep(2.0)
            self._wait_arduino_ready(timeout=4.0)
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._reader_thread = threading.Thread(
                target=self._read_serial_to_stdout, name="ArduinoReader", daemon=True
            )
            self._reader_thread.start()
            LOG.info("Arduino 已连接: %s", port)
            return True
        except serial.SerialException as exc:
            LOG.error("Arduino 连接失败: %s", exc)
            self._ser = None
            self._port = None
            return False

    def _wait_arduino_ready(self, timeout: float = 4.0) -> None:
        """等 Arduino 打印就绪信息（如 'Airbag Controller Ready'）后再返回，避免首条指令在初始化时丢失。"""
        if not self._ser or not self._ser.is_open:
            return
        deadline = time.monotonic() + timeout
        buf = ""
        while time.monotonic() < deadline:
            raw = self._ser.readline()
            if not raw:
                time.sleep(0.05)
                continue
            buf += raw.decode("utf-8", errors="replace")
            if "Ready" in buf or "ready" in buf:
                LOG.debug("Arduino 已就绪")
                return
            if len(buf) > 512:
                buf = buf[-256:]
        LOG.debug("Arduino 就绪等待超时，继续")

    def disconnect(self) -> None:
        self._reader_stop.set()
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self._ser = None
            self._port = None
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None

    def send_wire(self, cmd: str) -> None:
        """发送 3 字节命令，如 P11、V21、S00。"""
        if len(cmd) != 3:
            raise ValueError(f"串口命令长度必须为 3，实际: {cmd!r}")
        with self._lock:
            if not self._ser or not self._ser.is_open:
                raise RuntimeError("Arduino 串口未连接")
            self._ser.write(cmd.encode("utf-8"))
            LOG.info("-> Arduino: %s (已发送)", cmd)

    def pillow_adjust(self, action: str, zone: str) -> str:
        """PILLOW_UP -> 充气 5s (P?1)，PILLOW_DOWN -> 放气 10s (V?1)。"""
        airbag = ZONE_TO_AIRBAG.get(zone.upper())
        if not airbag:
            raise ValueError(f"未知区域: {zone}")
        if action == "PILLOW_UP":
            cmd = f"P{airbag}1"
        else:
            cmd = f"V{airbag}1"
        self.send_wire(cmd)
        return cmd

    def stop_all(self) -> None:
        """停止所有气囊操作。Arduino 端需能识别 S 并忽略后两字节。"""
        self.send_wire("S00")

    def reset_30s_async(self) -> None:
        """PILLOW_RESET：全部放气 30s，两轮 15s (V?2)。"""
        for i in (1, 2, 3, 4):
            self.send_wire(f"V{i}2")

        def second_round() -> None:
            try:
                for i in (1, 2, 3, 4):
                    self.send_wire(f"V{i}2")
            except Exception:
                LOG.exception("PILLOW_RESET 第二轮发送失败")

        threading.Timer(15.0, second_round).start()

    def _detect_port(self) -> Optional[str]:
        """优先选 Arduino (ttyACM/arduino)，避免误选压力传感器 (ttyUSB/CH340)。"""
        ports = list(serial.tools.list_ports.comports())
        # 第一轮：优先 Arduino 特征 (ttyACM、arduino)
        for p in ports:
            text = f"{p.device} {p.description}".lower()
            if "arduino" in text or "ttyacm" in text or "acm" in text:
                return p.device
        # 第二轮：其它串口 (ch340、ttyusb 等)
        for p in ports:
            text = f"{p.device} {p.description}".lower()
            if any(k in text for k in self.KEYWORDS):
                return p.device
        return ports[0].device if ports else None


class SleepModeManager:
    """睡眠模式：启动「实时打鼾与睡姿监控」脚本，融合压力+语音并自动止鼾。"""

    def __init__(self, project_root: Path, arduino: ArduinoBridge):
        self.project_root = project_root
        self.arduino = arduino
        self.stats = SleepStats()
        self.running = False
        self._lock = threading.Lock()
        self._control_proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self.running = True
            self.stats = SleepStats(started_at=now_str())
            self._start_snore_posture_control()
            LOG.info("睡眠模式已启动（打鼾+睡姿融合监控）")

    def stop(self) -> SleepStats:
        with self._lock:
            if not self.running:
                return self.stats
            self.running = False

        self._stop_snore_posture_control()
        self.stats.ended_at = now_str()
        if self.stats.started_at:
            try:
                start = datetime.strptime(self.stats.started_at, "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(self.stats.ended_at, "%Y-%m-%d %H:%M:%S")
                self.stats.duration_sec = int((end - start).total_seconds())
            except ValueError:
                pass
        # 合并 realtime 脚本写入的睡眠统计（总时长、打鼾时长、打鼾时睡姿占比、睡眠评分）
        stats_file = self.project_root / "sleep_session_stats.json"
        if stats_file.exists():
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
                self.stats.total_sleep_sec = float(data.get("total_sleep_sec", 0))
                self.stats.snoring_total_sec = float(data.get("snoring_total_sec", 0))
                self.stats.snoring_posture_ratio = dict(data.get("snoring_posture_ratio", {}))
                self.stats.sleep_score = int(data.get("sleep_score", 0))
            except Exception as e:
                LOG.warning("读取睡眠统计文件失败: %s", e)
        LOG.info("睡眠模式已停止")
        return self.stats

    def record_alarm(self, alarm: str) -> None:
        self.stats.alarms.append(alarm)

    def _start_snore_posture_control(self) -> None:
        """启动 realtime_snore_posture_control.py（压力+语音融合，自动止鼾）。"""
        control_script = self.project_root / "realtime_snore_posture_control.py"
        if not control_script.exists():
            LOG.warning("打鼾睡姿监控脚本不存在: %s", control_script)
            return
        if not self.arduino.connect_auto():
            LOG.error("无法连接 Arduino，无法启动监控")
            return
        arduino_port = self.arduino.current_port
        self.arduino.disconnect()
        pressure_port = self._detect_pressure_port(exclude=arduino_port) or "/dev/ttyUSB0"
        if pressure_port == "/dev/ttyUSB0" and not self._detect_pressure_port(exclude=arduino_port):
            LOG.warning("未检测到压力传感器串口，使用默认 %s", pressure_port)
        cmd = [
            "python3",
            str(control_script),
            "--arduino-port", arduino_port or "",
            "--pressure-port", pressure_port,
            "--project-root", str(self.project_root),
        ]
        self._control_proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
        )
        LOG.info("打鼾睡姿监控已启动 pid=%s，日志输出到当前终端", self._control_proc.pid)

    def _stop_snore_posture_control(self) -> None:
        if not self._control_proc:
            return
        if self._control_proc.poll() is None:
            self._control_proc.terminate()
            try:
                self._control_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._control_proc.kill()
        self._control_proc = None

    def _detect_pressure_port(self, exclude: Optional[str]) -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            t = f"{p.device} {p.description}".lower()
            if "ttyusb" in t or "ch340" in t or "ch341" in t:
                if p.device != exclude:
                    return p.device
        return next((p.device for p in ports if p.device != exclude), None)


class SmartPillowController:
    """解析上位机指令并调度 Arduino / 睡眠模式。"""

    def __init__(self, project_root: Path, arduino_baud: int):
        self.arduino = ArduinoBridge(baudrate=arduino_baud)
        self.sleep_mode = SleepModeManager(project_root, self.arduino)

    def handle_command(self, command: str) -> List[str]:
        command = command.strip()
        if not command:
            return ["IGNORED:EMPTY"]

        LOG.info("<- APP: %s", command)
        self.sleep_mode.stats.command_count += 1

        if command == "START_SLEEP":
            if not self.arduino.connect_auto():
                return ["ERR:ARDUINO_CONNECT_FAILED"]
            self.sleep_mode.start()
            return ["OK:START_SLEEP"]

        if command.startswith("ALARM:"):
            _, alarm = command.split(":", 1)
            try:
                datetime.strptime(alarm.strip(), "%H:%M")
            except ValueError:
                return ["ERR:BAD_ALARM_FORMAT"]
            self.sleep_mode.record_alarm(alarm.strip())
            return [f"OK:ALARM:{alarm.strip()}"]

        if command == "STOP_SLEEP":
            try:
                self.sleep_mode.stop()
                self.arduino.stop_all()
            except Exception as exc:
                return [f"ERR:STOP_SLEEP_FAILED:{exc}"]
            return ["OK:STOP_SLEEP", self.sleep_mode.stats.to_report_line()]

        if command == "PILLOW_RESET":
            if not self.arduino.connect_auto():
                return ["ERR:ARDUINO_CONNECT_FAILED"]
            self.arduino.reset_30s_async()
            self.sleep_mode.stats.reset_count += 1
            return ["OK:PILLOW_RESET"]

        if command.startswith("PILLOW_UP:") or command.startswith("PILLOW_DOWN:"):
            if not self.arduino.connect_auto():
                return ["ERR:ARDUINO_CONNECT_FAILED"]
            action, zone = command.split(":", 1)
            zone = zone.strip().upper()
            try:
                self.arduino.pillow_adjust(action.strip(), zone)
            except Exception as exc:
                return [f"ERR:ADJUST_FAILED:{exc}"]
            self.sleep_mode.stats.manual_adjust_events += 1
            return [f"OK:{action.strip()}:{zone}"]

        return ["ERR:UNKNOWN_COMMAND"]

    def shutdown(self) -> None:
        try:
            self.sleep_mode.stop()
        except Exception:
            LOG.exception("停止睡眠模式失败")
        try:
            self.arduino.stop_all()
        except Exception:
            pass
        self.arduino.disconnect()


class CommandServer:
    """蓝牙或 TCP 服务端，协议一致：按行收发。"""

    def __init__(
        self,
        channel: int,
        on_command: Callable[[str], List[str]],
        tcp_port: Optional[int] = None,
    ):
        self.channel = channel
        self.on_command = on_command
        self.tcp_port = tcp_port
        self._server: Optional[socket.socket] = None
        self._running = False

    def run_forever(self) -> None:
        self._running = True
        if self.tcp_port is not None:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind(("0.0.0.0", self.tcp_port))
            self._server.listen(1)
            LOG.info("TCP 监听 0.0.0.0:%d（协议与蓝牙一致）", self.tcp_port)
        else:
            try:
                self._server = socket.socket(_BT_AF, socket.SOCK_STREAM, _BT_PROTO)
                self._server.bind(("", self.channel))
                self._server.listen(1)
                LOG.info("蓝牙 RFCOMM 监听 channel %d", self.channel)
            except OSError as e:
                if e.errno == 97 or "not supported" in str(e).lower():
                    raise SystemExit(
                        "本机不支持蓝牙 socket。请改用 TCP: "
                        "python3 smart_pillow_controller.py --tcp-port 8765"
                    ) from e
                raise

        while self._running:
            client, addr = self._server.accept()
            LOG.info("客户端已连接: %s", addr)
            try:
                self._serve_client(client)
            finally:
                client.close()
                LOG.info("客户端已断开")

    def _serve_client(self, client: socket.socket) -> None:
        buffer = b""
        while self._running:
            data = client.recv(1024)
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                cmd = line.decode("utf-8", errors="ignore").strip()
                if not cmd:
                    continue
                responses = self.on_command(cmd)
                for resp in responses:
                    client.sendall((resp + "\n").encode("utf-8"))

    def stop(self) -> None:
        self._running = False
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="智能止鼾枕总控（蓝牙或 TCP）")
    parser.add_argument("--bt-channel", type=int, default=1, help="RFCOMM channel（使用蓝牙时）")
    parser.add_argument("--tcp-port", type=int, default=None, metavar="PORT",
                        help="改用 TCP 监听端口，如 8765")
    parser.add_argument("--arduino-baud", type=int, default=9600, help="Arduino 波特率")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent
    controller = SmartPillowController(project_root, arduino_baud=args.arduino_baud)
    server = CommandServer(
        args.bt_channel, controller.handle_command, tcp_port=args.tcp_port
    )

    def _shutdown(*args: object) -> None:
        LOG.info("正在退出…")
        server.stop()
        controller.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.run_forever()


if __name__ == "__main__":
    main()
