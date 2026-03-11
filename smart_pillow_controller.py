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

    def connect(self, port: str) -> bool:
        try:
            self._ser = serial.Serial(port, self.baudrate, timeout=0.2)
            self._port = port
            time.sleep(1.2)
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            LOG.info("Arduino 已连接: %s", port)
            return True
        except serial.SerialException as exc:
            LOG.error("Arduino 连接失败: %s", exc)
            self._ser = None
            self._port = None
            return False

    def disconnect(self) -> None:
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self._ser = None
            self._port = None

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
    """睡眠模式：启动压力监测子进程，根据姿态结果自动调节气囊。"""

    def __init__(self, project_root: Path, arduino: ArduinoBridge):
        self.project_root = project_root
        self.arduino = arduino
        self.stats = SleepStats()
        self.running = False
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pressure_proc: Optional[subprocess.Popen] = None
        self._pressure_reader: Optional[threading.Thread] = None
        self._last_auto_ts = 0.0
        self._cooldown_sec = 8.0

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self.running = True
            self._stop_event.clear()
            self.stats = SleepStats(started_at=now_str())
            self._start_pressure_process()
            self._worker = threading.Thread(target=self._auto_loop, daemon=True)
            self._worker.start()
            LOG.info("睡眠模式已启动")

    def stop(self) -> SleepStats:
        with self._lock:
            if not self.running:
                return self.stats
            self.running = False
            self._stop_event.set()

        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)

        self._stop_pressure_process()
        self.stats.ended_at = now_str()
        if self.stats.started_at:
            try:
                start = datetime.strptime(self.stats.started_at, "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(self.stats.ended_at, "%Y-%m-%d %H:%M:%S")
                self.stats.duration_sec = int((end - start).total_seconds())
            except ValueError:
                pass
        LOG.info("睡眠模式已停止")
        return self.stats

    def record_alarm(self, alarm: str) -> None:
        self.stats.alarms.append(alarm)

    def _auto_loop(self) -> None:
        """占位循环，实际触发由 _pressure_reader_loop 解析压力输出完成。"""
        while not self._stop_event.is_set():
            time.sleep(1.0)

    def _start_pressure_process(self) -> None:
        script = self.project_root / "Sensor" / "Pressure" / "realtime_pressure_inference.py"
        if not script.exists():
            LOG.warning("压力监测脚本不存在: %s", script)
            return

        port = self._detect_pressure_port(exclude=self.arduino.current_port)
        if not port:
            LOG.warning("未检测到压力传感器串口，跳过压力监测")
            return

        cmd = ["python3", str(script), "--port", port]
        self._pressure_proc = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        LOG.info("压力监测已启动: %s (pid=%s)", port, self._pressure_proc.pid)
        self._pressure_reader = threading.Thread(target=self._pressure_reader_loop, daemon=True)
        self._pressure_reader.start()

    def _stop_pressure_process(self) -> None:
        if not self._pressure_proc:
            return
        if self._pressure_proc.poll() is None:
            self._pressure_proc.terminate()
            try:
                self._pressure_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._pressure_proc.kill()
        self._pressure_proc = None
        self._pressure_reader = None

    def _detect_pressure_port(self, exclude: Optional[str]) -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())
        devices = [p.device for p in ports if p.device != exclude]
        return devices[0] if devices else None

    def _pressure_reader_loop(self) -> None:
        proc = self._pressure_proc
        if not proc or not proc.stdout:
            return

        for line in proc.stdout:
            if self._stop_event.is_set():
                break
            text = line.strip()
            if "[Result]" not in text:
                continue

            target_zone: Optional[str] = None
            if "Left" in text:
                target_zone = "R2"
            elif "Right" in text:
                target_zone = "L2"

            if not target_zone:
                continue

            now = time.time()
            if now - self._last_auto_ts < self._cooldown_sec:
                continue

            try:
                self.arduino.pillow_adjust("PILLOW_UP", target_zone)
                self.stats.snore_events += 1
                self.stats.auto_adjust_events += 1
                self._last_auto_ts = now
                LOG.info("自动调节 (%s) -> %s", text[:40], target_zone)
            except Exception:
                LOG.exception("自动调节发送失败")


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
