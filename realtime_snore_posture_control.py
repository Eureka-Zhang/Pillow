#!/usr/bin/env python3
"""
实时打鼾与睡姿监控 + 自动止鼾调控

- 融合压力睡姿与声音打鼾/睡姿（压力置信度更高）
- 检测到打鼾：先全放气 30s；若仍未止鼾则按融合睡姿执行对应充气序列
- 由总控在 START_SLEEP 时启动，STOP_SLEEP 时结束

用法（总控内部调用）:
  python3 realtime_snore_posture_control.py --arduino-port /dev/ttyACM0 --pressure-port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import serial
import serial.tools.list_ports

LOG = logging.getLogger("snore-posture")

# 睡姿类别与索引（与压力/语音一致）
POSTURE_NAMES = ["Middle", "Left-Back", "Right-Back", "Left", "Right"]
# 语音简写 -> 索引
VOICE_POSTURE_MAP = {
    "仰卧": 0,
    "仰卧头偏左": 1,
    "仰卧头偏右": 2,
    "左侧卧": 3,
    "右侧卧": 4,
}

# 融合权重：压力置信度更高
WEIGHT_PRESSURE = 0.8
WEIGHT_VOICE = 0.2

# 状态
STATE_IDLE = "idle"
STATE_DEFLATE_30S = "deflate_30s"
STATE_COOLDOWN = "cooldown"

DEFLATE_DURATION = 30.0
WAIT_AFTER_DEFLATE = 32.0  # 放气 30s 后再判定
COOLDOWN_SEC = 60.0
SNORE_CONFIRM_SEC = 2.0    # 连续打鼾判定时长


@dataclass
class PressureResult:
    posture_idx: int
    posture_name: str
    probs: List[float]
    ts: float = field(default_factory=time.time)


@dataclass
class VoiceResult:
    is_snoring: bool
    posture_idx: int
    snore_conf: float
    ts: float = field(default_factory=time.time)


def parse_pressure_line(line: str) -> Optional[PressureResult]:
    """解析压力推理输出: [Result] Middle     (Conf: 0.85) | Probs: ['0.85', ...]"""
    if "[Result]" not in line:
        return None
    try:
        name_part = re.search(r"\[Result\]\s+(\S+)\s+\(Conf:\s*([\d.]+)\)", line)
        probs_part = re.search(r"Probs:\s*\[([^\]]+)\]", line)
        if not name_part or not probs_part:
            return None
        name = name_part.group(1).strip()
        probs_str = probs_part.group(1).replace("'", "").split(",")
        probs = [float(p.strip()) for p in probs_str if p.strip()]
        if len(probs) != 5:
            return None
        idx = POSTURE_NAMES.index(name) if name in POSTURE_NAMES else 0
        return PressureResult(posture_idx=idx, posture_name=name, probs=probs)
    except Exception:
        return None


def parse_voice_line(line: str) -> Optional[VoiceResult]:
    """解析语音推理：打鼾/正常/静音 + 睡姿"""
    if "正常" not in line and "打鼾" not in line and "静音" not in line:
        return None
    is_snoring = "打鼾" in line
    posture_idx = 0
    for short, idx in VOICE_POSTURE_MAP.items():
        if short in line:
            posture_idx = idx
            break
    conf = 0.5
    m = re.search(r"([\d.]+)\)", line)
    if m:
        conf = float(m.group(1))
    return VoiceResult(is_snoring=is_snoring, posture_idx=posture_idx, snore_conf=conf)


class ArduinoSender:
    """向 Arduino 发送 P/V 命令，与总控协议一致；后台线程将 Arduino 回复打印到终端。"""

    def __init__(self, port: str, baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._reader_stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

    def _read_serial_to_stdout(self) -> None:
        """后台线程：将 Arduino 串口输出转发到终端。"""
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

    def _wait_arduino_ready(self, timeout: float = 4.0) -> None:
        """等 Arduino 打印就绪信息后再返回，避免首条指令在初始化时丢失。"""
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

    def connect(self) -> bool:
        try:
            self._reader_stop.clear()
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0.2)
            time.sleep(2.0)
            self._wait_arduino_ready(timeout=4.0)
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._reader_thread = threading.Thread(
                target=self._read_serial_to_stdout, name="ArduinoReader", daemon=True
            )
            self._reader_thread.start()
            LOG.info("Arduino 已连接: %s", self.port)
            return True
        except Exception as e:
            LOG.error("Arduino 连接失败: %s", e)
            return False

    def send(self, cmd: str) -> None:
        """发送 3 字节命令，如 P22, V12."""
        if len(cmd) != 3:
            raise ValueError(f"命令须 3 字节: {cmd!r}")
        with self._lock:
            if not self._ser or not self._ser.is_open:
                raise RuntimeError("Arduino 未连接")
            self._ser.write(cmd.encode("utf-8"))
            LOG.info("-> Arduino: %s", cmd)

    def deflate_30s(self) -> None:
        """全放气 30s：两轮 15s."""
        for i in (1, 2, 3, 4):
            self.send(f"V{i}2")

        def second_round() -> None:
            try:
                for i in (1, 2, 3, 4):
                    self.send(f"V{i}2")
            except Exception:
                LOG.exception("第二轮放气失败")
        threading.Timer(15.0, second_round).start()

    def stop_all(self) -> None:
        self.send("S00")

    def close(self) -> None:
        self._reader_stop.set()
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self._ser = None
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None


def run_posture_adjustment(arduino: ArduinoSender, posture_idx: int) -> None:
    """
    按睡姿执行充气。Arduino 已支持多气囊独立计时：
    - 不同气囊可同时充气（连续发 P12、P22 即 1 与 2 同时 10s）
    - 同一气囊重复命令仅重置该气囊计时（非阻塞，由 App 端调节时体现）
    """
    if posture_idx == 0:  # Middle：2号10s
        arduino.send("P22")
    elif posture_idx == 1:  # Left-Back：2号5s，再 3号10s（顺序执行）
        arduino.send("P21")
        arduino.send("P32")
    elif posture_idx == 2:  # Right-Back：3号5s，再 2号10s
        arduino.send("P31")
        arduino.send("P22")
    elif posture_idx == 3:  # Left：1号与2号同时充气 10s
        arduino.send("P12")
        arduino.send("P22")
    elif posture_idx == 4:  # Right：3号与4号同时充气 10s
        arduino.send("P32")
        arduino.send("P42")
    else:
        arduino.send("P22")


def reader_thread(
    proc: subprocess.Popen,
    result_queue: deque,
    parse_fn,
    name: str,
    stop_event: threading.Event,
) -> None:
    if not proc or not proc.stdout:
        return
    for line in iter(proc.stdout.readline, ""):
        if stop_event.is_set():
            break
        line = line.strip()
        if not line:
            continue
        r = parse_fn(line)
        if r is not None:
            result_queue.append(r)
        else:
            # 子进程有输出但解析失败时打 DEBUG，便于排查「一直没有数据更新」
            LOG.debug("[%s] 未解析: %s", name, line[:80])


def main() -> int:
    parser = argparse.ArgumentParser(description="实时打鼾与睡姿监控 + 自动止鼾")
    parser.add_argument("--arduino-port", type=str, required=True, help="Arduino 串口")
    parser.add_argument("--pressure-port", type=str, required=True, help="压力传感器串口")
    parser.add_argument("--project-root", type=str, default=None, help="项目根目录，默认当前目录")
    parser.add_argument("--voice-model", type=str, default=None, help="语音模型路径，默认自动")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    project_root = Path(args.project_root or ".").resolve()
    pressure_script = project_root / "Sensor" / "Pressure" / "realtime_pressure_inference.py"
    voice_script = project_root / "Sensor" / "Voice" / "realtime_voice_inference.py"
    voice_model = args.voice_model
    if not voice_model:
        for p in [project_root / "Sensor" / "Voice" / "voice_model.rknn", project_root / "Sensor" / "Voice" / "mlt_best.pth"]:
            if p.exists():
                voice_model = str(p)
                break
    if not pressure_script.exists():
        LOG.error("压力脚本不存在: %s", pressure_script)
        return 1
    if not voice_script.exists():
        LOG.error("语音脚本不存在: %s", voice_script)
        return 1

    arduino = ArduinoSender(args.arduino_port)
    if not arduino.connect():
        return 1

    pressure_proc = subprocess.Popen(
        [sys.executable, str(pressure_script), "--port", args.pressure_port],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    voice_cmd = [sys.executable, str(voice_script)]
    if voice_model:
        voice_cmd.extend(["--model_path", voice_model])
    if Path(voice_model or "").suffix == ".rknn":
        voice_cmd.extend(["--backend", "rknn"])
    voice_proc = subprocess.Popen(
        voice_cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    pressure_queue: deque = deque(maxlen=20)
    voice_queue: deque = deque(maxlen=20)
    stop_event = threading.Event()

    t_p = threading.Thread(
        target=reader_thread,
        args=(pressure_proc, pressure_queue, parse_pressure_line, "pressure", stop_event),
        daemon=True,
    )
    t_v = threading.Thread(
        target=reader_thread,
        args=(voice_proc, voice_queue, parse_voice_line, "voice", stop_event),
        daemon=True,
    )
    t_p.start()
    t_v.start()

    latest_pressure: Optional[PressureResult] = None
    latest_voice: Optional[VoiceResult] = None
    fused_probs = [0.2] * 5
    fused_posture_idx = 0
    state = STATE_IDLE
    state_entered_at = 0.0
    snoring_since: Optional[float] = None
    last_adjust_ts = 0.0

    def shutdown(*args: object) -> None:
        stop_event.set()
        # 写入睡眠统计供总控回传 App
        try:
            session_end = time.time()
            total_sleep_sec = max(0.0, session_end - session_start)
            snoring_ratio = snoring_total_sec / max(1.0, total_sleep_sec)
            posture_ratio = {
                name: (snoring_posture_secs[name] / snoring_total_sec if snoring_total_sec > 0 else 0.0)
                for name in POSTURE_NAMES
            }
            # 评分：0% 打鼾=100 分，100% 打鼾=40 分，线性
            sleep_score = max(0, min(100, round(100 - snoring_ratio * 60)))
            stats_path = project_root / "sleep_session_stats.json"
            payload = {
                "started_at": datetime.fromtimestamp(session_start).strftime("%Y-%m-%d %H:%M:%S"),
                "ended_at": datetime.fromtimestamp(session_end).strftime("%Y-%m-%d %H:%M:%S"),
                "total_sleep_sec": round(total_sleep_sec, 1),
                "snoring_total_sec": round(snoring_total_sec, 1),
                "snoring_posture_ratio": {k: round(v, 4) for k, v in posture_ratio.items()},
                "sleep_score": sleep_score,
            }
            stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            LOG.info("睡眠统计已写入 %s", stats_path)
        except Exception as e:
            LOG.exception("写入睡眠统计失败: %s", e)
        voice_proc.terminate()
        pressure_proc.terminate()
        try:
            voice_proc.wait(timeout=2)
            pressure_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            voice_proc.kill()
            pressure_proc.kill()
        arduino.stop_all()
        arduino.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    LOG.info("实时打鼾与睡姿监控已启动，融合策略: 压力 %.1f + 语音 %.1f", WEIGHT_PRESSURE, WEIGHT_VOICE)

    # 数据统计：睡眠总时长、打鼾时长、打鼾时睡姿占比、睡眠评分，结束时写入文件供总控回传 App
    session_start = time.time()
    snoring_total_sec = 0.0
    snoring_posture_secs: Dict[str, float] = {name: 0.0 for name in POSTURE_NAMES}
    LOOP_INTERVAL = 0.25

    had_new_data = False
    last_judgment_print = 0.0
    JUDGMENT_PRINT_INTERVAL = 10.0  # 无新数据时每 10 秒仍打印一次，便于确认是否有数据更新
    while not stop_event.is_set():
        while pressure_queue:
            latest_pressure = pressure_queue.popleft()
            had_new_data = True
        while voice_queue:
            latest_voice = voice_queue.popleft()
            had_new_data = True

        if latest_pressure is not None:
            p = latest_pressure.probs
            if latest_voice is not None:
                v_probs = [0.0] * 5
                v_probs[latest_voice.posture_idx] = 1.0
                fused_probs = [WEIGHT_PRESSURE * p[i] + WEIGHT_VOICE * v_probs[i] for i in range(5)]
            else:
                fused_probs = p[:]
            fused_posture_idx = int(max(range(5), key=lambda i: fused_probs[i]))

        if latest_voice is not None:
            if latest_voice.is_snoring:
                if snoring_since is None:
                    snoring_since = time.time()
            else:
                snoring_since = None

        now = time.time()

        # 统计：打鼾时段累加时长及该时段睡姿占比
        if snoring_since is not None:
            snoring_total_sec += LOOP_INTERVAL
            snoring_posture_secs[POSTURE_NAMES[fused_posture_idx]] += LOOP_INTERVAL

        should_print = had_new_data or (now - last_judgment_print >= JUDGMENT_PRINT_INTERVAL)

        # 实时判断打到终端：有新数据时打印；无新数据时每 10 秒也打印一次并带数据年龄，便于判断是否一直无更新
        if should_print:
            def _age(ts: float) -> str:
                s = now - ts
                if s < 60:
                    return "%.0fs前" % s
                return "%.1fmin前" % (s / 60)

            pressure_str = "-"
            if latest_pressure is not None:
                conf = latest_pressure.probs[latest_pressure.posture_idx]
                pressure_str = "%s(%.2f) %s" % (latest_pressure.posture_name, conf, _age(latest_pressure.ts))
            voice_str = "-"
            if latest_voice is not None:
                voice_str = "%s/%s %s" % ("打鼾" if latest_voice.is_snoring else "正常", POSTURE_NAMES[latest_voice.posture_idx], _age(latest_voice.ts))
            fuse_name = POSTURE_NAMES[fused_posture_idx]
            snore_str = "打鼾中" if snoring_since is not None else "静音"
            print("[判断] 压力=%s 语音=%s 融合=%s %s 状态=%s" % (pressure_str, voice_str, fuse_name, snore_str, state), flush=True)
            last_judgment_print = now
        had_new_data = False

        if state == STATE_IDLE:
            if snoring_since is not None and (now - snoring_since) >= SNORE_CONFIRM_SEC:
                msg = "检测到打鼾，进入 30s 全放气"
                LOG.info(msg)
                print("[判断] %s" % msg, flush=True)
                arduino.deflate_30s()
                state = STATE_DEFLATE_30S
                state_entered_at = now

        elif state == STATE_DEFLATE_30S:
            if (now - state_entered_at) >= WAIT_AFTER_DEFLATE:
                if snoring_since is not None and (now - snoring_since) >= SNORE_CONFIRM_SEC:
                    msg = "放气后仍打鼾，按睡姿调节: %s" % POSTURE_NAMES[fused_posture_idx]
                    LOG.info(msg)
                    print("[判断] %s" % msg, flush=True)
                    run_posture_adjustment(arduino, fused_posture_idx)
                    state = STATE_COOLDOWN
                    state_entered_at = now
                else:
                    state = STATE_IDLE
                    snoring_since = None

        elif state == STATE_COOLDOWN:
            if (now - state_entered_at) >= COOLDOWN_SEC:
                state = STATE_IDLE
                snoring_since = None

        time.sleep(LOOP_INTERVAL)

    shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
