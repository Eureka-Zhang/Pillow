# -*- coding: utf-8 -*-
"""
Real-time Snore Detection for Orange Pi 5 Plus (RK3588)
从话筒实时采集音频，使用滑动窗口进行打鼾检测和睡姿识别

支持推理后端:
- PyTorch CPU (默认)
- RKNN NPU (RK3588 加速，需安装 rknn-toolkit-lite2)

Mel频谱处理流程 (与训练时一致):
1. 重采样到 16kHz
2. 截取/补零到 3 秒
3. 高频预加重 (preemphasis)
4. 计算 Mel 频谱图 (n_mels=80, n_fft=400, hop_length=160, hamming窗)
5. 转换为对数刻度 (power_to_db)
6. 标准化 (mean=0, std=1)
7. 推理前再次标准化 (与 train_MLT.py 中 __getitem__ 一致)

Usage:
    # PyTorch CPU 推理
    python3 realtime_voice_inference.py --model_path ./mlt_best.pth
    
    # RKNN NPU 推理 (推荐，速度更快)
     python3 Sensor/Voice/realtime_voice_inference.py --model_path Sensor/Voice/voice_model.rknn --backend rknn
    
    # 列出音频设备
    python3 realtime_voice_inference.py --list_devices
    
    # 测试麦克风
    python3 realtime_inference.py --test_audio
"""

import os
import sys
import time
import argparse
import threading
import queue
from datetime import datetime

import numpy as np
import librosa
import sounddevice as sd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================
# 配置参数 (与 MEL_preprocess.py 的 "Mel" 配置完全一致)
# =====================
class Config:
    # Audio parameters
    SAMPLE_RATE = 16000          # 采样率 16kHz
    WINDOW_DURATION = 3.0        # 窗口长度 3秒
    SLIDE_INTERVAL = 1.0         # 滑动间隔 1秒
    
    # Mel spectrogram parameters (与 MEL_preprocess.py "Mel" preset 一致)
    N_MELS = 80                  # Mel频带数
    N_FFT = 400                  # FFT窗口 (25ms @ 16kHz)
    HOP_LENGTH = 160             # 跳帧长度 (10ms @ 16kHz)
    WIN_LENGTH = 400             # 窗长 = n_fft
    WINDOW = "hamming"           # 窗函数
    POWER = 2.0                  # 功率谱
    
    # 预处理选项
    PREEMPHASIS = True           # 高频预加重
    NORMALIZE = True             # 标准化 (第一次，在mel计算后)
    
    # Model parameters
    SNORE_CLASSES = 2
    POSTURE_CLASSES = 5          # 5类睡姿 (训练时排除了第6类)
    
    # Detection thresholds
    SNORE_THRESHOLD = 0.5        # 打鼾判定阈值
    SILENCE_THRESHOLD = 0.008     # 静音门限 (低于此值视为底噪，不推理)


# =====================
# Mel频谱提取 (与 MEL_preprocess.py 的 preprocess_audio_mel 完全一致)
# =====================
def extract_mel_spectrogram(audio, config=Config):
    """
    将音频转换为Mel频谱图
    完全复刻 MEL_preprocess.py 中的处理流程
    
    Args:
        audio: np.ndarray, 音频数据 (已经是 config.SAMPLE_RATE 采样率)
        config: 配置对象
    
    Returns:
        log_mel: np.ndarray, Mel频谱图 [n_mels, time_frames]
    """
    y = audio.astype(np.float32)
    
    # Step 1: 截取或补零到目标长度
    target_len = int(config.SAMPLE_RATE * config.WINDOW_DURATION)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    
    # Step 2: 高频预加重 (与 MEL_preprocess.py 一致)
    if config.PREEMPHASIS:
        y = librosa.effects.preemphasis(y)
    
    # Step 3: 计算Mel频谱图 (参数与 MEL_preprocess.py 完全一致)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=config.WINDOW,
        power=config.POWER
    )
    
    # Step 4: 转换为对数刻度 (与 MEL_preprocess.py 一致)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Step 5: 第一次标准化 (与 MEL_preprocess.py 一致)
    if config.NORMALIZE:
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
    
    return log_mel


def normalize_for_inference(feat):
    """
    推理前的标准化 (与 train_MLT.py 中 __getitem__ 一致)
    训练时加载 .npy 文件后会再做一次标准化
    
    Args:
        feat: np.ndarray, Mel频谱特征
    
    Returns:
        normalized_feat: np.ndarray
    """
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    return feat


# =====================
# 推理后端基类
# =====================
class InferenceBackend:
    """推理后端基类"""
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
    
    def predict(self, mel_input):
        """
        执行推理
        Args:
            mel_input: np.ndarray, shape [1, 1, n_mels, time_frames]
        Returns:
            snore_logits: np.ndarray, shape [1, 2]
            posture_logits: np.ndarray, shape [1, 5]
        """
        raise NotImplementedError
    
    def release(self):
        """释放资源"""
        pass


# =====================
# PyTorch CPU 后端
# =====================
class PyTorchBackend(InferenceBackend):
    """PyTorch CPU 推理后端"""
    
    def __init__(self, model_path, config):
        super().__init__(model_path, config)
        import torch
        from model_vo import CNN_TCN_MTL
        
        self.torch = torch
        self.device = torch.device("cpu")
        
        print(f"[PyTorch] Loading model from {model_path}...")
        self.model = CNN_TCN_MTL(
            snore_classes=config.SNORE_CLASSES,
            posture_classes=config.POSTURE_CLASSES
        )
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print("[PyTorch] Model loaded successfully (CPU)")
    
    def predict(self, mel_input):
        mel_tensor = self.torch.tensor(mel_input, dtype=self.torch.float32)
        mel_tensor = mel_tensor.to(self.device)
        
        with self.torch.no_grad():
            snore_logits, posture_logits = self.model(mel_tensor)
            return snore_logits.cpu().numpy(), posture_logits.cpu().numpy()


# =====================
# RKNN NPU 后端 (RK3588)
# =====================
class RKNNBackend(InferenceBackend):
    """RKNN NPU 推理后端 (RK3588 加速)"""
    
    def __init__(self, model_path, config):
        super().__init__(model_path, config)
        
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise ImportError(
                "RKNN-Toolkit-Lite2 未安装!\n"
                "请在 Orange Pi 上安装: pip3 install rknn-toolkit-lite2\n"
                "或从 https://github.com/rockchip-linux/rknn-toolkit2 获取"
            )
        
        print(f"[RKNN] Loading model from {model_path}...")
        self.rknn = RKNNLite()
        
        # 加载 RKNN 模型
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        
        # 初始化运行时环境
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)  # 使用全部3个NPU核心
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
        
        print("[RKNN] Model loaded successfully (NPU: RK3588)")
    
    def predict(self, mel_input):
        # RKNN 输入需要是 numpy array
        mel_input = mel_input.astype(np.float32)
        
        outputs = None
        # 尝试 1: 原始 4 维输入 (1, 1, 80, 301)
        try:
            outputs = self.rknn.inference(inputs=[mel_input])
        except Exception:
            outputs = None
            
        # 如果 4D 失败（抛异常 或 返回None），尝试 3D
        if outputs is None:
            # print(f"[WARN] 4D input failed, trying 3D input (1, 80, 301)...")
            try:
                mel_input_3d = np.squeeze(mel_input, axis=1) # 去掉 channel 维 -> (1, 80, 301)
                outputs = self.rknn.inference(inputs=[mel_input_3d])
            except Exception:
                outputs = None
        
        if outputs is None:
            # 彻底失败，返回默认值
            return np.array([[1.0, 0.0]]), np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])

        # outputs[0]: snore_logits [1, 2]
        # outputs[1]: posture_logits [1, 5]
        
        snore_logits = None
        posture_logits = None
        
        if len(outputs) == 2:
            # 自动检测输出顺序 (根据 shape)
            out0 = outputs[0]
            out1 = outputs[1]
            
            # snore: [1, 2], posture: [1, 5]
            if out0.shape[-1] == 2 and out1.shape[-1] == 5:
                snore_logits = out0
                posture_logits = out1
            elif out0.shape[-1] == 5 and out1.shape[-1] == 2:
                # print("[WARN] Model outputs swapped, auto-correcting...")
                posture_logits = out0
                snore_logits = out1
            else:
                # 无法判断或形状不对，按默认顺序
                snore_logits = out0
                posture_logits = out1
        elif len(outputs) == 1:
            # print(f"[WARN] Model returned 1 output instead of 2. Trying to infer...")
            out = outputs[0]
            if out.shape[-1] == 2:
                snore_logits = out
                posture_logits = np.zeros((1, 5)) # Dummy posture
                print("[WARN] Model only returned Snore Logits. Posture detection disabled.")
            elif out.shape[-1] == 5:
                posture_logits = out
                snore_logits = np.zeros((1, 2)) # Dummy snore
                snore_logits[0, 0] = 1.0 # Default to non-snore
                print("[WARN] Model only returned Posture Logits. Snore detection disabled.")
            else:
                print(f"[ERROR] Unknown output shape: {out.shape}")
                # Return dummy values if shape is unknown
                snore_logits = np.array([[1.0, 0.0]])
                posture_logits = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        else:
             print(f"[ERROR] Expected 1 or 2 outputs, got {len(outputs)}")
             snore_logits = np.array([[1.0, 0.0]])
             posture_logits = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        
        return snore_logits, posture_logits
    
    def release(self):
        if hasattr(self, 'rknn'):
            self.rknn.release()
            print("[RKNN] Resources released")


# =====================
# 实时推理器
# =====================
class RealtimeSnoreDetector:
    """实时打鼾检测器"""
    
    def __init__(self, model_path, backend="pytorch", config=Config):
        self.config = config
        self.backend_type = backend
        
        # 初始化推理后端
        if backend == "rknn":
            self.backend = RKNNBackend(model_path, config)
        else:
            self.backend = PyTorchBackend(model_path, config)
        
        # 音频缓冲区 (存储3秒的音频)
        self.buffer_size = int(config.SAMPLE_RATE * config.WINDOW_DURATION)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # 控制标志
        self.is_running = False
        self.stream = None
        
        # 结果队列
        self.result_queue = queue.Queue()
        
        # 睡姿标签 (与训练时的映射一致)
        self.posture_labels = {
            0: "仰卧 (Supine)",
            1: "仰卧头偏左 (Supine, left lateral head)",
            2: "仰卧头偏右 (Supine, right lateral head)",
            3: "左侧卧 (Left-side lying)",
            4: "右侧卧 (Right-side lying)"
        }
    
    def _audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            print(f"[WARN] Audio status: {status}")
        
        # 立体声转单声道 (取平均)
        if indata.ndim > 1 and indata.shape[1] >= 2:
            audio_data = np.mean(indata, axis=1).astype(np.float32)
        else:
            audio_data = indata.flatten().astype(np.float32)
        
        with self.buffer_lock:
            # 滑动缓冲区：移除旧数据，添加新数据
            shift_len = len(audio_data)
            self.audio_buffer[:-shift_len] = self.audio_buffer[shift_len:]
            self.audio_buffer[-shift_len:] = audio_data
    
    def _inference_loop(self):
        """推理循环"""
        interval = self.config.SLIDE_INTERVAL
        
        while self.is_running:
            start_time = time.time()
            
            # 获取当前缓冲区的音频
            with self.buffer_lock:
                audio = self.audio_buffer.copy()
            
            # 执行推理
            result = self._predict(audio)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            result["inference_time_ms"] = inference_time * 1000
            
            # 输出结果
            self._print_result(result)
            
            # 将结果放入队列
            self.result_queue.put(result)
            
            # 等待到下一个时间窗口
            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _predict(self, audio):
        """
        执行单次预测
        
        处理流程:
        1. extract_mel_spectrogram: 音频 -> Mel频谱 (包含预加重、标准化)
        2. normalize_for_inference: 推理前再次标准化 (与训练时一致)
        3. 模型推理
        """
        # 计算音频电平 (用于调试)
        audio_rms = np.sqrt(np.mean(audio ** 2))
        audio_peak = np.max(np.abs(audio))
        
        # 静音检测：底噪低于阈值时，直接返回"正常"，跳过模型推理
        if audio_rms < self.config.SILENCE_THRESHOLD:
            return {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "is_snoring": False,
                "snore_confidence": 0.0,
                "snore_probs": np.array([1.0, 0.0]),
                "posture_pred": 0,
                "posture_probs": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                "audio_rms": float(audio_rms),
                "audio_peak": float(audio_peak),
                "is_silent": True
            }
        
        # Step 1: 提取Mel频谱 (与 MEL_preprocess.py 一致)
        mel = extract_mel_spectrogram(audio, self.config)
        
        # Step 2: 推理前标准化 (与 train_MLT.py __getitem__ 一致)
        mel = normalize_for_inference(mel)
        
        # Step 3: 转换为模型输入格式 [1, 1, F, T]
        mel_input = mel[np.newaxis, np.newaxis, :, :]
        
        # Step 4: 模型推理
        snore_logits, posture_logits = self.backend.predict(mel_input)
        
        # 调试: 打印原始 logits (每10次打印一次)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        if self._debug_count % 10 == 1:
            print(f"\n[DEBUG] snore_logits: {snore_logits[0]}, posture_logits: {posture_logits[0][:3]}...")
        
        # Step 5: 计算概率 (softmax)
        snore_probs = self._softmax(snore_logits[0])
        posture_probs = self._softmax(posture_logits[0])
        
        # 打鼾判定：打鼾概率大于 SNORE_THRESHOLD(0.5) 才判为打鼾
        snore_confidence = float(snore_probs[1])
        is_snoring = snore_confidence > self.config.SNORE_THRESHOLD
        posture_pred = np.argmax(posture_probs)
        
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_snoring": is_snoring,
            "snore_confidence": snore_confidence,  # P(snoring)
            "snore_probs": snore_probs,                 # [P(non-snore), P(snore)]
            "posture_pred": int(posture_pred),
            "posture_probs": posture_probs,             # 5类睡姿概率
            "audio_rms": float(audio_rms),              # 音频 RMS 电平
            "audio_peak": float(audio_peak),            # 音频峰值
            "is_silent": False
        }
    
    @staticmethod
    def _softmax(x):
        """计算 softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def _print_result(self, result):
        """打印检测结果"""
        timestamp = result["timestamp"]
        is_snoring = result["is_snoring"]
        snore_probs = result["snore_probs"]
        posture_pred = result.get("posture_pred", 0)
        posture_probs = result.get("posture_probs", np.zeros(5))
        inference_ms = result.get("inference_time_ms", 0)
        audio_rms = result.get("audio_rms", 0)
        is_silent = result.get("is_silent", False)
        
        # 获取睡姿名称
        posture_name = self.posture_labels.get(posture_pred, "Unknown")
        # 简化的睡姿名称用于显示
        simple_posture_name = posture_name.split('(')[0].strip()
        
        # 音量条 (可视化)
        vol_bar_len = int(min(audio_rms * 100, 20))
        vol_bar = "█" * vol_bar_len + "░" * (20 - vol_bar_len)
        
        # 清空当前行并打印
        print("\r" + " " * 120, end="\r")
        
        if is_silent:
            # 静音/底噪状态
            print(f"[{timestamp}] ⚪ 静音 | 音量:{audio_rms:.4f} [{vol_bar}]")
        elif is_snoring:
            snore_conf = snore_probs[1]
            posture_conf = posture_probs[posture_pred]
            print(f"[{timestamp}] 🔴 打鼾 ({snore_conf:.2f}) | 睡姿: {simple_posture_name} ({posture_conf:.2f}) | 音量:{audio_rms:.4f} [{vol_bar}] | {inference_ms:.0f}ms")
        else:
            snore_conf = snore_probs[0] # 正常的置信度
            posture_conf = posture_probs[posture_pred]
            print(f"[{timestamp}] 🟢 正常 ({snore_conf:.2f}) |  音量:{audio_rms:.4f} [{vol_bar}] | {inference_ms:.0f}ms")
    
    def start(self, device_id=None):
        """开始实时检测"""
        if self.is_running:
            print("[WARN] Detector is already running")
            return
        
        self.is_running = True
        
        # 配置音频流参数 (es8388 使用双通道)
        stream_params = {
            "samplerate": self.config.SAMPLE_RATE,
            "channels": 2,  # 双通道 (es8388)
            "dtype": np.float32,
            "blocksize": int(self.config.SAMPLE_RATE * 0.1),  # 100ms块
            "callback": self._audio_callback
        }
        
        # 指定设备 (使用设备索引号)
        if device_id is not None:
            stream_params["device"] = device_id
        
        # 启动音频流
        print(f"[INFO] Starting audio stream (SR={self.config.SAMPLE_RATE}Hz)...")
        self.stream = sd.InputStream(**stream_params)
        self.stream.start()
        
        # 等待缓冲区填满
        print(f"[INFO] Filling buffer ({self.config.WINDOW_DURATION}s)...")
        time.sleep(self.config.WINDOW_DURATION)
        
        # 启动推理线程
        print("[INFO] Starting inference loop...")
        print("=" * 80)
        print(f"Real-time Snore Detection | Backend: {self.backend_type.upper()}")
        print(f"Window: {self.config.WINDOW_DURATION}s | Interval: {self.config.SLIDE_INTERVAL}s")
        print("Press Ctrl+C to stop")
        print("=" * 80)
        
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
    
    def stop(self):
        """停止检测"""
        self.is_running = False
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # 释放后端资源
        self.backend.release()
        
        print("\n[INFO] Detector stopped")
    
    def get_latest_result(self, timeout=None):
        """获取最新的检测结果 (供外部调用)"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# =====================
# 工具函数
# =====================
def list_audio_devices():
    """列出所有可用的音频设备"""
    print("\n" + "=" * 70)
    print("Available Audio Devices | 可用音频设备")
    print("=" * 70)
    
    devices = sd.query_devices()
    
    # Orange Pi 5 Plus 常见的音频设备名称
    orangepi_hints = ["es8388", "rockchip", "hdmi", "analog", "headphone"]
    
    for i, device in enumerate(devices):
        in_ch = device['max_input_channels']
        
        if in_ch > 0:  # 只显示有输入的设备
            default_marker = ""
            hint_marker = ""
            
            if i == sd.default.device[0]:
                default_marker = " [DEFAULT]"
            
            # 检查是否是 Orange Pi 常见设备
            name_lower = device['name'].lower()
            for hint in orangepi_hints:
                if hint in name_lower:
                    hint_marker = " [Orange Pi]"
                    break
            
            print(f"[{i}] {device['name']}{default_marker}{hint_marker}")
            print(f"    Inputs: {in_ch}ch, Rate: {device['default_samplerate']:.0f}Hz")
    
    print("=" * 70)
    print("\n提示: Orange Pi 5 Plus 板载 3.5mm 麦克风通常是 es8388 或 analog 设备")
    print("使用 --audio_device <ID> 指定设备，例如: --audio_device 0")


def test_audio_device(device_id=None, duration=2):
    """测试音频设备"""
    print(f"\n[TEST] Recording {duration}s of audio...")
    
    if device_id is not None:
        print(f"[TEST] Using device ID: {device_id}")
    
    try:
        recording = sd.rec(
            int(duration * Config.SAMPLE_RATE),
            samplerate=Config.SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            device=device_id
        )
        sd.wait()
        
        # 计算音量
        rms = np.sqrt(np.mean(recording ** 2))
        peak = np.max(np.abs(recording))
        
        print(f"[TEST] Recording successful!")
        print(f"       RMS: {rms:.6f}, Peak: {peak:.6f}")
        
        if peak < 0.001:
            print("[WARN] ⚠️  Audio level is very low!")
            print("       请检查:")
            print("       1. 麦克风是否正确插入 3.5mm 接口")
            print("       2. 是否选择了正确的音频设备 (--audio_device)")
            print("       3. alsamixer 中麦克风是否已开启并调高音量")
        else:
            print("[OK] ✅ Audio device is working properly!")
        
        return True
    except Exception as e:
        print(f"[ERROR] ❌ Audio test failed: {e}")
        print("\n可能的解决方案:")
        print("1. 安装音频库: sudo apt-get install portaudio19-dev python3-pyaudio")
        print("2. 检查 ALSA 配置: arecord -l")
        print("3. 尝试其他设备 ID: python3 realtime_inference.py --list_devices")
        return False


def test_mel_extraction():
    """测试Mel频谱提取"""
    print("\n[TEST] Testing Mel spectrogram extraction...")
    
    # 生成测试音频 (3秒白噪声)
    duration = Config.WINDOW_DURATION
    sr = Config.SAMPLE_RATE
    test_audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    
    # 提取Mel频谱
    start_time = time.time()
    mel = extract_mel_spectrogram(test_audio, Config)
    mel_normalized = normalize_for_inference(mel)
    mel_time = (time.time() - start_time) * 1000
    
    print(f"[TEST] Input audio shape: {test_audio.shape}")
    print(f"[TEST] Mel spectrogram shape: {mel.shape}")
    print(f"[TEST] Expected shape: ({Config.N_MELS}, ~{int(sr * duration / Config.HOP_LENGTH) + 1})")
    print(f"[TEST] Mel extraction time: {mel_time:.1f}ms")
    print(f"[TEST] Mel mean: {mel.mean():.4f}, std: {mel.std():.4f}")
    print(f"[TEST] After normalize: mean={mel_normalized.mean():.4f}, std={mel_normalized.std():.4f}")
    print("[OK] ✅ Mel extraction test passed!")


def benchmark_inference(model_path, backend="pytorch", num_runs=10):
    """推理性能测试"""
    print(f"\n[BENCHMARK] Testing inference performance ({backend})...")
    
    # 生成测试数据
    test_audio = np.random.randn(int(Config.SAMPLE_RATE * Config.WINDOW_DURATION)).astype(np.float32) * 0.1
    
    # 初始化后端
    if backend == "rknn":
        inference_backend = RKNNBackend(model_path, Config)
    else:
        inference_backend = PyTorchBackend(model_path, Config)
    
    # 预热
    mel = extract_mel_spectrogram(test_audio, Config)
    mel = normalize_for_inference(mel)
    mel_input = mel[np.newaxis, np.newaxis, :, :]
    
    _ = inference_backend.predict(mel_input)
    
    # 测试
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = inference_backend.predict(mel_input)
        times.append((time.time() - start) * 1000)
    
    inference_backend.release()
    
    print(f"[BENCHMARK] Results ({num_runs} runs):")
    print(f"  Mean: {np.mean(times):.2f}ms")
    print(f"  Min:  {np.min(times):.2f}ms")
    print(f"  Max:  {np.max(times):.2f}ms")
    print(f"  Std:  {np.std(times):.2f}ms")


# =====================
# 主函数
# =====================
def main():
    parser = argparse.ArgumentParser(
        description="Real-time Snore Detection for Orange Pi 5 Plus (RK3588)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="./mlt_best.pth",
                        help="Path to model (.pth for PyTorch, .rknn for RKNN)")
    parser.add_argument("--backend", type=str, choices=["pytorch", "rknn"], default="pytorch",
                        help="Inference backend: pytorch (CPU) or rknn (NPU)")
    
    # 音频参数
    parser.add_argument("--audio_device", type=int, default=None,
                        help="Audio input device ID. Use --list_devices to see available devices")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Audio sample rate")
    parser.add_argument("--window", type=float, default=3.0,
                        help="Window duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Slide interval in seconds")
    
    # 工具选项
    parser.add_argument("--list_devices", action="store_true",
                        help="List available audio devices and exit")
    parser.add_argument("--test_audio", action="store_true",
                        help="Test audio device and exit")
    parser.add_argument("--test_mel", action="store_true",
                        help="Test Mel spectrogram extraction and exit")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark and exit")
    
    args = parser.parse_args()
    
    # 列出设备
    if args.list_devices:
        list_audio_devices()
        return
    
    # 测试Mel提取
    if args.test_mel:
        test_mel_extraction()
        return
    
    # 测试音频
    if args.test_audio:
        list_audio_devices()
        test_audio_device(args.audio_device)
        return
    
    # 性能测试
    if args.benchmark:
        if not os.path.exists(args.model_path):
            print(f"[ERROR] Model file not found: {args.model_path}")
            sys.exit(1)
        benchmark_inference(args.model_path, args.backend)
        return
    
    # 更新配置
    Config.SAMPLE_RATE = args.sample_rate
    Config.WINDOW_DURATION = args.window
    Config.SLIDE_INTERVAL = args.interval
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        sys.exit(1)
    
    # 自动检测后端
    if args.model_path.endswith('.rknn'):
        args.backend = "rknn"
        print("[INFO] Detected RKNN model, using NPU backend")
    elif args.model_path.endswith('.pth'):
        args.backend = "pytorch"
        print("[INFO] Detected PyTorch model, using CPU backend")
    
    # 创建检测器
    detector = RealtimeSnoreDetector(
        model_path=args.model_path,
        backend=args.backend,
        config=Config
    )
    
    try:
        # 启动检测
        detector.start(device_id=args.audio_device)
        
        # 主循环
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
