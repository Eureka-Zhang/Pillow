import serial
import time
import struct
import numpy as np
import argparse
import threading
import queue
import os
import sys

# 尝试导入 RKNN
try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("[WARN] rknn-toolkit-lite2 not found! Will simulate inference.")
    RKNNLite = None

# =====================
# 配置参数
# =====================
class Config:
    # 串口设置
    SERIAL_PORT = "/dev/ttyUSB0"
    BAUDRATE = 115200
    
    # 采样设置
    SAMPLE_INTERVAL = 0.25  # 250ms
    BATCH_SIZE = 4          # 4帧一组 (1秒)
    
    # 数据预处理
    NORM_SCALE = 4095.0     # 归一化分母 (根据传感器量程调整，假设12bit)
    
    # 模型路径
    MODEL_PATH = "Sensor/Pressure/pressure_model.rknn"
    
    # 类别名称 (对应 model_pr.py)
    CLASS_NAMES = ["Middle", "Left-Back", "Right-Back", "Left", "Right"]

# =====================
# 全局变量
# =====================
latest_pressure_matrix = None
matrix_lock = threading.Lock()
running = True

# =====================
# 推理核心类
# =====================
class PressureInference:
    def __init__(self, model_path):
        self.rknn = None
        if RKNNLite and os.path.exists(model_path):
            self.rknn = RKNNLite()
            
            # 加载模型
            print(f"[RKNN] Loading model: {model_path}")
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                print(f"[RKNN] Load model failed!")
                self.rknn = None
                return
                
            # 初始化运行时 (使用 NPU 核心)
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            if ret != 0:
                print(f"[RKNN] Init runtime failed!")
                self.rknn = None
                return
            print(f"[RKNN] Model loaded successfully!")
        else:
            print(f"[WARN] RKNN model not loaded (simulating mode)")

    def predict(self, input_data):
        """
        执行推理
        Args:
            input_data: np.ndarray, shape (4, 256), float32, normalized 0-1
        Returns:
            class_id (int), probs (list)
        """
        if self.rknn is None:
            return 0, [0.9, 0.1, 0.0, 0.0, 0.0]
        
        # 尝试扩展为 4 维: (1, 1, 4, 256)
        # input_data is (4, 256)
        input_tensor = input_data[np.newaxis, np.newaxis, :, :]  # -> (1, 1, 4, 256)
        input_tensor = input_tensor.astype(np.float32)
        
        try:
            outputs = self.rknn.inference(inputs=[input_tensor])
        except Exception as e:
            # 备选：尝试 (1, 4, 256, 1)
            try:
                # print("[WARN] Retrying with shape (1, 4, 256, 1)...")
                input_tensor = input_data[np.newaxis, :, :, np.newaxis]
                input_tensor = input_tensor.astype(np.float32)
                outputs = self.rknn.inference(inputs=[input_tensor])
            except Exception:
                outputs = None

        if outputs is None:
            return 0, [0.0]*5
        
        # 解析输出
        logits = outputs[0][0]  # (5,)
        probs = self._softmax(logits)
        class_id = np.argmax(probs)
        
        return class_id, probs

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def release(self):
        if self.rknn:
            self.rknn.release()

# =====================
# 采样与推理线程
# =====================
def inference_thread_func(model_path):
    global running
    
    # 初始化推理引擎
    engine = PressureInference(model_path)
    
    buffer = []
    print("[Inference] Thread started. Waiting for data...")
    
    while running:
        start_time = time.time()
        
        # 1. 采样最新帧
        current_frame = None
        with matrix_lock:
            if latest_pressure_matrix is not None:
                current_frame = latest_pressure_matrix.copy()
        
        if current_frame is not None:
            # 展平为 (256,)
            flat_frame = current_frame.flatten()
            buffer.append(flat_frame)
        else:
            # 如果没有数据，且 buffer 为空，可以不做动作
            # 如果 buffer 不为空，可能需要补上一帧或者等待
            if len(buffer) > 0:
                # 简单策略：重复上一帧
                buffer.append(buffer[-1])
        
        # 2. 攒够 4 帧 -> 推理
        if len(buffer) == Config.BATCH_SIZE:
            # 拼接 -> (4, 256)
            input_batch = np.array(buffer, dtype=np.float32)
            
            # 归一化
            input_batch = input_batch / Config.NORM_SCALE
            
            # 执行推理
            class_id, probs = engine.predict(input_batch)
            class_name = Config.CLASS_NAMES[class_id]
            prob_val = probs[class_id]
            
            # 打印结果
            print(f"[Result] {class_name:<10} (Conf: {prob_val:.2f}) | Probs: {[f'{p:.2f}' for p in probs]}")
            
            # 清空缓冲区
            buffer = []
            
        # 3. 控制 250ms 周期
        elapsed = time.time() - start_time
        sleep_time = Config.SAMPLE_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
            
    engine.release()
    print("[Inference] Thread stopped")

# =====================
# 串口解析逻辑 (复用)
# =====================
def process_packet(packet):
    global latest_pressure_matrix
    
    if packet[0] != 0xAA or packet[1] != 0xAB or packet[2] != 0xAC:
        return

    data_bytes = packet[3:-1]
    try:
        values = struct.unpack('<256h', data_bytes)
        arr = np.array(values)
        matrix = arr.reshape(16, 16)
        
        with matrix_lock:
            latest_pressure_matrix = matrix
    except Exception:
        pass

def serial_loop(port, baudrate):
    global running
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        print(f"[Serial] Opened {port} @ {baudrate}")
        
        buffer = bytearray()
        PACKET_SIZE = 516
        
        while running:
            if ser.in_waiting:
                buffer.extend(ser.read(ser.in_waiting))
            
            while len(buffer) >= 3:
                header_idx = buffer.find(b'\xAA\xAB\xAC')
                if header_idx == -1:
                    if len(buffer) > PACKET_SIZE * 2:
                        buffer = buffer[-PACKET_SIZE:]
                    break
                
                if len(buffer) < header_idx + PACKET_SIZE:
                    break
                
                packet = buffer[header_idx : header_idx + PACKET_SIZE]
                process_packet(packet)
                buffer = buffer[header_idx + PACKET_SIZE:]
                
            time.sleep(0.001)
            
    except Exception as e:
        print(f"[Serial] Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

# =====================
# 主函数
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=Config.SERIAL_PORT)
    parser.add_argument("--model", default=Config.MODEL_PATH)
    args = parser.parse_args()
    
    # 启动推理线程
    t_inf = threading.Thread(target=inference_thread_func, args=(args.model,), daemon=True)
    t_inf.start()
    
    # 在主线程运行串口接收 (或也是子线程)
    try:
        serial_loop(args.port, Config.BAUDRATE)
    except KeyboardInterrupt:
        print("\nStopping...")
        global running
        running = False
        time.sleep(1)

if __name__ == "__main__":
    main()
