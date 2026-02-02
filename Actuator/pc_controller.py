#!/usr/bin/env python3
"""
气囊控制器 - PC端键盘控制程序
通过串口与Arduino通信，控制4个气囊的充气和放气

键盘布局（竖列对应同一气囊，横行对应同一操作）：
┌──────────┬─────────┬─────────┬─────────┬─────────┐
│   操作   │  气囊1  │  气囊2  │  气囊3  │  气囊4  │
├──────────┼─────────┼─────────┼─────────┼─────────┤
│ 充气 5s  │    1    │    2    │    3    │    4    │
│ 充气10s  │    Q    │    W    │    E    │    R    │
│ 放气10s  │    A    │    S    │    D    │    F    │
│ 放气15s  │    Z    │    X    │    C    │    V    │
└──────────┴─────────┴─────────┴─────────┴─────────┘

其他按键：
- ESC: 紧急停止所有操作
- H: 显示帮助信息
- Ctrl+C: 退出程序

依赖安装：
pip install pyserial pynput
"""

import serial
import serial.tools.list_ports
import time
import sys
from pynput import keyboard

# 键盘映射配置
# 格式: 按键 -> (命令类型, 气囊编号, 档位, 显示时间)
KEY_MAPPING = {
    # 充气 5秒 (数字键 1-4) - 档位1
    '1': ('P', 1, 1, 5),
    '2': ('P', 2, 1, 5),
    '3': ('P', 3, 1, 5),
    '4': ('P', 4, 1, 5),
    
    # 充气 10秒 (Q W E R) - 档位2
    'q': ('P', 1, 2, 10),
    'w': ('P', 2, 2, 10),
    'e': ('P', 3, 2, 10),
    'r': ('P', 4, 2, 10),
    
    # 放气 10秒 (A S D F) - 档位1
    'a': ('V', 1, 1, 10),
    's': ('V', 2, 1, 10),
    'd': ('V', 3, 1, 10),
    'f': ('V', 4, 1, 10),
    
    # 放气 15秒 (Z X C V) - 档位2
    'z': ('V', 1, 2, 15),
    'x': ('V', 2, 2, 15),
    'c': ('V', 3, 2, 15),
    'v': ('V', 4, 2, 15),
}


class AirbagController:
    def __init__(self, port=None, baudrate=9600):
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        
        # 自动检测或使用指定端口
        if port:
            self.port = port
        else:
            self.port = self.auto_detect_port()
    
    def auto_detect_port(self):
        """自动检测Arduino串口"""
        ports = serial.tools.list_ports.comports()
        
        print("检测到的串口设备：")
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device} - {port.description}")
        
        # 常见的Arduino串口特征
        arduino_keywords = ['Arduino', 'CH340', 'USB Serial', 'ttyUSB', 'ttyACM']
        
        for port in ports:
            for keyword in arduino_keywords:
                if keyword.lower() in port.description.lower() or keyword.lower() in port.device.lower():
                    print(f"\n自动选择: {port.device}")
                    return port.device
        
        if ports:
            print(f"\n未找到Arduino，使用第一个串口: {ports[0].device}")
            return ports[0].device
        
        return None
    
    def connect(self):
        """连接到Arduino"""
        if not self.port:
            print("错误：未找到可用的串口设备")
            print("请检查Arduino是否已连接，或手动指定端口")
            return False
        
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # 等待Arduino重置
            
            # 清除缓冲区
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # 读取Arduino的启动消息
            while self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"Arduino: {line}")
            
            print(f"\n成功连接到 {self.port}")
            return True
            
        except serial.SerialException as e:
            print(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.serial_conn and self.serial_conn.is_open:
            self.send_command('S', 0, 0)  # 停止所有操作
            time.sleep(0.1)
            self.serial_conn.close()
            print("已断开连接")
    
    def send_command(self, cmd_type, airbag, duration):
        """发送命令到Arduino"""
        if not self.serial_conn or not self.serial_conn.is_open:
            print("错误：串口未连接")
            return
        
        command = f"{cmd_type}{airbag}{duration}"
        self.serial_conn.write(command.encode('utf-8'))
        
        # 等待并读取响应
        time.sleep(0.1)
        while self.serial_conn.in_waiting:
            response = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
            if response:
                print(f"Arduino: {response}")
    
    def emergency_stop(self):
        """紧急停止"""
        print("\n!!! 紧急停止 !!!")
        self.send_command('S', 0, 0)
    
    def print_help(self):
        """打印帮助信息"""
        help_text = """
╔═══════════════════════════════════════════════════════════════╗
║                    气囊控制器 - 操作指南                       ║
╠═══════════════════════════════════════════════════════════════╣
║  键盘布局（竖列=同一气囊，横行=同一操作）                      ║
║                                                               ║
║  ┌──────────┬─────────┬─────────┬─────────┬─────────┐         ║
║  │   操作   │  气囊1  │  气囊2  │  气囊3  │  气囊4  │         ║
║  ├──────────┼─────────┼─────────┼─────────┼─────────┤         ║
║  │ 充气 5s  │    1    │    2    │    3    │    4    │         ║
║  │ 充气10s  │    Q    │    W    │    E    │    R    │         ║
║  │ 放气10s  │    A    │    S    │    D    │    F    │         ║
║  │ 放气15s  │    Z    │    X    │    C    │    V    │         ║
║  └──────────┴─────────┴─────────┴─────────┴─────────┘         ║
║                                                               ║
║  特殊按键：                                                    ║
║  - ESC    : 紧急停止所有操作                                   ║
║  - H      : 显示此帮助信息                                     ║
║  - Ctrl+C : 退出程序                                           ║
╚═══════════════════════════════════════════════════════════════╝
"""
        print(help_text)


def main():
    # 解析命令行参数
    port = None
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    # 创建控制器实例
    controller = AirbagController(port=port)
    
    # 连接到Arduino
    if not controller.connect():
        print("\n无法连接到Arduino，请检查：")
        print("1. Arduino是否已通过USB连接")
        print("2. Arduino程序是否已上传")
        print("3. 串口是否被其他程序占用")
        print("\n使用方法: python pc_controller.py [串口名称]")
        print("例如: python pc_controller.py /dev/ttyUSB0")
        print("  或: python pc_controller.py COM3")
        sys.exit(1)
    
    # 打印帮助信息
    controller.print_help()
    print("程序已就绪，等待键盘输入...\n")
    
    # 键盘事件处理
    def on_press(key):
        try:
            # 处理字符键
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                
                if char in KEY_MAPPING:
                    cmd_type, airbag, level, display_time = KEY_MAPPING[char]
                    action = "充气" if cmd_type == 'P' else "放气"
                    print(f"\n按键 [{key.char.upper()}]: 气囊{airbag} {action} {display_time}秒")
                    controller.send_command(cmd_type, airbag, level)
                
                elif char == 'h':
                    controller.print_help()
            
            # 处理特殊键
            elif key == keyboard.Key.esc:
                controller.emergency_stop()
        
        except Exception as e:
            print(f"错误: {e}")
    
    def on_release(key):
        # Ctrl+C 退出
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            pass  # 仅检测Ctrl键释放，实际退出由KeyboardInterrupt处理
    
    # 启动键盘监听
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    try:
        # 保持程序运行
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n正在退出...")
    finally:
        listener.stop()
        controller.disconnect()
        print("程序已退出")


if __name__ == "__main__":
    main()
