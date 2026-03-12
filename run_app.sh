#!/bin/bash
# 智能止鼾枕 - 整体 App 联合运行（TCP 模式）
# 启动总控后，用 App 或 nc <本机IP> 8765 连接，发送 START_SLEEP / STOP_SLEEP 等指令。

set -e
cd "$(dirname "$0")"
TCP_PORT="${TCP_PORT:-8765}"
echo "启动总控: TCP 0.0.0.0:${TCP_PORT}（协议与蓝牙一致）"
echo "连接示例: nc <本机IP> ${TCP_PORT}  然后发送 START_SLEEP"
echo "---"
exec python3 smart_pillow_controller.py --tcp-port "$TCP_PORT" "$@"
