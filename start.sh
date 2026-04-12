#!/bin/bash
set -e

# === 可覆盖的默认环境变量 ===
export COMMAND_FORWARD_URL="${COMMAND_FORWARD_URL:-http://192.168.20.13:9010/task/voice/run}"
export COMMAND_FORWARD_TIMEOUT="${COMMAND_FORWARD_TIMEOUT:-5}"

# === 初始化 conda ===
source /home/orangepi/miniconda3/etc/profile.d/conda.sh
conda activate voice310

# === 进入项目目录 ===
cd /home/orangepi/voice/voice-fastapi

# === 启动服务 ===
exec python main.py --config config/app_config.json
