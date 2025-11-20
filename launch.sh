#!/bin/bash
cd /home/orangepi/projects/voice/voice-fastapi-dev

export COMMAND_FORWARD_URL="http://127.0.0.1:8089/send/message"
export COMMAND_FORWARD_TIMEOUT="5"  # 可选

python main.py --config config/app_config.json "$@"
