#!/bin/bash
cd /home/orangepi/projects/voice/voice-fastapi-dev

python main.py --config config/app_config.json "$@"
