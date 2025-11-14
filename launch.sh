#!/bin/bash
cd /home/orangepi/projects/voice/voice-fastapi

python main.py --config config/app_config.json "$@"
