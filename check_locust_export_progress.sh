#!/bin/bash

# Define the command to check and run
COMMAND="python -m GEE.locust_processing.cli --balanced-sampling --country Ethiopia --progress-file ethiopia_export_progress.json"
LOG_FILE="ethiopia_export_may_8_2025.log"

# Check if the command is running
if ! pgrep -f "$COMMAND" > /dev/null; then
    echo "Command not running, starting it now..."
    nohup $COMMAND > $LOG_FILE 2>&1 &
    echo "Command started with PID $!"
else
    echo "Command is already running."
fi