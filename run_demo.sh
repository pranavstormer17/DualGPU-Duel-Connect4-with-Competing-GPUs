#!/bin/bash
set -e
mkdir -p replay
./build.sh
echo "Running demo..."
./bin/dualgpu_duel | tee replay/console_output.txt
echo ""
echo "Replay written to replay/replay.txt"
