#!/bin/bash

SESSION="jupyter"
PORT=8888

REPO=$(git rev-parse --show-toplevel)
cd $REPO

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Found existing tmux session: $SESSION"
    echo "Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "No existing tmux session named '$SESSION' found."
echo "Starting Jupyter inside a new tmux session..."
echo ""
echo "After Jupyter starts, detach with: Ctrl+b, then d"
echo ""

tmux new -s "$SESSION" "jupyter notebook --no-browser --port=$PORT"