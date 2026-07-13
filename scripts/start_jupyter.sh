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
echo "Starting Jupyter inside a new detached tmux session..."

# Fix: Added -d to start the session in the background
tmux new -d -s "$SESSION" "jupyter notebook --no-browser --port=$PORT"

# Give Jupyter a couple of seconds to spin up
sleep 3

echo ""
echo "Jupyter is now running in the background!"
echo "To view the logs or grab the login token, run: tmux attach -t $SESSION"
echo "To leave it running and hide it again, press: Ctrl+b, then d"
echo ""