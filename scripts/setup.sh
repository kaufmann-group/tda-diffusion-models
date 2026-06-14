#!/bin/bash

# Set virtual environment as .venv, adds Overleaf project as remote Git repository, installs libraries from requirements.txt and adds .venv to Python Kernal for Jupyter

set -e

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

set -a
source .env
set +a

if ! git remote | grep -q 'overleaf'; then
    git remote add overleaf "https://git@git.overleaf.com/$OVERLEAF_ID"
    echo "Added Overleaf remote."
else
    echo "Overleaf remote exists"
fi

python3 -m venv .venv
source $REPO/.venv/bin/activate

case ":$PATH:" in
    *":$REPO/.venv/bin:"*) ;;
    *) export PATH="$REPO/.venv/bin:$PATH" ;;
esac

pip install --upgrade pip
pip install -r ./scripts/requirements.txt
python -m ipykernel install --user --name=repo-env --display-name "Repository Environment"

# the only thing that works on the rcac server instead you could just brew install eigen and manually change the path in the make file
rm -rf "$REPO/.eigen"
mkdir -p "$REPO/.eigen"
git clone --depth 1 https://gitlab.com/libeigen/eigen.git "$REPO/eigen"
cp -a "$REPO/eigen/Eigen" "$REPO/.eigen/"
rm -rf "$REPO/eigen"

make -C src/utils/msep

echo ""
echo "Setup complete."
echo "Activate the environment with: source .venv/bin/activate"
