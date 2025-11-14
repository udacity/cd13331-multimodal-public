export DATA_ROOT=/voc/data
export UV_CACHE_DIR=${DATA_ROOT}/.cache/uv
export WHEELS_REPO=${DATA_ROOT}/.cache/wheels
export VENV_ROOT=${DATA_ROOT}/venvs

mkdir -p $UV_CACHE_DIR
mkdir -p $WHEELS_REPO
mkdir -p ${VENV_ROOT}

curl -L -O https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl
mv flash_attn-2.8.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl ${WHEELS_REPO}

echo "Searching for pyproject.toml files..."
pyproject_files=$(find . -name "pyproject.toml" -type f | sort)
echo "Found files:"
echo "$pyproject_files"
echo ""

for pyproject in $pyproject_files; do
    echo "----------------------------------------"
    echo "Processing: $pyproject"
    
    # Get the root directory (parent directory of pyproject.toml)
    # (the sed part is needed to remove the "./" from the path, like in 
    # ./l1/demos/1-pydantic/)
    root=$(dirname "$pyproject" | sed "s|\./||g")
    echo "  Root directory: $root"
    
    # Set the environment variable
    export UV_PROJECT_ENVIRONMENT="${VENV_ROOT}/${root}"
    echo "  UV_PROJECT_ENVIRONMENT set to: $UV_PROJECT_ENVIRONMENT"
    
    # Create the folder
    echo "  Creating directory: $UV_PROJECT_ENVIRONMENT"
    mkdir -p "${UV_PROJECT_ENVIRONMENT}"
    if [ $? -eq 0 ]; then
        echo "  ✓ Directory created successfully"
    else
        echo "  ✗ Failed to create directory"
        continue
    fi
    
    # # Go to the root folder and run uv sync
    echo "  Changing to directory: $root"
    echo "  Running: uv sync"
    (cd "$root" && pwd && uv sync)
    if [ $? -eq 0 ]; then
        echo "  ✓ uv sync completed successfully"
    else
        echo "  ✗ uv sync failed"
    fi

    mkdir -p "$root/.vscode"

cat << EOF > "${root}/.vscode/settings.json"
{
  "python.defaultInterpreterPath": "${UV_PROJECT_ENVIRONMENT}/bin/python",
  "git.openRepositoryInParentFolders": "never"
}
EOF
    
    echo "  Finished processing: $root"
    echo ""
done