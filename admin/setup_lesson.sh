# Setup the environment for a lesson
# Usage:
# > bash setup_lesson.sh [lesson]
# Example:
# > bash setup_lesson.sh l1

lesson="${1}"

export DATA_ROOT=/mnt/test-data

export UV_CACHE_DIR=${DATA_ROOT}/.cache/uv

mkdir -p $UV_CACHE_DIR

mkdir -p ${DATA_ROOT}/.cache/wheels

if [ "$lesson" == "l1" ]; then
	curl -L -O https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl
	mv flash_attn-2.8.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl /workspace/.cache/wheels
fi

cd $lesson
export UV_PROJECT_ENVIRONMENT=/workspace/.venv

uv sync
