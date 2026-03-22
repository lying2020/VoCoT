#!/usr/bin/env bash
# 一步创建 conda 环境 vocot 并安装 VoCoT 依赖（requirements.txt）。
# 用法：
#   ./setup_conda.sh              # 仅创建环境并 pip install -r requirements.txt（需已能 import torch 或随后自行装 PyTorch）
#   ./setup_conda.sh cu121        # 先按 CUDA 12.1 从 PyTorch 官方 wheel 源安装 torch/torchvision，再装 requirements
#   ./setup_conda.sh cpu          # CPU 版 PyTorch
# 也可通过环境变量覆盖：
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./setup_conda.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
ENV_NAME="${CONDA_ENV_NAME:-vocot}"
REQ="$REPO_ROOT/requirements.txt"

die() { echo "error: $*" >&2; exit 1; }

[[ -f "$REQ" ]] || die "找不到 $REQ，请在 VoCoT 仓库根目录运行本脚本。"

command -v conda >/dev/null 2>&1 || die "未找到 conda，请先安装 Miniconda/Anaconda 并初始化 shell。"

# 若已存在同名环境，默认跳过创建（可用 FORCE_RECREATE=1 强制删除后重建）
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "${FORCE_RECREATE:-0}" == "1" ]]; then
    echo "移除已有环境: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
  else
    echo "conda 环境已存在: $ENV_NAME（设置 FORCE_RECREATE=1 可删除并重建）"
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "创建 conda 环境: $ENV_NAME (python=3.9)"
  conda create -n "$ENV_NAME" python=3.9 -y
fi

run_pip() {
  conda run -n "$ENV_NAME" "$@"
}

echo "升级 pip..."
run_pip python -m pip install --upgrade pip

# 可选：先安装 PyTorch（减少与 CUDA 的版本冲突）
# 优先级：TORCH_INDEX_URL 环境变量 > 第一个参数 cu121/cpu/...
TORCH_ARG="${1:-}"
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  echo "按 TORCH_INDEX_URL 安装 torch torchvision: ${TORCH_INDEX_URL}"
  run_pip pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
elif [[ -n "$TORCH_ARG" ]]; then
  case "$TORCH_ARG" in
    cpu)
      IDX="https://download.pytorch.org/whl/cpu"
      ;;
    cu118)
      IDX="https://download.pytorch.org/whl/cu118"
      ;;
    cu121)
      IDX="https://download.pytorch.org/whl/cu121"
      ;;
    cu124)
      IDX="https://download.pytorch.org/whl/cu124"
      ;;
    cu126)
      IDX="https://download.pytorch.org/whl/cu126"
      ;;
    *)
      die "未知 CUDA 选项: $TORCH_ARG（可用: cpu, cu118, cu121, cu124, cu126）"
      ;;
  esac
  echo "安装 torch torchvision（$TORCH_ARG）: $IDX"
  run_pip pip install torch torchvision --index-url "$IDX"
else
  echo "未指定 PyTorch CUDA 变体；将直接安装 requirements.txt（其中含 torch>=2.0）。"
  echo "若需与系统 CUDA 对齐，可重跑: $0 cu121"
fi

echo "安装 requirements.txt ..."
run_pip pip install -r "$REQ"

echo ""
echo "完成。使用前先激活环境:"
echo "  conda activate $ENV_NAME"
echo ""
echo "可选：Flash Attention（编译失败可跳过）:"
echo "  conda activate $ENV_NAME && pip install flash-attn --no-build-isolation"
