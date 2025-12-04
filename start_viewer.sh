#!/bin/bash
# BidirectionalGaitNet Viewer 一键启动脚本
# 在gaitnet conda环境中运行仿真viewer

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================"
echo "  BidirectionalGaitNet Viewer 启动"
echo -e "======================================${NC}"

# 检查conda是否安装
if [ ! -f ~/miniconda3/bin/activate ]; then
    echo -e "${RED}错误: 未找到conda安装${NC}"
    exit 1
fi

# 激活gaitnet环境
echo -e "${GREEN}[1/5] 激活gaitnet conda环境...${NC}"
source ~/miniconda3/bin/activate gaitnet

# 检查环境是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "gaitnet" ]; then
    echo -e "${RED}错误: 无法激活gaitnet环境${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 环境激活成功${NC}"

# 进入项目根目录
echo -e "${GREEN}[2/5] 切换到项目目录...${NC}"
cd /home/huang/scu_robotics/BidirectionalGaitNet
echo -e "${GREEN}✓ 当前目录: $(pwd)${NC}"

# 创建必要的目录
echo -e "${GREEN}[3/5] 检查并创建必要目录...${NC}"
mkdir -p c3d
echo -e "${GREEN}✓ 目录检查完成${NC}"

# 设置环境变量
echo -e "${GREEN}[4/5] 设置环境变量...${NC}"
export LD_LIBRARY_PATH=$HOME/pkgenv/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONHOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)/python:$CONDA_PREFIX/lib/python3.6/site-packages

echo "  LD_LIBRARY_PATH: $HOME/pkgenv/lib:$CONDA_PREFIX/lib"
echo "  PYTHONHOME: $CONDA_PREFIX"
echo "  PYTHONPATH: $(pwd)/python"
echo -e "${GREEN}✓ 环境变量设置完成${NC}"

# 启动viewer
echo -e "${GREEN}[5/5] 启动Viewer...${NC}"
echo -e "${BLUE}======================================"
echo "  正在加载仿真环境..."
echo -e "======================================${NC}"

cd data
../build/viewer/viewer env.xml

# 检查退出状态
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Viewer正常退出${NC}"
elif [ $EXIT_CODE -eq 255 ]; then
    echo -e "${BLUE}ℹ Viewer已关闭${NC}"
else
    echo -e "${RED}✗ Viewer异常退出 (退出码: $EXIT_CODE)${NC}"
fi
