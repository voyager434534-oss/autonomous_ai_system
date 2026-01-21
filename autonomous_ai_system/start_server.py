#!/usr/bin/env python3
"""
启动脚本 - 在CPU/GPU服务器上运行AI系统
自动检测设备并优化配置
"""
import os
import sys
import torch
import subprocess
from pathlib import Path


def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ 检测到 {gpu_count} 个GPU: {gpu_name}")
        return True
    else:
        print("○ 未检测到GPU,将使用CPU运行")
        return False


def install_dependencies():
    """安装依赖"""
    print("\n[安装] 检查并安装依赖包...")

    requirements_file = Path(__file__).parent / "requirements.txt"

    if requirements_file.exists():
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ 依赖安装完成")
    else:
        print("! requirements.txt 不存在,跳过依赖安装")


def optimize_for_device():
    """根据设备优化配置"""
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        # GPU优化
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        print("✓ 已启用GPU优化")
    else:
        # CPU优化
        torch.set_num_threads(os.cpu_count() or 4)
        print(f"✓ 已设置CPU线程数: {torch.get_num_threads()}")

    return use_gpu


def create_directories():
    """创建必要的目录"""
    dirs = ["sub_ai_agents", "saved_models", "logs", "temp"]

    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

    print("✓ 工作目录已创建")


def start_server():
    """启动主服务"""
    print("\n" + "="*50)
    print("启动自主AI系统")
    print("="*50)

    # 检查设备
    use_gpu = check_gpu()

    # 优化配置
    optimize_for_device()

    # 创建目录
    create_directories()

    # 检查主文件
    system_file = Path(__file__).parent / "integrated_system.py"

    if not system_file.exists():
        print(f"错误: 找不到 {system_file}")
        sys.exit(1)

    # 启动服务
    print("\n" + "="*50)
    print("服务启动中...")
    print(f"设备: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"PyTorch版本: {torch.__version__}")
    print("="*50 + "\n")

    # 导入并运行
    import integrated_system

    # 系统已在integrated_system.py中启动


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="启动自主AI系统")
    parser.add_argument("--install", action="store_true",
                       help="安装依赖包")
    parser.add_argument("--check-only", action="store_true",
                       help="仅检查系统环境")

    args = parser.parse_args()

    print("="*50)
    print("自主AI系统启动器")
    print("="*50)

    # 仅检查环境
    if args.check_only:
        print("\n[环境检查]")
        check_gpu()
        print(f"\nPython版本: {sys.version}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"工作目录: {Path.cwd()}")
        return

    # 安装依赖
    if args.install:
        install_dependencies()
        return

    # 启动服务器
    start_server()


if __name__ == "__main__":
    main()
