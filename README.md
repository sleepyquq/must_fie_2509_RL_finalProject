# BipedalWalker 强化学习项目

## 项目描述
使用 SAC 算法训练 BipedalWalker-v3，并生成演示视频。

## 环境依赖
- Python 3.8 - 3.10 (建议)
- Windows, Linux, 或 macOS

## 安装步骤
1. 打开终端或命令行。
2. 创建虚拟环境:
   python -m venv .venv
3. 激活虚拟环境:
   - Windows: .venv\Scripts\activate
   - Linux/Mac: source .venv/bin/activate
4. 安装依赖:
   pip install -r requirements.txt 

## 运行项目
运行以下命令开始训练并生成视频：
python main.py

## 输出
程序运行结束后，.\bipedal_sac_logs\final_videos目录下会生成得分最高的三个视频。