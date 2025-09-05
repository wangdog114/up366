# === 阶段 1: 基础环境和依赖安装 ===

# 1. 选择一个包含 CUDA 12.8 的官方基础镜像
# nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04 是一个不错的选择，它包含了完整的开发工具链
# "devel" 版本包含了编译工具，对于某些需要编译的pip包更友好
# "cudnn" 包含了 cuDNN 库，很多深度学习框架会用到
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# 2. 设置环境变量，避免交互式提示
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 3. 更换 apt 源为中国大陆镜像 (以清华源为例) 并安装系统依赖
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    build-essential \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. 设置 Python 和 pip 的软链接，并更换 pip 源
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# === 阶段 2: 项目代码和依赖 ===

# 5. 创建工作目录
WORKDIR /app

# 6. 复制依赖文件并安装
# (Docker分层缓存优化：只有当 requirements.txt 变化时，才会重新执行 pip install)
COPY requirements.txt ./
RUN pip install numpy==1.24.0 typing_extensions==4.15.0 --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt

# 7. 复制模型文件到镜像内的标准缓存位置
# 这会使镜像体积变大，但实现了模型与镜像的绑定
COPY .cache/torch /root/.cache/torch
COPY .cache/whisper /root/.cache/whisper
COPY .cache/huggingface /root/.cache/huggingface

# 8. 复制项目源代码和配置文件模板
COPY project/ ./project/
COPY config/ ./config/

# === 阶段 3: 运行配置 ===

# 9. 暴露端口 (如果你的应用是网络服务)
# EXPOSE 8000

# 10. 设置默认启动命令
# 假设你的主程序是 project/main.py
# 这个命令可以在 docker run 时被覆盖
CMD ["python", "project/main.py"]
