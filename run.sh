#!/bin/bash
#激活环境
conda activate AAA 

# 安装依赖
pip install -r requirements.txt

# 运行FastAPI服务器
uvicorn main:app --reload --host 127.0.0.1 --port 8000
