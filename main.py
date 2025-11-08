import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 导入分离出去的模块
from state_manager import StateManager
from event_handler import ImageFileEventHandler
from api_routes import register_routes

# 初始化状态管理器
state_manager = StateManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件 - 在应用开始接收请求前执行
    await register_routes(app, state_manager, templates)
    print("应用启动成功，路由已注册")
    yield
    # 关闭事件 - 在应用关闭前执行
    # 关闭文件监控
    if hasattr(state_manager, 'observer') and state_manager.observer:
        state_manager.observer.stop()
        state_manager.observer.join()
    # 关闭所有WebSocket连接
    await state_manager.close_all_websockets()
    print("应用关闭成功")

# 初始化FastAPI应用，使用lifespan事件处理器
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8848)