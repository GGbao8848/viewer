import os
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Response, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
import asyncio
import re
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 全局状态管理
class StateManager:
    def __init__(self):
        self.current_directory = None
        self.image_files = []
        self.observer = None
        self.websockets = set()
        self.sort_order = "desc"  # 默认降序排序
        self.filters = {"wubao": True, "zhengbao": True, "unclassified": True}  # 默认显示所有分类
        self.current_image_index = 0
        self.image_operations = {}

    def set_directory(self, directory):
        self.current_directory = directory
        self.image_files = self._get_image_files(directory)
        self.current_image_index = 0
        self.image_operations = {}
        self._start_watching(directory)
        self._notify_clients()

    def _get_image_files(self, directory):
        if not os.path.exists(directory):
            return []
        
        # 过滤图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        files = []
        
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                # 提取文件名中的数字用于排序
                numbers = re.findall(r'\d+', file)
                sort_key = int(numbers[0]) if numbers else 0
                files.append((file, sort_key))
        
        # 按排序键排序
        files.sort(key=lambda x: x[1], reverse=self.sort_order == "desc")
        
        return [file for file, _ in files]

    def _start_watching(self, directory):
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        self.observer = Observer()
        event_handler = ImageFileEventHandler(self)
        self.observer.schedule(event_handler, directory, recursive=False)
        self.observer.start()

    def update_image_files(self):
        if self.current_directory:
            self.image_files = self._get_image_files(self.current_directory)
            self._notify_clients()

    def set_sort_order(self, order):
        self.sort_order = order
        self.update_image_files()

    def set_filters(self, filters):
        self.filters.update(filters)
        self._notify_clients()

    def add_websocket(self, websocket):
        self.websockets.add(websocket)

    def remove_websocket(self, websocket):
        self.websockets.remove(websocket)

    def _notify_clients(self):
        data = {
            "image_files": self.get_filtered_images(),
            "current_directory": self.current_directory,
            "current_index": self.current_image_index
        }
        asyncio.run(self._async_notify_clients(data))

    async def _async_notify_clients(self, data):
        for websocket in list(self.websockets):
            try:
                await websocket.send_text(json.dumps(data))
            except Exception:
                pass

    def get_filtered_images(self):
        if not self.current_directory:
            return []
        
        wubao_dir = os.path.join(os.path.dirname(self.current_directory), "wubao")
        zhengbao_dir = os.path.join(os.path.dirname(self.current_directory), "zhengbao")
        
        filtered = []
        for file in self.image_files:
            is_wubao = os.path.exists(os.path.join(wubao_dir, file))
            is_zhengbao = os.path.exists(os.path.join(zhengbao_dir, file))
            is_unclassified = not is_wubao and not is_zhengbao
            
            if (self.filters["wubao"] and is_wubao) or \
               (self.filters["zhengbao"] and is_zhengbao) or \
               (self.filters["unclassified"] and is_unclassified):
                filtered.append({
                    "name": file,
                    "status": "wubao" if is_wubao else "zhengbao" if is_zhengbao else "unclassified"
                })
        
        return filtered

    def classify_image(self, filename, classification):
        if not self.current_directory:
            return False
        
        src_path = os.path.join(self.current_directory, filename)
        parent_dir = os.path.dirname(self.current_directory)
        
        # 移除旧的分类
        for old_class in ["wubao", "zhengbao"]:
            old_path = os.path.join(parent_dir, old_class, filename)
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except:
                    pass
        
        # 设置新的分类
        if classification in ["wubao", "zhengbao"]:
            class_dir = os.path.join(parent_dir, classification)
            os.makedirs(class_dir, exist_ok=True)
            try:
                shutil.copy2(src_path, os.path.join(class_dir, filename))
                self._notify_clients()
                return True
            except:
                return False
        
        # 清除分类
        self._notify_clients()
        return True

    def set_current_image_index(self, index):
        if 0 <= index < len(self.get_filtered_images()):
            self.current_image_index = index
            return True
        return False

    def get_current_image(self):
        filtered = self.get_filtered_images()
        if 0 <= self.current_image_index < len(filtered):
            return filtered[self.current_image_index]["name"]
        return None

    def get_image_path(self, filename):
        if not self.current_directory:
            return None
        return os.path.join(self.current_directory, filename)

    def apply_image_operation(self, filename, operation, value=None):
        if filename not in self.image_operations:
            self.image_operations[filename] = {}
        
        if operation == "reset":
            self.image_operations[filename] = {}
        else:
            self.image_operations[filename][operation] = value

    def get_image_with_operations(self, filename, temp_operations=None):
        image_path = self.get_image_path(filename)
        if not os.path.exists(image_path):
            return None
        
        # 读取原始图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        
        # 应用操作 - 如果提供了临时操作，则使用临时操作，否则使用存储的操作
        operations = temp_operations if temp_operations is not None else self.image_operations.get(filename, {})
        
        # 旋转操作
        if "rotate" in operations:
            angle = operations["rotate"]
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # 灰度操作
        if "grayscale" in operations and operations["grayscale"]:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # 转回RGB以保持通道数一致
        
        # 亮度对比度操作
        if "brightness" in operations or "contrast" in operations:
            brightness = operations.get("brightness", 0)
            contrast = operations.get("contrast", 1.0)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        return image

# 监控文件系统变化的处理器
class ImageFileEventHandler(FileSystemEventHandler):
    def __init__(self, state_manager):
        self.state_manager = state_manager

    def on_any_event(self, event):
        # 只处理文件创建、修改、删除事件，且不处理目录
        if not event.is_directory and isinstance(event, (FileModifiedEvent, FileCreatedEvent, FileDeletedEvent)):
            # 检查是否为图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
            if any(event.src_path.lower().endswith(ext) for ext in image_extensions):
                self.state_manager.update_image_files()

# 初始化状态管理器
state_manager = StateManager()

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/set_directory")
def set_directory(directory: str = Body(..., embed=True)):
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail="无效的目录路径")
    state_manager.set_directory(directory)
    return {"status": "success", "directory": directory, "images": state_manager.get_filtered_images()}

@app.get("/get_images")
def get_images():
    return {"images": state_manager.get_filtered_images(), "current_directory": state_manager.current_directory}

@app.post("/set_sort_order")
def set_sort_order(order: str = Body(..., embed=True)):
    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="排序顺序必须是'asc'或'desc'")
    state_manager.set_sort_order(order)
    return {"status": "success", "order": order, "images": state_manager.get_filtered_images()}

@app.post("/set_filters")
def set_filters(filters: dict = Body(..., embed=True)):
    state_manager.set_filters(filters)
    return {"status": "success", "filters": filters, "images": state_manager.get_filtered_images()}

@app.post("/classify_image")
def classify_image(filename: str = Body(..., embed=True), classification: str = Body(..., embed=True)):
    if classification not in ["wubao", "zhengbao", "unclassified"]:
        raise HTTPException(status_code=400, detail="分类必须是'wubao'、'zhengbao'或'unclassified'")
    success = state_manager.classify_image(filename, classification)
    if success:
        return {"status": "success", "filename": filename, "classification": classification}
    else:
        raise HTTPException(status_code=500, detail="分类失败")

@app.post("/set_current_image")
def set_current_image(index: int = Body(..., embed=True)):
    success = state_manager.set_current_image_index(index)
    if success:
        return {"status": "success", "index": index}
    else:
        raise HTTPException(status_code=400, detail="无效的图像索引")

@app.get("/get_current_image")
def get_current_image():
    filename = state_manager.get_current_image()
    if filename:
        return {"filename": filename, "index": state_manager.current_image_index}
    else:
        return {"filename": None, "index": -1}

@app.post("/apply_image_operation")
def apply_image_operation(filename: str = Body(..., embed=True), operation: str = Body(..., embed=True), value: str = Body(None, embed=True)):
    if operation == "rotate":
        value = float(value) if value else 0
    elif operation == "grayscale":
        value = value.lower() == "true"
    elif operation == "brightness":
        value = int(value) if value else 0
    elif operation == "contrast":
        value = float(value) if value else 1.0
    
    state_manager.apply_image_operation(filename, operation, value)
    return {"status": "success"}

@app.post("/apply_global_image_operations")
def apply_global_image_operations(filename: str = Body(..., embed=True), params: dict = Body(..., embed=True), just_get: bool = Body(False, embed=True)):
    if just_get:
        # 仅获取处理后的图像，不保存操作状态
        # 创建临时操作字典
        temp_operations = {}
        
        # 应用全局参数到临时操作字典
        if "rotate" in params:
            temp_operations["rotate"] = float(params["rotate"])
        if "brightness" in params:
            temp_operations["brightness"] = int(params["brightness"])
        if "contrast" in params:
            temp_operations["contrast"] = float(params["contrast"])
        if "grayscale" in params:
            temp_operations["grayscale"] = params["grayscale"]
        
        # 获取处理后的图像
        image = state_manager.get_image_with_operations(filename, temp_operations)
        if image is None:
            raise HTTPException(status_code=404, detail="图像未找到")
        
        # 将图像转换为字节流
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    else:
        # 重置该图像的所有操作
        if filename not in state_manager.image_operations:
            state_manager.image_operations[filename] = {}
        else:
            state_manager.image_operations[filename] = {}
        
        # 应用全局参数
        if "rotate" in params:
            state_manager.apply_image_operation(filename, "rotate", float(params["rotate"]))
        if "brightness" in params:
            state_manager.apply_image_operation(filename, "brightness", int(params["brightness"]))
        if "contrast" in params:
            state_manager.apply_image_operation(filename, "contrast", float(params["contrast"]))
        if "grayscale" in params:
            state_manager.apply_image_operation(filename, "grayscale", params["grayscale"])
        
        return {"status": "success"}

@app.get("/get_processed_image")
def get_processed_image(filename: str):
    # 使用默认参数调用，确保与修改后的函数兼容
    image = state_manager.get_image_with_operations(filename, None)
    if image is None:
        raise HTTPException(status_code=404, detail="图像未找到")
    
    # 将图像转换为字节流
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.post("/clear_classification")
def clear_classification(filename: str = None):
    if filename:
        # 清除指定图像的分类
        state_manager.classify_image(filename, "unclassified")
    else:
        # 清除所有图像的分类
        if state_manager.current_directory:
            parent_dir = os.path.dirname(state_manager.current_directory)
            for class_dir in ["wubao", "zhengbao"]:
                dir_path = os.path.join(parent_dir, class_dir)
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        try:
                            os.remove(os.path.join(dir_path, file))
                        except:
                            pass
            state_manager._notify_clients()
    return {"status": "success"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state_manager.add_websocket(websocket)
    
    try:
        while True:
            # 保持WebSocket连接活跃
            await websocket.receive_text()
    except WebSocketDisconnect:
        state_manager.remove_websocket(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8848)