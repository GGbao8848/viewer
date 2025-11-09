from fastapi import HTTPException, Body, WebSocket, WebSocketDisconnect, Response
from starlette.requests import Request
import asyncio
import json
import os
import cv2

async def register_routes(app, state_manager, templates):
    """注册所有API路由"""
    
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
    def set_sort_order(order: str = Body(..., embed=True), sort_type: str = Body(None, embed=True)):
        if order not in ["asc", "desc"]:
            raise HTTPException(status_code=400, detail="排序顺序必须是'asc'或'desc'")
        if sort_type is not None and sort_type not in ["name", "date", "size"]:
            raise HTTPException(status_code=400, detail="排序类型必须是'name'、'date'或'size'")
        state_manager.set_sort_order(order, sort_type)
        return {"status": "success", "order": order, "sort_type": sort_type, "images": state_manager.get_filtered_images()}

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
        # 处理可能的None值
        if value is None:
            processed_value = None
        elif operation == "rotate":
            processed_value = float(value) if value else 0
        elif operation == "grayscale":
            processed_value = value.lower() == "true"
        elif operation == "brightness":
            processed_value = int(value) if value else 0
        elif operation == "contrast":
            processed_value = float(value) if value else 1.0
        elif operation == "adaptive_threshold":
            processed_value = value.lower() == "true"
        elif operation == "denoise":
            processed_value = value.lower() == "true"
        elif operation == "closing":
            processed_value = value.lower() == "true"
        elif operation == "opening":
            processed_value = value.lower() == "true"
        elif operation == "scale":
            processed_value = float(value) if value else 1.0
        elif operation == "sharpen":
            processed_value = float(value) if value else 0.0
        elif operation == "saturation":
            processed_value = float(value) if value else 1.0
        elif operation == "flipHorizontal":
            processed_value = value.lower() == "true"
        elif operation == "flipVertical":
            processed_value = value.lower() == "true"
        else:
            processed_value = value
        
        state_manager.apply_image_operation(filename, operation, processed_value)
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
            if "adaptiveThreshold" in params:
                temp_operations["adaptive_threshold"] = params["adaptiveThreshold"]
                temp_operations["adaptive_threshold_blockSize"] = params.get("adaptiveThresholdBlockSize", 11)
                temp_operations["adaptive_threshold_c"] = params.get("adaptiveThresholdC", 2)
            if "denoise" in params:
                temp_operations["denoise"] = params["denoise"]
                temp_operations["denoise_strength"] = params.get("denoiseStrength", 10)
            if "closing" in params:
                temp_operations["closing"] = params["closing"]
                temp_operations["closing_kernelSize"] = params.get("closingKernelSize", 3)
            if "opening" in params:
                temp_operations["opening"] = params["opening"]
                temp_operations["opening_kernelSize"] = params.get("openingKernelSize", 3)
            # 新增功能参数
            if "scale" in params:
                temp_operations["scale"] = float(params["scale"])
            if "sharpen" in params:
                temp_operations["sharpen"] = float(params["sharpen"])
            if "saturation" in params:
                temp_operations["saturation"] = float(params["saturation"])
            if "flipHorizontal" in params:
                temp_operations["flipHorizontal"] = params["flipHorizontal"]
            if "flipVertical" in params:
                temp_operations["flipVertical"] = params["flipVertical"]
            
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
            if "adaptiveThreshold" in params:
                state_manager.apply_image_operation(filename, "adaptive_threshold", params["adaptiveThreshold"])
                state_manager.apply_image_operation(filename, "adaptive_threshold_blockSize", params.get("adaptiveThresholdBlockSize", 11))
                state_manager.apply_image_operation(filename, "adaptive_threshold_c", params.get("adaptiveThresholdC", 2))
            if "denoise" in params:
                state_manager.apply_image_operation(filename, "denoise", params["denoise"])
                state_manager.apply_image_operation(filename, "denoise_strength", params.get("denoiseStrength", 10))
            if "closing" in params:
                state_manager.apply_image_operation(filename, "closing", params["closing"])
                state_manager.apply_image_operation(filename, "closing_kernelSize", params.get("closingKernelSize", 3))
            if "opening" in params:
                state_manager.apply_image_operation(filename, "opening", params["opening"])
                state_manager.apply_image_operation(filename, "opening_kernelSize", params.get("openingKernelSize", 3))
            # 新增功能参数
            if "scale" in params:
                state_manager.apply_image_operation(filename, "scale", float(params["scale"]))
            if "sharpen" in params:
                state_manager.apply_image_operation(filename, "sharpen", float(params["sharpen"]))
            if "saturation" in params:
                state_manager.apply_image_operation(filename, "saturation", float(params["saturation"]))
            if "flipHorizontal" in params:
                state_manager.apply_image_operation(filename, "flipHorizontal", params["flipHorizontal"])
            if "flipVertical" in params:
                state_manager.apply_image_operation(filename, "flipVertical", params["flipVertical"])
            
            return {"status": "success"}

    @app.get("/get_processed_image")
    def get_processed_image(filename: str):
        # 使用默认参数调用，确保与修改后的函数兼容
        image = state_manager.get_image_with_operations(filename, None)
        if image is None:
            raise HTTPException(status_code=404, detail="图像未找到")
    
    @app.post("/export_processed_images")
    def export_processed_images(params: dict = Body(..., embed=True)):
        """批量导出处理后的图像到export目录（与wubao和zhengbao同级）"""
        if not state_manager.current_directory:
            raise HTTPException(status_code=400, detail="未设置当前目录")
        
        # 获取父目录（wubao和zhengbao所在的目录）
        parent_dir = os.path.dirname(state_manager.current_directory)
        # 创建export目录
        export_dir = os.path.join(parent_dir, "export")
        os.makedirs(export_dir, exist_ok=True)
        
        # 获取需要处理的图像列表
        filtered_images = state_manager.get_filtered_images()
        if not filtered_images:
            raise HTTPException(status_code=400, detail="没有可导出的图像")
        
        # 处理并导出每个图像
        success_count = 0
        error_count = 0
        
        for img_info in filtered_images:
            filename = img_info["name"]
            try:
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
                if "adaptiveThreshold" in params:
                    temp_operations["adaptive_threshold"] = params["adaptiveThreshold"]
                    temp_operations["adaptive_threshold_blockSize"] = params.get("adaptiveThresholdBlockSize", 11)
                    temp_operations["adaptive_threshold_c"] = params.get("adaptiveThresholdC", 2)
                if "denoise" in params:
                    temp_operations["denoise"] = params["denoise"]
                    temp_operations["denoise_strength"] = params.get("denoiseStrength", 10)
                if "closing" in params:
                    temp_operations["closing"] = params["closing"]
                    temp_operations["closing_kernelSize"] = params.get("closingKernelSize", 3)
                if "opening" in params:
                    temp_operations["opening"] = params["opening"]
                    temp_operations["opening_kernelSize"] = params.get("openingKernelSize", 3)
                # 新增功能参数
                if "scale" in params:
                    temp_operations["scale"] = float(params["scale"])
                if "sharpen" in params:
                    temp_operations["sharpen"] = float(params["sharpen"])
                if "saturation" in params:
                    temp_operations["saturation"] = float(params["saturation"])
                if "flipHorizontal" in params:
                    temp_operations["flipHorizontal"] = params["flipHorizontal"]
                if "flipVertical" in params:
                    temp_operations["flipVertical"] = params["flipVertical"]
                
                # 获取处理后的图像
                image = state_manager.get_image_with_operations(filename, temp_operations)
                if image is not None:
                    # 保存到export目录
                    export_path = os.path.join(export_dir, filename)
                    # 确保图像格式正确
                    if len(image.shape) == 2:
                        # 灰度图
                        cv2.imwrite(export_path, image)
                    else:
                        # RGB图转换为BGR
                        cv2.imwrite(export_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"导出图像 {filename} 失败: {str(e)}")
                error_count += 1
        
        return {
            "status": "success",
            "export_dir": export_dir,
            "success_count": success_count,
            "error_count": error_count,
            "total_count": len(filtered_images)
        }
        
        # 将图像转换为字节流
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    @app.get("/get_original_image")
    def get_original_image(filename: str):
        """获取原始图像（不应用任何处理）"""
        image_path = state_manager.get_image_path(filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="图像未找到")
        
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=500, detail="无法读取图像")
        
        # 将图像转换为字节流
        _, buffer = cv2.imencode('.jpg', image)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    @app.get("/get_image_metadata")
    def get_image_metadata(filename: str):
        """获取图像元数据"""
        image_path = state_manager.get_image_path(filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="图像未找到")
        
        try:
            # 获取基本文件信息
            file_stats = os.stat(image_path)
            
            # 读取图像获取尺寸和通道数
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("无法读取图像")
            
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            
            # 获取文件扩展名作为格式
            _, ext = os.path.splitext(filename)
            format_type = ext.lower().lstrip('.')
            
            metadata = {
                "filename": filename,
                "width": width,
                "height": height,
                "channels": channels,
                "format": format_type,
                "size_kb": round(file_stats.st_size / 1024, 2),
                "created_time": datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                "modified_time": datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return metadata
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取元数据失败: {str(e)}")

    @app.get("/get_thumbnail")
    def get_thumbnail(filename: str):
        """获取缩略图"""
        image = state_manager.generate_thumbnail(filename)
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
        client_id = id(websocket)
        state_manager.add_websocket(websocket)
        state_manager.last_ping_time[client_id] = asyncio.get_event_loop().time()
        
        try:
            # 首次连接时发送完整数据
            initial_data = {
                "type": "full_update",
                "image_files": state_manager.get_filtered_images(),
                "current_directory": state_manager.current_directory,
                "current_index": state_manager.current_image_index
            }
            await websocket.send_text(json.dumps(initial_data))
            
            # 启动心跳任务
            asyncio.create_task(state_manager.send_heartbeat())
            
            while True:
                # 接收客户端消息
                data = await websocket.receive_text()
                
                # 处理客户端消息
                try:
                    message = json.loads(data)
                    # 处理心跳响应
                    if message.get("type") == "pong":
                        # 更新客户端的最后响应时间
                        state_manager.last_ping_time[client_id] = asyncio.get_event_loop().time()
                except json.JSONDecodeError:
                    pass  # 忽略无效的JSON消息
        except WebSocketDisconnect:
            state_manager.remove_websocket(websocket)
            state_manager.last_ping_time.pop(client_id, None)
        except Exception as e:
            print(f"WebSocket error: {e}")
            state_manager.remove_websocket(websocket)
            state_manager.last_ping_time.pop(client_id, None)