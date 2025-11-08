import os
import re
import json
import asyncio
from datetime import datetime
import cv2
import numpy as np
import shutil
from watchdog.observers import Observer

class StateManager:
    def __init__(self):
        self.current_directory = None
        self.image_files = []
        self.observer = None
        self.websockets = set()
        self.sort_type = "name"  # 默认按名称排序
        self.sort_order = "desc"  # 默认降序排序
        self.filters = {"wubao": True, "zhengbao": True, "unclassified": True}  # 默认显示所有分类
        self.current_image_index = 0
        self.image_operations = {}
        # 用于增量更新的上一次状态
        self.last_filtered_images = []
        # 心跳包相关
        self.last_ping_time = {}  # 记录每个客户端的最后ping时间
        self.ping_interval = 10  # 心跳包间隔时间(秒)

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
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']
        files = []
        
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                # 获取文件信息
                file_stats = os.stat(file_path)
                
                # 根据排序类型获取相应的排序键
                if self.sort_type == "name":
                    # 尝试提取更复杂的数字模式，包括浮点数
                    try:
                        # 查找所有可能的数字（整数和浮点数）
                        number_patterns = re.findall(r'\d+\.\d+|\d+', file)
                        if number_patterns:
                            # 转换为数字（优先浮点数）
                            numbers = []
                            for pattern in number_patterns:
                                if '.' in pattern:
                                    numbers.append(float(pattern))
                                else:
                                    numbers.append(int(pattern))
                            # 组合多个数字作为排序键，并添加原始文件名作为最后一个元素
                            sort_key = tuple(numbers) + (file,)
                        else:
                            # 如果没有数字，使用一个固定的浮点数前缀和文件名作为排序键
                            sort_key = (float('-inf'), file)
                    except:
                        # 如果解析失败，回退到使用固定前缀和文件名作为排序键
                        sort_key = (float('-inf'), file)
                elif self.sort_type == "date":
                    # 使用文件修改时间作为排序键
                    sort_key = file_stats.st_mtime
                elif self.sort_type == "size":
                    # 使用文件大小作为排序键
                    sort_key = file_stats.st_size
                
                files.append((file, sort_key))
        
        # 按排序键排序
        # 使用自定义比较函数确保类型安全
        def safe_sort_key(item):
            key = item[1]
            # 根据不同类型进行安全处理
            if self.sort_type == "name":
                # 确保所有键都是元组
                if not isinstance(key, tuple):
                    return (float('-inf'), str(key))
                # 转换所有元素为字符串以确保类型安全的比较
                safe_key = []
                for part in key:
                    safe_key.append(str(part))
                return tuple(safe_key)
            else:
                # 对于日期和大小，直接返回数值
                return key
        
        files.sort(key=safe_sort_key, reverse=self.sort_order == "desc")
        
        return [file for file, _ in files]

    def _start_watching(self, directory):
        # 停止之前的监控
        if hasattr(self, 'observer') and self.observer:
            self.observer.stop()
            self.observer.join()
        
        # 创建观察者
        self.observer = Observer()
        # 创建事件处理器
        from src.event_handler import ImageFileEventHandler
        event_handler = ImageFileEventHandler(self)
        # 开始监控目录
        self.observer.schedule(event_handler, directory, recursive=False)
        self.observer.start()

    def update_image_files(self):
        if self.current_directory:
            self.image_files = self._get_image_files(self.current_directory)
            self._notify_clients()

    def set_sort_order(self, order, sort_type=None):
        self.sort_order = order
        if sort_type is not None:
            self.sort_type = sort_type
        self.update_image_files()

    def set_filters(self, filters):
        self.filters.update(filters)
        self._notify_clients()

    def add_websocket(self, websocket):
        self.websockets.add(websocket)

    def remove_websocket(self, websocket):
        self.websockets.remove(websocket)

    def _notify_clients(self):
        # 获取当前过滤后的图像列表
        current_filtered = self.get_filtered_images()
        # 计算增量更新数据
        incremental_data = self._calculate_incremental_update(current_filtered)
        # 保存当前状态用于下次比较
        self.last_filtered_images = current_filtered
        # 发送增量更新
        asyncio.run(self._async_notify_clients(incremental_data))

    def _calculate_incremental_update(self, current_filtered):
        """计算增量更新数据，只返回变化的部分"""
        if not self.last_filtered_images:
            # 首次更新，发送完整数据
            return {
                "type": "full_update",
                "image_files": current_filtered,
                "current_directory": self.current_directory,
                "current_index": self.current_image_index
            }
        
        # 检查是否有变化
        if len(current_filtered) != len(self.last_filtered_images):
            # 数量变化，发送完整更新
            return {
                "type": "full_update",
                "image_files": current_filtered,
                "current_directory": self.current_directory,
                "current_index": self.current_image_index
            }
        
        # 检查每个图像的状态是否变化
        has_changes = False
        for i, (current, last) in enumerate(zip(current_filtered, self.last_filtered_images)):
            if current["name"] != last["name"] or current["status"] != last["status"]:
                has_changes = True
                break
        
        if has_changes:
            # 内容变化，发送完整更新
            return {
                "type": "full_update",
                "image_files": current_filtered,
                "current_directory": self.current_directory,
                "current_index": self.current_image_index
            }
        
        # 没有变化，不发送更新
        return None

    async def _async_notify_clients(self, data):
        if data is None:
            return
            
        for websocket in list(self.websockets):
            try:
                await websocket.send_text(json.dumps(data))
            except Exception:
                # 移除无法通信的客户端
                try:
                    self.websockets.remove(websocket)
                except KeyError:
                    pass
                    
    async def send_heartbeat(self):
        """定期发送心跳包"""
        while True:
            await asyncio.sleep(self.ping_interval)
            current_time = asyncio.get_event_loop().time()
            # 检查并移除超时的连接
            for websocket in list(self.websockets):
                try:
                    # 发送ping消息
                    await websocket.send_text(json.dumps({"type": "ping"}))
                    # 更新最后ping时间
                    self.last_ping_time[id(websocket)] = current_time
                except Exception:
                    # 连接已断开，移除客户端
                    try:
                        self.websockets.remove(websocket)
                        self.last_ping_time.pop(id(websocket), None)
                    except KeyError:
                        pass
    
    async def close_all_websockets(self):
        """关闭所有WebSocket连接"""
        for ws in list(self.websockets):
            try:
                await ws.close()
            except:
                pass
        self.websockets.clear()
        self.last_ping_time.clear()
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
            
            # 计算旋转后的边界框尺寸，避免图像被裁剪
            cos_angle = abs(np.cos(np.radians(angle)))
            sin_angle = abs(np.sin(np.radians(angle)))
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # 调整旋转矩阵的平移分量
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        
        # 灰度操作
        if "grayscale" in operations and operations["grayscale"]:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # 转回RGB以保持通道数一致
        
        # 亮度对比度操作
        if "brightness" in operations or "contrast" in operations:
            brightness = operations.get("brightness", 0)
            contrast = operations.get("contrast", 1.0)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        # 自适应二值化操作
        if "adaptive_threshold" in operations and operations["adaptive_threshold"]:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # 获取参数
            block_size = operations.get("adaptive_threshold_blockSize", 11)
            c = operations.get("adaptive_threshold_c", 2)
            # 确保block_size是奇数
            block_size = max(3, block_size)
            if block_size % 2 == 0:
                block_size += 1
            # 应用自适应二值化
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, block_size, c)
            # 转回RGB
            image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # 去噪声操作
        if "denoise" in operations and operations["denoise"]:
            strength = operations.get("denoise_strength", 10)
            # 应用非局部均值去噪
            image = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        
        # 闭运算操作
        if "closing" in operations and operations["closing"]:
            kernel_size = operations.get("closing_kernelSize", 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # 开运算操作
        if "opening" in operations and operations["opening"]:
            kernel_size = operations.get("opening_kernelSize", 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # 缩放操作
        if "scale" in operations:
            scale = operations["scale"]
            if scale != 1.0:
                width = int(image.shape[1] * scale)
                height = int(image.shape[0] * scale)
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # 锐化操作
        if "sharpen" in operations:
            sharpen_strength = operations["sharpen"]
            if sharpen_strength > 0:
                # 创建锐化核
                kernel = np.array([[-1, -1, -1],
                                  [-1, 9 + sharpen_strength, -1],
                                  [-1, -1, -1]])
                # 确保图像是8位无符号整数类型
                if image.dtype != np.uint8:
                    image = cv2.convertScaleAbs(image)
                image = cv2.filter2D(image, -1, kernel)
        
        # 饱和度操作
        if "saturation" in operations:
            saturation_factor = operations["saturation"]
            if saturation_factor != 1.0:
                # 转换到HSV颜色空间
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                # 调整饱和度通道
                hsv[..., 1] = cv2.multiply(hsv[..., 1], saturation_factor)
                # 确保值在有效范围内
                hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
                # 转换回RGB颜色空间
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 翻转操作
        if "flipHorizontal" in operations and operations["flipHorizontal"]:
            image = cv2.flip(image, 1)  # 水平翻转
        if "flipVertical" in operations and operations["flipVertical"]:
            image = cv2.flip(image, 0)  # 垂直翻转
        
        return image

    def generate_thumbnail(self, filename, max_width=200, max_height=150):
        """生成缩略图"""
        image_path = self.get_image_path(filename)
        if not os.path.exists(image_path):
            return None
        
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 计算缩略图尺寸
        height, width = image.shape[:2]
        
        # 计算缩放比例
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 调整图像大小
        if scale < 1.0:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image