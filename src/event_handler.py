from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

class ImageFileEventHandler(FileSystemEventHandler):
    def __init__(self, state_manager):
        self.state_manager = state_manager

    def on_any_event(self, event):
        # 只处理文件创建、修改、删除事件，且不处理目录
        if not event.is_directory and isinstance(event, (FileModifiedEvent, FileCreatedEvent, FileDeletedEvent)):
            # 检查是否为图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']
            if any(event.src_path.lower().endswith(ext) for ext in image_extensions):
                self.state_manager.update_image_files()