"""YOLOv8 模型模块

提供 YOLOv8 人脸检测模型的初始化和使用功能。
"""

import os
from pathlib import Path
import numpy as np
from typing import Tuple, List, Optional, Union
from ultralytics import YOLO

from ..utils.logging import get_logger

logger = get_logger(__name__)


class YOLOFaceDetector:
    """YOLOv8 人脸检测器"""

    def __init__(self, model_path: str = "models/yolov8n-face.pt"):
        """初始化 YOLOv8 人脸检测器

        Args:
            model_path: 模型文件路径
        """
        self.model = None
        self.model_path = Path(model_path)
        self._init_model()

    def _init_model(self) -> None:
        """初始化 YOLOv8 模型"""
        try:
            if not self.model_path.exists():
                logger.info("下载 YOLOv8 Face 模型...")
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            logger.error(f"YOLOv8 模型初始化失败: {str(e)}")
            raise

    def detect_face(
        self,
        img: np.ndarray,
        conf_threshold: float = 0.5,
        return_all: bool = False,
        expand_ratio: float = 0.3,
    ) -> Union[
        Tuple[bool, Optional[np.ndarray], Optional[Tuple[float, float, float, float]]],
        Tuple[bool, List[np.ndarray], List[Tuple[float, float, float, float]]],
    ]:
        """检测图片中的人脸

        Args:
            img: 输入图片
            conf_threshold: 置信度阈值
            return_all: 是否返回所有检测到的人脸
            expand_ratio: 边界框扩展比例

        Returns:
            如果 return_all 为 False:
                Tuple[bool, Optional[np.ndarray], Optional[Tuple]]:
                - 是否检测到人脸
                - 裁剪后的人脸图片（如果检测到）
                - 人脸边界框坐标 (x1, y1, x2, y2)（如果检测到）
            如果 return_all 为 True:
                Tuple[bool, List[np.ndarray], List[Tuple]]:
                - 是否检测到人脸
                - 裁剪后的人脸图片列表
                - 人脸边界框坐标列表
        """
        try:
            results = self.model(img, conf=conf_threshold)

            if len(results[0].boxes) == 0:
                return (False, img if not return_all else [], None if not return_all else [])

            # 获取所有人脸
            boxes = results[0].boxes
            conf = boxes.conf.cpu().numpy()
            if len(conf) == 0:
                return (False, img if not return_all else [], None if not return_all else [])

            h, w = img.shape[:2]

            if not return_all:
                # 只返回置信度最高的人脸
                best_idx = np.argmax(conf)
                box = boxes[best_idx].xyxy.cpu().numpy()[0]
                face_img, bbox = self._process_face_box(img, box, expand_ratio, h, w)
                return True, face_img, bbox
            else:
                # 返回所有人脸
                face_imgs = []
                face_boxes = []
                for box in boxes.xyxy.cpu().numpy():
                    face_img, bbox = self._process_face_box(img, box, expand_ratio, h, w)
                    face_imgs.append(face_img)
                    face_boxes.append(bbox)
                return True, face_imgs, face_boxes

        except Exception as e:
            logger.error(f"人脸检测失败: {str(e)}")
            return (False, img if not return_all else [], None if not return_all else [])

    def _process_face_box(
        self,
        img: np.ndarray,
        box: np.ndarray,
        expand_ratio: float,
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """处理人脸边界框

        Args:
            img: 输入图片
            box: 边界框坐标
            expand_ratio: 扩展比例
            height: 图片高度
            width: 图片宽度

        Returns:
            Tuple[np.ndarray, Tuple]: 裁剪后的人脸图片和调整后的边界框坐标
        """
        x1, y1, x2, y2 = box

        # 扩展边界框
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, x1 - w * expand_ratio)
        x2 = min(width, x2 + w * expand_ratio)
        y1 = max(0, y1 - h * expand_ratio)
        y2 = min(height, y2 + h * expand_ratio)

        # 确保边界框为正方形
        w = x2 - x1
        h = y2 - y1
        square_size = max(w, h)

        # 计算人脸中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 以中心点为中心，扩展为正方形
        half_size = square_size / 2
        x1 = max(0, center_x - half_size)
        x2 = min(width, center_x + half_size)
        y1 = max(0, center_y - half_size)
        y2 = min(height, center_y + half_size)

        # 如果正方形超出图像边界，调整位置保持正方形大小
        if x1 == 0:
            x2 = min(width, x1 + square_size)
        if x2 == width:
            x1 = max(0, x2 - square_size)
        if y1 == 0:
            y2 = min(height, y1 + square_size)
        if y2 == height:
            y1 = max(0, y2 - square_size)

        # 裁剪人脸区域
        face_img = img[int(y1):int(y2), int(x1):int(x2)]
        bbox = (x1, y1, x2, y2)

        return face_img, bbox 