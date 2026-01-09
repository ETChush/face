"""MediaPipe 模型模块

提供 MediaPipe 人脸关键点检测功能。
使用 MediaPipe 0.10+ 的 Tasks API。
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, Any, List
import os

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MediaPipeFaceDetector:
    """MediaPipe 人脸关键点检测器 (使用 Tasks API)"""

    def __init__(self):
        """初始化 MediaPipe 人脸关键点检测器"""
        self.model_path = self._get_model_path()
        self.face_landmarker = self._create_landmarker()

    def _get_model_path(self) -> str:
        """获取模型文件路径
        
        Returns:
            str: 模型文件路径
        """
        # 检查多个可能的模型文件位置
        possible_paths = [
            "models/face_landmarker.task",
            "models/face_landmarker_v2.task",
            os.path.join(os.path.dirname(__file__), "../../models/face_landmarker.task"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"找到模型文件: {path}")
                return path
        
        # 如果找不到模型文件，使用默认路径
        default_path = "models/face_landmarker.task"
        logger.warning(f"未找到模型文件，将使用: {default_path}")
        logger.warning("请从 https://github.com/google/mediapipe/releases 下载 face_landmarker.task")
        return default_path

    def _create_landmarker(self) -> Optional[vision.FaceLandmarker]:
        """创建 FaceLandmarker 实例
        
        Returns:
            Optional[vision.FaceLandmarker]: FaceLandmarker 实例
        """
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("FaceLandmarker 创建成功")
            return landmarker
        except Exception as e:
            logger.error(f"创建 FaceLandmarker 失败: {str(e)}")
            return None

    def get_face_landmarks(
        self, img: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[Any]:
        """获取人脸关键点

        Args:
            img: 输入图片
            bbox: 可选的人脸边界框坐标 (x1, y1, x2, y2)

        Returns:
            Optional[Any]: MediaPipe 检测到的人脸关键点
        """
        if self.face_landmarker is None:
            logger.error("FaceLandmarker 未初始化")
            return None

        try:
            # 如果提供了边界框，先裁剪图片
            face_img = img
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                face_img = img[y1:y2, x1:x2]

            # 转换为RGB格式
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 创建 MediaPipe 图像对象
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # 检测关键点
            detection_result = self.face_landmarker.detect(mp_image)

            if not detection_result.face_landmarks:
                return None

            # 转换为兼容的数据结构
            face_landmarks = self._convert_to_legacy_format(detection_result.face_landmarks[0])

            # 如果使用了裁剪，需要将关键点坐标转换回原图坐标系
            if bbox is not None:
                face_landmarks = self._convert_landmarks(face_landmarks, bbox, img.shape)

            return face_landmarks

        except Exception as e:
            logger.error(f"人脸关键点检测失败: {str(e)}")
            return None

    def _convert_to_legacy_format(self, landmarks: List) -> Any:
        """将新格式的关键点转换为兼容旧 API 的格式
        
        Args:
            landmarks: 新格式的关键点列表
            
        Returns:
            Any: 兼容旧格式的关键点对象
        """
        class NormalizedLandmark:
            def __init__(self, x, y, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        class NormalizedLandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks

        converted_landmarks = [NormalizedLandmark(lm.x, lm.y, lm.z) for lm in landmarks]
        return NormalizedLandmarkList(converted_landmarks)

    def get_eye_landmarks(self, face_landmarks: Any) -> Optional[np.ndarray]:
        """获取眼睛关键点

        Args:
            face_landmarks: MediaPipe 检测到的面部关键点

        Returns:
            Optional[np.ndarray]: 眼睛关键点坐标数组
        """
        try:
            # 使用更精确的眼睛关键点
            # 左眼外角(33)、内角(133)，右眼内角(362)、外角(263)
            left_eye_outer = np.array(
                [face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]
            )
            left_eye_inner = np.array(
                [face_landmarks.landmark[133].x, face_landmarks.landmark[133].y]
            )
            right_eye_inner = np.array(
                [face_landmarks.landmark[362].x, face_landmarks.landmark[362].y]
            )
            right_eye_outer = np.array(
                [face_landmarks.landmark[263].x, face_landmarks.landmark[263].y]
            )

            # 计算左右眼中心点
            left_eye = (left_eye_outer + left_eye_inner) / 2
            right_eye = (right_eye_outer + right_eye_inner) / 2

            return np.array([left_eye, right_eye])

        except Exception as e:
            logger.error(f"眼睛关键点提取失败: {str(e)}")
            return None

    def _convert_landmarks(
        self, face_landmarks: Any, bbox: Tuple[float, float, float, float], img_shape: tuple
    ) -> Any:
        """将裁剪图上的面部关键点坐标转换到原图坐标系

        Args:
            face_landmarks: MediaPipe 检测到的面部关键点
            bbox: 边界框坐标 (x1, y1, x2, y2)
            img_shape: 原图的形状 (height, width)

        Returns:
            Any: 转换后的面部关键点
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            crop_width = x2 - x1
            crop_height = y2 - y1

            # 获取原图的宽高
            img_height, img_width = img_shape[:2]

            # 创建新的关键点列表
            converted_landmarks_list = []

            # 复制并转换所有关键点
            for i, landmark in enumerate(face_landmarks.landmark):
                # 将相对坐标（0-1）转换为原图坐标系
                new_x = (landmark.x * crop_width + x1) / img_width
                new_y = (landmark.y * crop_height + y1) / img_height
                new_z = landmark.z
                converted_landmarks_list.append(type(landmark)(new_x, new_y, new_z))

            # 创建新的 NormalizedLandmarkList
            converted_landmarks = type(face_landmarks)(converted_landmarks_list)

            return converted_landmarks

        except Exception as e:
            logger.error(f"关键点坐标转换失败: {str(e)}")
            return None

    def draw_face_mesh(
        self,
        image: np.ndarray,
        face_landmarks: Any,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """绘制人脸网格

        Args:
            image: 输入图像
            face_landmarks: MediaPipe 检测到的面部关键点
            color: BGR颜色元组

        Returns:
            np.ndarray: 绘制后的图像
        """
        if face_landmarks is None:
            return image

        # 创建图像副本并确保是BGR格式
        if image.shape[2] == 4:  # RGBA格式
            img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            img = image.copy()

        # 绘制关键点
        h, w = img.shape[:2]
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img, (x, y), 1, color, -1)

        # 绘制一些关键连接线（简化版本）
        # 眼睛轮廓
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        for i in range(len(left_eye_indices) - 1):
            idx1 = left_eye_indices[i]
            idx2 = left_eye_indices[i + 1]
            if idx1 < len(face_landmarks.landmark) and idx2 < len(face_landmarks.landmark):
                x1 = int(face_landmarks.landmark[idx1].x * w)
                y1 = int(face_landmarks.landmark[idx1].y * h)
                x2 = int(face_landmarks.landmark[idx2].x * w)
                y2 = int(face_landmarks.landmark[idx2].y * h)
                cv2.line(img, (x1, y1), (x2, y2), color, 1)
        
        for i in range(len(right_eye_indices) - 1):
            idx1 = right_eye_indices[i]
            idx2 = right_eye_indices[i + 1]
            if idx1 < len(face_landmarks.landmark) and idx2 < len(face_landmarks.landmark):
                x1 = int(face_landmarks.landmark[idx1].x * w)
                y1 = int(face_landmarks.landmark[idx1].y * h)
                x2 = int(face_landmarks.landmark[idx2].x * w)
                y2 = int(face_landmarks.landmark[idx2].y * h)
                cv2.line(img, (x1, y1), (x2, y2), color, 1)

        return img