"""MediaPipe 模型模块

提供 MediaPipe 人脸关键点检测功能。
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class MediaPipeFaceDetector:
    """MediaPipe 人脸关键点检测器"""

    def __init__(self):
        """初始化 MediaPipe 人脸关键点检测器"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.1,
        )
        # 添加绘制工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

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
        try:
            # 如果提供了边界框，先裁剪图片
            face_img = img
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                face_img = img[y1:y2, x1:x2]

            # 转换为RGB格式
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 检测关键点
            results = self.face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                return None

            # 如果使用了裁剪，需要将关键点坐标转换回原图坐标系
            if bbox is not None:
                return self._convert_landmarks(
                    results.multi_face_landmarks[0], bbox, img.shape
                )

            return results.multi_face_landmarks[0]

        except Exception as e:
            logger.error(f"人脸关键点检测失败: {str(e)}")
            return None

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
            converted_landmarks = type(face_landmarks)()

            # 复制并转换所有关键点
            for i, landmark in enumerate(face_landmarks.landmark):
                new_landmark = type(landmark)()
                # 将相对坐标（0-1）转换为原图坐标系
                new_landmark.x = (landmark.x * crop_width + x1) / img_width
                new_landmark.y = (landmark.y * crop_height + y1) / img_height
                new_landmark.z = landmark.z
                converted_landmarks.landmark.append(new_landmark)

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

        # 将相对坐标转换为像素坐标
        drawing_spec = self.mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1)
        
        # 绘制人脸网格
        self.mp_drawing.draw_landmarks(
            image=img,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_spec
        )

        return img 