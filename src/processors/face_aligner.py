"""人脸对齐模块

提供图像对齐相关功能。
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

class FaceAligner:
    """人脸对齐器"""

    def __init__(self):
        """初始化人脸对齐器"""
        pass

    def align_image(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        target_eye_distance: Optional[float] = None,
        debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对齐图像

        Args:
            img: 输入图像
            keypoints: 眼睛关键点坐标 [[left_x, left_y], [right_x, right_y]]
            target_size: 目标尺寸 (height, width)
            target_eye_distance: 目标眼睛间距
            debug: 是否启用调试模式

        Returns:
            Tuple[np.ndarray, np.ndarray]: (对齐后的图像, 变换后的关键点)
        """
        h, w = img.shape[:2]
        if debug:
            logger.debug(f"输入图像尺寸: {w}x{h}")
            logger.debug(f"输入关键点坐标: {keypoints}")
        
        # 确保关键点是浮点数类型
        keypoints = keypoints.astype(np.float32)
        
        # 计算眼睛中心点和角度
        left_eye, right_eye = keypoints
        eye_center = (left_eye + right_eye) / 2
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = -np.degrees(np.arctan2(dy, dx))
        
        if debug:
            logger.debug(f"左眼坐标: {left_eye}")
            logger.debug(f"右眼坐标: {right_eye}")
            logger.debug(f"眼睛中心点: {eye_center}")
            logger.debug(f"需要旋转角度: {angle:.2f}度")
        
        # 计算当前眼睛间距
        current_eye_distance = np.linalg.norm(right_eye - left_eye)
        if debug:
            logger.debug(f"当前眼睛间距: {current_eye_distance:.2f}像素")
        
        # 计算旋转后的图像尺寸
        cos_angle = np.abs(np.cos(np.radians(angle)))
        sin_angle = np.abs(np.sin(np.radians(angle)))
        rotated_w = int(h * sin_angle + w * cos_angle)
        rotated_h = int(h * cos_angle + w * sin_angle)
        
        if debug:
            logger.debug(f"旋转后的图像尺寸: {rotated_w}x{rotated_h}")
        
        # 根据目标参数计算缩放比例
        if target_eye_distance is not None:
            # 使用目标眼睛间距计算缩放比例
            scale = target_eye_distance / current_eye_distance
            if debug:
                logger.debug(f"目标眼睛间距: {target_eye_distance:.2f}像素")
                logger.debug(f"根据眼睛间距计算的缩放比例: {scale:.2f}")
        elif target_size is not None:
            # 使用目标尺寸计算缩放比例
            target_h, target_w = target_size
            scale_h = target_h / rotated_h
            scale_w = target_w / rotated_w
            scale = min(scale_h, scale_w)
            if debug:
                logger.debug(f"目标尺寸: {target_w}x{target_h}")
                logger.debug(f"水平缩放比例: {scale_w:.2f}")
                logger.debug(f"垂直缩放比例: {scale_h:.2f}")
                logger.debug(f"最终选择的缩放比例: {scale:.2f}")
        else:
            scale = 1.0
            if debug:
                logger.debug("未指定目标参数，使用原始尺寸")
            
        # 计算目标尺寸
        if target_size is not None:
            target_h, target_w = target_size
            final_w, final_h = target_w, target_h
        else:
            final_w = int(rotated_w * scale)
            final_h = int(rotated_h * scale)
            
        if debug:
            logger.debug(f"最终输出尺寸: {final_w}x{final_h}")
            
        # 计算旋转和缩放后眼睛中心的位置
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 计算旋转和缩放后眼睛中心的新位置
        rotated_x = eye_center[0] * cos_a - eye_center[1] * sin_a
        rotated_y = eye_center[0] * sin_a + eye_center[1] * cos_a
        scaled_x = rotated_x * scale
        scaled_y = rotated_y * scale
        
        # 计算旋转和缩放后左右眼睛的位置
        rotated_left_x = left_eye[0] * cos_a - left_eye[1] * sin_a
        rotated_left_y = left_eye[0] * sin_a + left_eye[1] * cos_a
        rotated_right_x = right_eye[0] * cos_a - right_eye[1] * sin_a
        rotated_right_y = right_eye[0] * sin_a + right_eye[1] * cos_a
        
        scaled_left_x = rotated_left_x * scale
        scaled_left_y = rotated_left_y * scale
        scaled_right_x = rotated_right_x * scale
        scaled_right_y = rotated_right_y * scale
        
        if debug:
            logger.debug(f"旋转后的左眼坐标: ({rotated_left_x:.2f}, {rotated_left_y:.2f})")
            logger.debug(f"旋转后的右眼坐标: ({rotated_right_x:.2f}, {rotated_right_y:.2f})")
            logger.debug(f"旋转后的眼睛中心位置: ({rotated_x:.2f}, {rotated_y:.2f})")
            logger.debug(f"缩放后的左眼坐标: ({scaled_left_x:.2f}, {scaled_left_y:.2f})")
            logger.debug(f"缩放后的右眼坐标: ({scaled_right_x:.2f}, {scaled_right_y:.2f})")
            logger.debug(f"缩放后的眼睛中心位置: ({scaled_x:.2f}, {scaled_y:.2f})")
            # 计算旋转和缩放后的眼睛间距
            rotated_eye_distance = np.sqrt((rotated_right_x - rotated_left_x)**2 + 
                                         (rotated_right_y - rotated_left_y)**2)
            scaled_eye_distance = rotated_eye_distance * scale
            logger.debug(f"旋转后的眼睛间距: {rotated_eye_distance:.2f}像素")
            logger.debug(f"缩放后的眼睛间距: {scaled_eye_distance:.2f}像素")
        
        # 计算需要的平移量，使眼睛中心位于目标图像中心
        tx = final_w / 2 - scaled_x
        ty = final_h / 2 - scaled_y
        
        if debug:
            logger.debug(f"平移量: dx={tx:.2f}, dy={ty:.2f}")
        
        # 创建变换矩阵
        rotation_matrix = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a, scale * cos_a, ty]
        ], dtype=np.float32)
        
        if debug:
            logger.debug(f"变换矩阵:\n{rotation_matrix}")
        
        # 将输入图像转换为4通道（带透明通道）
        if img.shape[2] == 3:
            # 创建4通道图像，alpha通道设为完全不透明
            img_rgba = np.dstack((img, np.full(img.shape[:2], 255, dtype=np.uint8)))
            if debug:
                logger.debug("将3通道图像转换为4通道RGBA格式")
        else:
            img_rgba = img
            if debug:
                logger.debug("输入图像已经是4通道RGBA格式")
            
        # 应用仿射变换，使用 BORDER_TRANSPARENT 确保透明区域正确处理
        transformed = cv2.warpAffine(
            img_rgba,
            rotation_matrix,
            (final_w, final_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_TRANSPARENT,
            dst=np.zeros((final_h, final_w, 4), dtype=np.uint8)  # 每次创建新的透明背景
        )
        
        # 变换关键点
        ones = np.ones(shape=(len(keypoints), 1), dtype=np.float32)
        points_ones = np.hstack([keypoints, ones])
        transformed_points = rotation_matrix.dot(points_ones.T).T
        
        if debug:
            logger.debug(f"变换后图像尺寸: {transformed.shape}")
            logger.debug(f"变换后关键点坐标: {transformed_points}")
            logger.debug(f"变换后眼睛中心位置: {transformed_points.mean(axis=0)}")
            # 计算变换后的眼睛间距，用于验证
            transformed_eye_distance = np.linalg.norm(transformed_points[1] - transformed_points[0])
            logger.debug(f"变换后眼睛间距: {transformed_eye_distance:.2f}像素")
            if target_eye_distance is not None:
                error = abs(transformed_eye_distance - target_eye_distance)
                logger.debug(f"与目标眼睛间距的误差: {error:.2f}像素")
        
        return transformed, transformed_points 