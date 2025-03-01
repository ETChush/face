"""视频变形效果模块

提供图像变形相关功能。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ImageMorpher:
    """图像变形器"""

    def __init__(self):
        """初始化图像变形器"""
        pass

    def get_delaunay_triangles(
        self, rect: Tuple[int, int, int, int], points: np.ndarray
    ) -> Optional[np.ndarray]:
        """获取Delaunay三角剖分

        Args:
            rect: 矩形区域 (x, y, width, height)
            points: 关键点坐标数组

        Returns:
            Optional[np.ndarray]: 三角形顶点坐标数组，形状为 (n, 3, 2)
        """
        try:
            subdiv = cv2.Subdiv2D(rect)
            for p in points:
                subdiv.insert((int(p[0]), int(p[1])))

            triangles = subdiv.getTriangleList()
            # 将三角形顶点坐标转换为 (3, 2) 形状的数组
            triangles = triangles.reshape(-1, 3, 2)
            return triangles

        except Exception as e:
            logger.error(f"三角剖分失败: {str(e)}")
            return None

    def morph_triangle(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        t: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """对单个三角形进行变形

        Args:
            img1: 第一张图片
            img2: 第二张图片
            t1: 第一张图片上的三角形顶点
            t2: 第二张图片上的三角形顶点
            t: 目标三角形顶点
            alpha: 混合系数 (0-1)

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple]: (变形后的三角形区域, mask, 边界矩形)
        """
        try:
            # 获取三角形的边界矩形
            r1 = cv2.boundingRect(t1)
            r2 = cv2.boundingRect(t2)

            # 检查边界矩形的有效性
            if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
                # 返回空结果
                return (
                    np.zeros((max(r1[3], 1), max(r1[2], 1), 3), dtype=np.float32),
                    np.zeros((max(r1[3], 1), max(r1[2], 1), 3), dtype=np.float32),
                    r1,
                )

            # 偏移三角形顶点
            t1_offset = t1 - np.array([[r1[0], r1[1]]])
            t2_offset = t2 - np.array([[r2[0], r2[1]]])
            t_offset = t - np.array([[r1[0], r1[1]]])

            # 填充三角形mask，使用抗锯齿
            mask = np.zeros((r1[3], r1[2]), dtype=np.float32)
            cv2.fillConvexPoly(mask, t_offset.astype(np.int32), 1.0, cv2.LINE_AA)

            # 扩展mask到3通道
            mask = mask[:, :, np.newaxis]
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # 提取矩形区域
            img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]].astype(np.float32)
            img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]].astype(np.float32)

            # 检查提取的区域是否有效
            if img1_rect.size == 0 or img2_rect.size == 0:
                # 返回空结果
                return (
                    np.zeros((max(r1[3], 1), max(r1[2], 1), 3), dtype=np.float32),
                    np.zeros((max(r1[3], 1), max(r1[2], 1), 3), dtype=np.float32),
                    r1,
                )

            size = (r1[2], r1[3])

            # 计算仿射变换矩阵
            warp_mat1 = cv2.getAffineTransform(
                t1_offset.astype(np.float32), t_offset.astype(np.float32)
            )
            warp_mat2 = cv2.getAffineTransform(
                t2_offset.astype(np.float32), t_offset.astype(np.float32)
            )

            # 应用仿射变换
            try:
                img1_warped = cv2.warpAffine(
                    img1_rect,
                    warp_mat1,
                    size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                img2_warped = cv2.warpAffine(
                    img2_rect,
                    warp_mat2,
                    size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            except cv2.error:
                # 如果变换失败，返回空结果
                return (
                    np.zeros((max(r1[3], 1), max(r1[2], 1), 3), dtype=np.float32),
                    np.zeros((max(r1[3], 1), max(r1[2], 1), 3), dtype=np.float32),
                    r1,
                )

            # 混合两个图片
            img_morphed = (1.0 - alpha) * img1_warped + alpha * img2_warped

            return img_morphed, mask, r1

        except Exception as e:
            logger.warning(f"三角形变形失败: {str(e)}")
            # 返回空结果
            return (
                np.zeros((max(r1[3], 1), max(r1[2], 1), 3), dtype=np.float32),
                np.zeros((max(r1[3], 1), max(r1[2], 1), 3), dtype=np.float32),
                r1,
            ) 