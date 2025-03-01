"""图像处理工具模块

提供图像处理相关的工具函数。
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def read_image(image_path: str) -> Optional[np.ndarray]:
    """读取图像文件

    Args:
        image_path: 图像文件路径

    Returns:
        Optional[np.ndarray]: 读取的图像数组，读取失败返回None
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    return img


def save_image(image: np.ndarray, save_path: str) -> bool:
    """保存图像文件

    Args:
        image: 图像数组，可以是3通道(BGR)或4通道(BGRA)
        save_path: 保存路径

    Returns:
        bool: 是否保存成功
    """
    # 获取文件扩展名
    ext = save_path.lower().split('.')[-1]
    
    # 根据文件扩展名和图像通道数选择保存参数
    if ext == 'png':
        # PNG格式支持透明通道
        if image.shape[2] == 3:
            # 如果是3通道图像，添加透明通道
            image = np.dstack((image, np.full(image.shape[:2], 255, dtype=np.uint8)))
        # 使用较低的PNG压缩级别以提高速度
        return cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        # 其他格式（jpg等）
        if image.shape[2] == 4:
            # 如果是4通道图像，只保存RGB通道
            image = image[:, :, :3]
        return cv2.imwrite(save_path, image)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """调整图像大小

    Args:
        image: 输入图像
        size: 目标尺寸 (width, height)

    Returns:
        np.ndarray: 调整大小后的图像
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


def draw_landmarks(
    image: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
    thickness: int = -1,
) -> np.ndarray:
    """在图像上绘制关键点

    Args:
        image: 输入图像
        points: 关键点坐标数组
        color: BGR颜色元组
        radius: 点的半径
        thickness: 线条粗细，-1表示填充

    Returns:
        np.ndarray: 绘制后的图像
    """
    # 如果是RGBA图像，转换为BGR
    if image.shape[2] == 4:
        img = image[:, :, :3].copy()
    else:
        img = image.copy()
        
    # 确保关键点坐标在图像范围内
    points = np.clip(points, [0, 0], [img.shape[1] - 1, img.shape[0] - 1])
    # 转换为整数坐标
    points = points.round().astype(np.int32)
        
    for point in points:
        cv2.circle(img, tuple(point), radius, color, thickness)
        
    # 如果原图是RGBA，转换回RGBA
    if image.shape[2] == 4:
        img = np.dstack((img, image[:, :, 3]))
    return img


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """在图像上绘制边界框

    Args:
        image: 输入图像
        bbox: 边界框坐标 (x1, y1, x2, y2)
        color: BGR颜色元组
        thickness: 线条粗细
        label: 可选的标签文本

    Returns:
        np.ndarray: 绘制后的图像
    """
    # 如果是RGBA图像，转换为BGR
    if image.shape[2] == 4:
        img = image[:, :, :3].copy()
    else:
        img = image.copy()
        
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        # 添加标签文本
        font_scale = 0.6
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 获取文本大小
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # 绘制文本背景
        cv2.rectangle(
            img,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1,
        )
        
        # 绘制文本
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
    
    # 如果原图是RGBA，转换回RGBA
    if image.shape[2] == 4:
        img = np.dstack((img, image[:, :, 3]))
    return img 