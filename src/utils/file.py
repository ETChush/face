"""文件操作工具模块

提供文件处理相关的工具函数。
"""

import os
import re
from typing import List
from ..utils.logging import get_logger


def get_sorted_image_files(input_dir: str) -> List[str]:
    """获取排序后的图片文件列表

    排序规则：
    1. 文件名以数字开头的按照数字升序排序
    2. 其他文件按照字节序排序
    3. 过滤掉隐藏文件（以.开头的文件）
    4. 确保文件是真实的图片文件

    Args:
        input_dir: 输入目录路径

    Returns:
        List[str]: 排序后的图片文件列表
    """
    logger = get_logger(__name__)
    
    # 获取所有图片文件，过滤掉隐藏文件
    img_files = []
    for f in os.listdir(input_dir):
        # 跳过隐藏文件
        if f.startswith("."):
            logger.debug(f"跳过隐藏文件: {f}")
            continue
        
        # 检查文件扩展名
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            logger.debug(f"跳过非图片文件: {f}")
            continue
            
        # 检查是否为真实文件（不是目录或链接）
        full_path = os.path.join(input_dir, f)
        if not os.path.isfile(full_path):
            logger.debug(f"跳过非文件: {f}")
            continue
            
        img_files.append(f)
        logger.debug(f"添加图片文件: {f}")

    # 定义排序键函数
    def sort_key(filename: str) -> tuple:
        # 提取文件名开头的数字部分
        match = re.match(r"^(\d+)", filename)
        if match:
            # 如果文件名以数字开头，按数字排序
            return (0, int(match.group(1)))
        else:
            # 否则按字节序排序
            return (1, filename)

    # 排序并返回
    sorted_files = sorted(img_files, key=sort_key)
    logger.debug(f"排序后的文件列表: {sorted_files}")
    return sorted_files


def ensure_directory(directory: str) -> None:
    """确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径
    """
    os.makedirs(directory, exist_ok=True) 