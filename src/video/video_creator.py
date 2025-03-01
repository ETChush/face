"""视频生成模块

提供视频生成相关功能。
"""

import os
import cv2
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from ..utils.logging import get_logger
from ..utils.image import read_image
from ..utils.file import get_sorted_image_files
from .morphing import ImageMorpher

logger = get_logger(__name__)


class LayeredVideoCreator:
    """透明叠放视频生成器"""

    def create_video(
        self,
        output_dir: str,
        fps: int = 30,
        transition_frames: int = 30,
        debug: bool = False
    ) -> None:
        """生成叠放效果的视频

        将处理后的图片按顺序叠放，生成平滑过渡的视频。

        Args:
            output_dir: 输出目录，包含已处理的图片
            fps: 视频帧率，默认30
            transition_frames: 每张图片过渡的帧数，默认30
            debug: 是否启用调试模式
        """
        try:
            # 获取所有已处理的PNG图片
            processed_images = sorted([f for f in os.listdir(output_dir) 
                                    if f.endswith('.png') and not f.startswith('marked_')])
            
            if not processed_images:
                raise ValueError("未找到处理后的图片")

            if debug:
                logger.debug(f"找到 {len(processed_images)} 张处理后的图片")

            # 读取第一张图片获取尺寸
            first_img = cv2.imread(os.path.join(output_dir, processed_images[0]), cv2.IMREAD_UNCHANGED)
            if first_img is None:
                raise ValueError("无法读取第一张图片")

            height, width = first_img.shape[:2]
            
            # 创建视频写入器
            video_path = os.path.join(output_dir, "layered_transition.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # 初始化画布（黑色背景）
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            if debug:
                logger.debug("开始生成视频")

            # 计算总帧数
            total_frames = len(processed_images) * transition_frames + fps * 2  # 包括最后2秒的停留时间
            
            with tqdm(total=total_frames, desc="生成视频", unit="帧") as pbar:
                # 逐张图片处理
                for i, img_name in enumerate(processed_images):
                    img_path = os.path.join(output_dir, img_name)
                    current_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    
                    if current_img is None:
                        logger.warning(f"跳过无法读取的图片: {img_name}")
                        pbar.update(transition_frames)  # 更新进度条
                        continue

                    # 确保图片有透明通道
                    if current_img.shape[2] != 4:
                        logger.warning(f"图片 {img_name} 没有透明通道，跳过")
                        pbar.update(transition_frames)  # 更新进度条
                        continue

                    # 分离RGB和透明通道
                    rgb = current_img[:, :, :3]
                    alpha = current_img[:, :, 3]

                    # 生成过渡帧
                    for frame in range(transition_frames):
                        # 计算当前帧的透明度
                        opacity = frame / transition_frames
                        
                        # 创建当前帧的透明遮罩
                        current_alpha = (alpha.astype(float) * opacity / 255)
                        current_alpha = np.stack([current_alpha] * 3, axis=2)
                        
                        # 混合图像
                        blended = canvas.copy()
                        mask = current_alpha > 0
                        blended[mask] = (blended[mask] * (1 - current_alpha[mask]) + 
                                       rgb[mask] * current_alpha[mask])
                        
                        # 写入帧
                        video_writer.write(blended.astype(np.uint8))
                        pbar.update(1)  # 更新进度条
                    
                    # 更新画布
                    canvas = blended.copy()

                    if debug:
                        logger.debug(f"已处理图片 {i+1}/{len(processed_images)}")

                # 在最后保持几秒
                for _ in range(fps * 2):  # 保持2秒
                    video_writer.write(canvas.astype(np.uint8))
                    pbar.update(1)  # 更新进度条

            # 释放资源
            video_writer.release()

            if debug:
                logger.debug(f"视频已生成: {video_path}")

        except Exception as e:
            logger.error(f"生成视频失败: {str(e)}")
            raise


class VideoCreator:
    """视频生成器"""

    def __init__(self):
        """初始化视频生成器"""
        self.morpher = ImageMorpher()

    def create_video(
        self,
        output_dir: str,
        fps: int = 24,
        debug: bool = False,
        transition_time: float = 0.5,
        stay_time: float = 0.25,
    ) -> None:
        """从对齐的图片创建视频，添加变形效果

        Args:
            output_dir: 输出目录路径
            fps: 视频帧率，默认24fps
            debug: 是否启用调试模式
            transition_time: 过渡时间（秒），默认0.5秒
            stay_time: 停留时间（秒），默认0.25秒
        """
        try:
            # 获取所有处理后的图片
            image_files = get_sorted_image_files(output_dir)
            image_files = [
                f for f in image_files if not f.startswith("marked_") and not f.startswith("base_")
            ]

            if not image_files:
                logger.error("没有找到处理后的图片")
                return

            if debug:
                logger.debug(f"找到 {len(image_files)} 张图片")

            # 读取所有图片
            images = []
            for img_file in image_files:
                img_path = os.path.join(output_dir, img_file)
                img = read_image(img_path)
                if img is None:
                    logger.warning(f"无法读取图片: {img_file}")
                    continue
                images.append(img.astype(np.float32))

            if len(images) < 2:
                logger.error("需要至少2张有效图片来创建视频")
                return

            height, width = images[0].shape[:2]
            if debug:
                logger.debug(f"图片尺寸: {width}x{height}")

            # 创建视频写入器
            video_path = os.path.join(output_dir, "morphing.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            if not out.isOpened():
                logger.error("无法创建视频文件")
                return

            # 减少特征点数量，只保留关键点
            points = np.array(
                [
                    [0, 0],
                    [width - 1, 0],
                    [0, height - 1],
                    [width - 1, height - 1],  # 四个角点
                    [width // 3, height // 3],
                    [2 * width // 3, height // 3],  # 左右眼位置
                    [width // 2, 2 * height // 3],  # 嘴巴位置
                ],
                dtype=np.float32,
            )

            try:
                # 获取Delaunay三角剖分
                rect = (0, 0, width, height)
                triangles = self.morpher.get_delaunay_triangles(rect, points)

                if triangles is None or len(triangles) == 0:
                    logger.error("三角剖分失败")
                    return

                if debug:
                    logger.debug(f"生成了 {len(triangles)} 个三角形")

                # 预计算所有三角形的mask
                triangle_masks = []
                for triangle in triangles:
                    try:
                        r1 = cv2.boundingRect(triangle)
                        t_offset = triangle - np.array([[r1[0], r1[1]]])

                        mask = np.zeros((r1[3], r1[2]), dtype=np.float32)
                        cv2.fillConvexPoly(mask, t_offset.astype(np.int32), 1.0, cv2.LINE_AA)
                        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                        triangle_masks.append((mask, r1))
                    except Exception as e:
                        if debug:
                            logger.warning(f"预计算三角形mask失败: {str(e)}")
                        continue

                # 计算帧数
                frames_per_transition = int(transition_time * fps)
                frames_per_stay = int(stay_time * fps)

                if debug:
                    logger.debug(f"过渡帧数: {frames_per_transition}")
                    logger.debug(f"停留帧数: {frames_per_stay}")

                with tqdm(total=len(images) - 1, desc="生成视频") as pbar:
                    for i in range(len(images) - 1):
                        img1 = images[i]
                        img2 = images[i + 1]

                        # 创建过渡帧
                        for j in tqdm(
                            range(frames_per_transition),
                            desc=f"帧 {i+1}/{len(images)-1}",
                            leave=False,
                        ):
                            try:
                                alpha = j / frames_per_transition
                                morphed = np.zeros_like(img1)
                                total_mask = np.zeros_like(img1)

                                # 对每个三角形进行变形
                                for k, triangle in enumerate(triangles):
                                    try:
                                        t1 = triangle
                                        t2 = triangle  # 静态变形，三角形位置不变
                                        t = t1  # 不需要计算中间位置

                                        mask, r1 = triangle_masks[k]
                                        morphed_triangle, _, _ = self.morpher.morph_triangle(
                                            img1, img2, t1, t2, t, alpha
                                        )

                                        # 累积变形结果和mask
                                        morphed[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]] += (
                                            morphed_triangle * mask
                                        )
                                        total_mask[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]] += mask

                                    except Exception as e:
                                        if debug:
                                            logger.warning(f"处理三角形 {k} 失败: {str(e)}")
                                        continue

                                # 标准化结果
                                total_mask[total_mask == 0] = 1
                                morphed = morphed / total_mask

                                # 写入帧
                                out.write(morphed.astype(np.uint8))

                            except Exception as e:
                                if debug:
                                    logger.warning(f"处理过渡帧 {j} 失败: {str(e)}")
                                # 在出错时，直接使用目标图片
                                out.write(img2.astype(np.uint8))

                        # 写入目标图片
                        for _ in range(frames_per_stay):
                            out.write(img2.astype(np.uint8))

                        pbar.update(1)

            finally:
                # 释放视频写入器
                out.release()
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    logger.info(f"视频已保存至: {video_path}")
                else:
                    logger.error("视频生成失败")

        except Exception as e:
            logger.error(f"创建视频失败: {str(e)}")
            raise 