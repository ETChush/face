"""人脸处理主流程模块

提供基准图片处理和后续图片处理功能。
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

from ..utils.logging import get_logger
from ..utils.image import read_image, save_image, draw_landmarks, draw_bbox
from ..models.yolo import YOLOFaceDetector
from ..models.arcface import ArcFaceExtractor
from ..models.mediapipe import MediaPipeFaceDetector
from ..processors.face_aligner import FaceAligner
from ..utils.file import get_sorted_image_files
from ..video.video_creator import LayeredVideoCreator

logger = get_logger(__name__)


class FaceProcessor:
    """人脸处理器"""

    def __init__(self):
        """初始化人脸处理器"""
        self.yolo_detector = YOLOFaceDetector()
        self.arcface_extractor = ArcFaceExtractor()
        self.mediapipe_detector = MediaPipeFaceDetector()
        self.face_aligner = FaceAligner()
        self.video_creator = LayeredVideoCreator()

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个向量的余弦相似度

        Args:
            v1: 第一个向量
            v2: 第二个向量

        Returns:
            float: 余弦相似度，范围[-1, 1]
        """
        # 确保向量是一维的
        if v1.ndim > 1 and v1.shape[0] == 1:
            v1 = v1.flatten()
        if v2.ndim > 1 and v2.shape[0] == 1:
            v2 = v2.flatten()

        # 计算余弦相似度
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def process_base_image(
        self, img_path: str, output_dir: str, debug_dir: str, debug: bool, 
        base_size: Optional[Tuple[int, int]] = (1080, 1920),
        add_border: bool = True
    ) -> Dict[str, Any]:
        """处理基准图片

        处理流程：
        1. 使用YOLOv8检测人脸，取第一张人脸
        2. 使用MediaPipe提取人脸关键点
        3. 按照关键点将人脸调整至眼睛水平，嘴巴在下的正常状态
        4. 剪切缩放人脸为112x112大小方便ArcFace处理
        5. 使用InsightFace获取人脸的嵌入向量作为后续比较的基准
        6. 以眼睛为中心，调整整个图片的角度和大小，保留原图部分
        7. 将图片调整为指定尺寸（默认1920x1080）
        8. 可选添加白边（默认开启）

        Args:
            img_path: 图片路径
            output_dir: 输出目录
            debug_dir: 调试输出目录
            debug: 是否启用调试模式
            base_size: 基准图片的目标尺寸 (height, width)，默认为 (1080, 1920)
            add_border: 是否添加白边，默认为True

        Returns:
            Dict[str, Any]: 包含基准图片信息的字典
        """
        try:
            if debug:
                logger.debug(f"开始处理基准图片: {img_path}")

            # 使用tqdm创建处理步骤的进度条
            steps = [
                "读取图片",
                "添加白边",
                "检测人脸",
                "提取关键点",
                "提取特征",
                "对齐图片",
                "保存结果"
            ]
            
            with tqdm(total=len(steps), desc="处理基准图片", unit="步") as pbar:
                # 1. 读取图片
                img = read_image(img_path)
                if img is None:
                    raise ValueError(f"无法读取基准图片: {img_path}")
                pbar.update(1)
                pbar.set_description(f"处理基准图片 - {steps[1]}")

                # 2. 添加白边
                if add_border:
                    if debug:
                        logger.debug("添加白边")
                    
                    h, w = img.shape[:2]
                    border_width = int(min(h, w) * 0.02)
                    bordered_img = np.full((h + 2*border_width, w + 2*border_width, 3), 255, dtype=np.uint8)
                    bordered_img[border_width:border_width+h, border_width:border_width+w] = img
                    img = bordered_img
                pbar.update(1)
                pbar.set_description(f"处理基准图片 - {steps[2]}")

                # 3. 人脸检测
                has_face, face_img, bbox = self.yolo_detector.detect_face(img)
                if not has_face:
                    raise ValueError("基准图片未检测到人脸")
                pbar.update(1)
                pbar.set_description(f"处理基准图片 - {steps[3]}")

                # 4. 关键点检测
                face_landmarks = self.mediapipe_detector.get_face_landmarks(img, bbox)
                if face_landmarks is None:
                    raise ValueError("MediaPipe 未检测到人脸关键点")

                keypoints = self.mediapipe_detector.get_eye_landmarks(face_landmarks)
                if keypoints is None:
                    raise ValueError("无法获取眼睛关键点")

                h, w = img.shape[:2]
                keypoints = np.array([[p[0] * w, p[1] * h] for p in keypoints])
                pbar.update(1)
                pbar.set_description(f"处理基准图片 - {steps[4]}")

                # 5. 特征提取
                face_embedding, arcface_input = self.arcface_extractor.extract_features(face_img, debug)
                if np.all(face_embedding == 0):
                    raise ValueError("人脸特征提取失败")
                pbar.update(1)
                pbar.set_description(f"处理基准图片 - {steps[5]}")

                # 6. 图片对齐
                aligned_img, transformed_points = self.face_aligner.align_image(
                    img, keypoints, target_size=base_size, debug=debug
                )
                pbar.update(1)
                pbar.set_description(f"处理基准图片 - {steps[6]}")

                # 7. 保存结果
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                aligned_path = os.path.join(output_dir, f"{base_name}.png")
                save_image(aligned_img, aligned_path)

                if debug:
                    if img.shape[2] == 4:
                        marked_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    else:
                        marked_img = img.copy()
                    
                    marked_img = draw_bbox(marked_img, bbox, color=(0, 255, 0), thickness=3, label=f"Base Face {1}")
                    marked_img = draw_landmarks(marked_img, transformed_points)
                    marked_img = self.mediapipe_detector.draw_face_mesh(marked_img, face_landmarks)
                    marked_path = os.path.join(debug_dir, f"marked_base_{os.path.basename(img_path)}")
                    save_image(marked_img, marked_path)

                    if not np.all(face_embedding == 0):
                        arcface_path = os.path.join(debug_dir, f"base_arcface_{os.path.basename(img_path)}")
                        save_image(arcface_input, arcface_path)
                pbar.update(1)

            # 返回基准图片数据
            return {
                "keypoints": transformed_points,
                "face_embedding": face_embedding,
                "img_size": base_size,
                "eye_distance": np.linalg.norm(transformed_points[1] - transformed_points[0]),
                "original_img": img,
                "aligned_img": aligned_img,
            }

        except Exception as e:
            logger.error(f"处理基准图片失败: {str(e)}")
            raise

    def process_image(
        self,
        input_path: str,
        output_dir: str,
        debug_dir: str,
        base_data: Dict[str, Any],
        debug: bool,
        similarity_threshold: float = 0.5,
        add_border: bool = True,
    ) -> None:
        """处理单张图片

        处理流程：
        1. 使用YOLOv8检测所有人脸
        2. 对每个人脸进行关键点提取、对齐和特征提取
        3. 计算与基准人脸的相似度，选择最相似的人脸（相似度必须大于阈值）
        4. 以眼睛中心点为圆心，调整整个图片的角度和大小，与基准图片保持一致

        Args:
            input_path: 输入图片路径
            output_dir: 输出目录
            debug_dir: 调试输出目录
            base_data: 基准图片数据，包含人脸特征向量和目标尺寸
            debug: 是否启用调试模式
            similarity_threshold: 人脸相似度阈值，默认0.5
            add_border: 是否添加白边，默认为True
        """
        try:
            img = read_image(input_path)
            if img is None:
                raise ValueError("无法读取图片")

            # 添加白边（在所有处理之前）
            if add_border:
                if debug:
                    logger.debug("添加白边")
                
                # 计算边框宽度（使用原始图片尺寸的一定比例）
                h, w = img.shape[:2]
                border_width = int(min(h, w) * 0.02)  # 使用2%的边框宽度
                
                # 创建带白边的新画布
                bordered_img = np.full((h + 2*border_width, w + 2*border_width, 3), 255, dtype=np.uint8)
                
                # 放置主图片
                bordered_img[border_width:border_width+h, border_width:border_width+w] = img
                
                # 更新图片
                img = bordered_img

                if debug:
                    logger.debug(f"添加白边后的图片尺寸: {img.shape}")

            # 1. 使用 YOLOv8 进行人脸检测，获取所有人脸
            has_face, face_imgs, face_boxes = self.yolo_detector.detect_face(
                img, return_all=True
            )

            # 初始化变量
            face_results = []
            best_face = None

            if has_face:
                # 2. 对每个检测到的人脸进行处理
                for idx, (face_img, bbox) in enumerate(zip(face_imgs, face_boxes)):
                    try:
                        # 在裁剪的人脸上检测关键点
                        face_landmarks = self.mediapipe_detector.get_face_landmarks(
                            img, bbox
                        )

                        if face_landmarks is None:
                            if debug:
                                logger.debug(f"人脸 {idx + 1} 未检测到关键点，跳过")
                            continue

                        # 获取眼睛关键点
                        keypoints = self.mediapipe_detector.get_eye_landmarks(face_landmarks)
                        if keypoints is None:
                            continue

                        h, w = img.shape[:2]
                        keypoints = np.array([[p[0] * w, p[1] * h] for p in keypoints])

                        # 提取对齐后人脸的特征
                        face_embedding, arcface_input = self.arcface_extractor.extract_features(
                            face_img, debug
                        )

                        # 检查特征向量是否有效
                        if face_embedding is not None and not np.all(face_embedding == 0):
                            # 计算与基准图片的相似度
                            base_embedding = base_data["face_embedding"]
                            similarity = self._cosine_similarity(face_embedding, base_embedding)

                            # 保存结果
                            face_result = {
                                "similarity": similarity,
                                "bbox": bbox,
                                "index": idx,
                                "keypoints": keypoints,
                                "face_landmarks": face_landmarks,
                                "embedding": face_embedding,
                            }
                            face_results.append(face_result)

                            if debug:
                                logger.debug(f"人脸 {idx + 1} 余弦相似度: {similarity:.4f}")

                    except Exception as e:
                        logger.warning(f"处理人脸 {idx + 1} 时出错: {str(e)}")
                        continue

                if face_results:
                    # 3. 选择相似度最高的人脸
                    best_face = max(face_results, key=lambda x: x["similarity"])

            # 如果是调试模式且有调试目录，则生成并保存调试图片
            if debug and debug_dir is not None:
                marked_img = img.copy()

                # 绘制所有检测到的人脸框和信息
                if face_results:
                    for result in face_results:
                        face_idx = result["index"]
                        face_sim = result["similarity"]
                        face_bbox = result["bbox"]
                        is_best = result == best_face

                        # 使用不同颜色标记最相似的人脸和其他人脸
                        color = (0, 255, 0) if is_best else (0, 165, 255)  # 绿色为最相似，橙色为其他
                        thickness = 3 if is_best else 2

                        # 绘制边界框和标签
                        label = f"Face {face_idx + 1}: {face_sim:.4f}"
                        if is_best:
                            label += " (Best Match)"
                        marked_img = draw_bbox(
                            marked_img, face_bbox, color, thickness, label
                        )

                        # 如果是最佳匹配的人脸，绘制关键点和人脸网格
                        if is_best:
                            marked_img = draw_landmarks(
                                marked_img, result["keypoints"], color=(0, 0, 255)
                            )
                            marked_img = self.mediapipe_detector.draw_face_mesh(
                                marked_img, result["face_landmarks"]
                            )

                else:
                    # 如果没有检测到人脸，添加文字说明
                    cv2.putText(
                        marked_img,
                        "No faces detected",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )

                # 保存带标记的调试图片，保留原始格式
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                ext = os.path.splitext(input_path)[1]
                marked_path = os.path.join(debug_dir, f"marked_{os.path.basename(input_path)}")
                save_image(marked_img, marked_path)
                if debug:
                    logger.debug(f"调试图片已保存: {os.path.basename(marked_path)}")

                if face_results and best_face is not None:
                    # 保存ArcFace输入图像
                    arcface_path = os.path.join(
                        debug_dir, f"arcface_{os.path.basename(input_path)}"
                    )
                    save_image(arcface_input, arcface_path)
                    logger.debug(f"ArcFace输入图像已保存: {os.path.basename(arcface_path)}")

            # 如果找到了最佳匹配的人脸且相似度超过阈值，进行图片对齐
            if best_face is not None and best_face["similarity"] >= similarity_threshold:
                # 获取目标尺寸和眼睛间距
                TARGET_HEIGHT, TARGET_WIDTH = base_data["img_size"]
                target_eye_distance = base_data["eye_distance"]

                # 整图对齐
                aligned_img, transformed_points = self.face_aligner.align_image(
                    img, 
                    best_face["keypoints"], 
                    target_size=(TARGET_HEIGHT, TARGET_WIDTH),
                    target_eye_distance=target_eye_distance,
                    debug=debug
                )

                # 保存对齐后的图片，使用PNG格式
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.png")
                save_image(aligned_img, output_path)
            else:
                if debug:
                    if best_face is None:
                        logger.warning(f"未找到有效的人脸匹配: {os.path.basename(input_path)}")
                    else:
                        logger.warning(
                            f"最佳匹配相似度 ({best_face['similarity']:.4f}) 低于阈值 ({similarity_threshold})"
                        )

        except Exception as e:
            logger.error(f"处理失败: {os.path.basename(input_path)} - {str(e)}")
            raise

    def process_images(
        self,
        input_dir: str,
        output_dir: str,
        debug: bool = False,
        similarity_threshold: float = 0.5,
        base_size: Optional[Tuple[int, int]] = (1080, 1920),  # 默认尺寸为 1920x1080
        add_border: bool = True,  # 是否添加白边
        create_video: bool = True,  # 是否生成视频
        fps: int = 30,  # 视频帧率
        transition_frames: int = 30,  # 过渡帧数
    ) -> None:
        """处理目录中的所有图片

        Args:
            input_dir: 输入图片目录
            output_dir: 输出目录
            debug: 是否启用调试信息
            similarity_threshold: 人脸相似度阈值
            base_size: 基准图片的目标尺寸 (height, width)，默认为 (1080, 1920)
            add_border: 是否添加白边，默认为True
            create_video: 是否生成视频，默认为True
            fps: 视频帧率，默认30
            transition_frames: 过渡帧数，默认30
        """
        # 创建调试目录
        debug_dir = os.path.join(output_dir, "debug") if debug else None
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        # 获取并处理基准图片
        img_files = get_sorted_image_files(input_dir)
        if debug:
            logger.debug(f"找到以下图片文件：")
            for img_file in img_files:
                logger.debug(f"  - {img_file}")

        if not img_files:
            raise ValueError("未找到支持的图片文件")

        base_img_path = os.path.join(input_dir, img_files[0])
        logger.info(f"处理基准图片: {os.path.basename(base_img_path)}")
            
        try:
            base_data = self.process_base_image(
                base_img_path, 
                output_dir, 
                debug_dir, 
                debug, 
                base_size,
                add_border
            )
        except Exception as e:
            logger.error(f"处理基准图片时出错: {str(e)}")
            raise

        # 处理其余图片
        remaining_files = img_files[1:]
        logger.info(f"开始处理其余 {len(remaining_files)} 张图片...")
        
        with tqdm(total=len(remaining_files), desc="处理图片", unit="张") as pbar:
            for img_file in remaining_files:
                input_path = os.path.join(input_dir, img_file)
                if debug:
                    logger.debug(f"处理图片: {input_path}")
                try:
                    self.process_image(
                        input_path,
                        output_dir,
                        debug_dir,
                        base_data,
                        debug,
                        similarity_threshold,
                        add_border,
                    )
                    pbar.update(1)  # 更新进度条
                except Exception as e:
                    logger.error(f"处理图片时出错: {os.path.basename(input_path)} - {str(e)}")
                    raise

        # 生成视频
        if create_video:
            logger.info("开始生成视频...")
            try:
                self.video_creator.create_video(
                    output_dir=output_dir,
                    fps=fps,
                    transition_frames=transition_frames,
                    debug=debug
                )
                logger.info("视频生成完成")
            except Exception as e:
                logger.error(f"生成视频时出错: {str(e)}")
                raise 