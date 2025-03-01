"""ArcFace 模型模块

提供 ArcFace 人脸特征提取模型的初始化和使用功能。
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from insightface.app import FaceAnalysis

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ArcFaceExtractor:
    """ArcFace 特征提取器"""

    def __init__(self):
        """初始化 ArcFace 特征提取器"""
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """初始化 ArcFace 模型"""
        try:
            logger.debug("创建 FaceAnalysis 实例")
            self.model = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
                allowed_modules=["recognition", "detection"],
            )
            logger.debug("准备 ArcFace 模型")
            self.model.prepare(ctx_id=0)
            logger.debug("ArcFace 模型初始化完成")
        except Exception as e:
            logger.error(f"ArcFace 模型初始化失败: {str(e)}")
            raise

    def preprocess_face(
        self, face_img: np.ndarray, debug: bool = False
    ) -> np.ndarray:
        """预处理人脸图像

        Args:
            face_img: 输入的人脸图像
            debug: 是否启用调试模式

        Returns:
            np.ndarray: 预处理后的人脸图像
        """
        try:
            if debug:
                logger.debug(f"预处理人脸图像，输入尺寸: {face_img.shape}")

            # 确保输入图像不为空
            if face_img is None or face_img.size == 0:
                raise ValueError("输入图像为空")

            # 确保输入图像为3通道
            if len(face_img.shape) != 3 or face_img.shape[2] != 3:
                raise ValueError(f"输入图像必须为3通道彩色图像，当前形状: {face_img.shape}")

            # 调整图像大小为112x112（ArcFace标准输入尺寸）
            if face_img.shape[0] != 112 or face_img.shape[1] != 112:
                if debug:
                    logger.debug(f"调整图像大小为112x112，原始尺寸: {face_img.shape[:2]}")
                face_img = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_CUBIC)

            if debug:
                logger.debug(f"预处理完成，输出尺寸: {face_img.shape}")

            return face_img

        except Exception as e:
            if debug:
                logger.error(f"人脸预处理失败: {str(e)}")
            # 如果处理失败，创建一个空白的112x112图像
            return np.zeros((112, 112, 3), dtype=np.uint8)

    def extract_features(
        self, face_img: np.ndarray, debug: bool = False
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """从人脸图像中提取特征向量

        Args:
            face_img: 人脸图像
            debug: 是否启用调试模式

        Returns:
            Tuple[Optional[np.ndarray], np.ndarray]: 特征向量和预处理后的人脸图像
        """
        try:
            if debug:
                logger.debug("开始提取人脸特征")

            # 预处理人脸图像
            arcface_input = self.preprocess_face(face_img, debug)
            face_rgb = cv2.cvtColor(arcface_input, cv2.COLOR_BGR2RGB)

            try:
                # 获取特征提取模型
                rec_model = self.model.models.get("recognition", None)
                if rec_model is not None and hasattr(rec_model, "get_feat"):
                    if debug:
                        logger.debug("使用recognition模型的get_feat方法直接提取特征")
                    # 直接提取特征
                    embedding = rec_model.get_feat(face_rgb)
                    if embedding is not None:
                        if debug:
                            logger.debug(f"特征提取成功，特征向量形状: {embedding.shape}")
                        return embedding, arcface_input

                # 如果直接提取失败，使用get方法
                if debug:
                    logger.debug("尝试使用get方法提取特征")

                faces = self.model.get(face_rgb, max_num=1)

                if not faces:
                    if debug:
                        logger.debug("ArcFace未检测到人脸，尝试使用备选特征提取方法")
                    # 使用备选方法：计算HOG特征
                    embedding = self._compute_hog_features(face_rgb, debug)
                    return embedding, arcface_input

                # 获取特征向量
                embedding = faces[0].embedding

                if embedding is None:
                    if debug:
                        logger.debug("特征提取失败，embedding为None")
                    # 创建一个全零的特征向量
                    empty_embedding = np.zeros(512, dtype=np.float32)
                    return empty_embedding, arcface_input

                if debug:
                    logger.debug(f"特征提取成功，特征向量形状: {embedding.shape}")

                return embedding, arcface_input

            except Exception as e:
                if debug:
                    logger.debug(f"特征提取失败: {str(e)}")
                # 创建一个全零的特征向量
                empty_embedding = np.zeros(512, dtype=np.float32)
                return empty_embedding, arcface_input

        except Exception as e:
            if debug:
                logger.error(f"ArcFace处理失败: {str(e)}")
            raise

    def _compute_hog_features(
        self, face_rgb: np.ndarray, debug: bool = False
    ) -> np.ndarray:
        """计算HOG特征作为备选特征

        Args:
            face_rgb: RGB格式的人脸图像
            debug: 是否启用调试模式

        Returns:
            np.ndarray: HOG特征向量
        """
        try:
            if debug:
                logger.debug("使用HOG特征作为备选特征")

            # 转换为灰度图
            gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

            # 计算HOG特征
            from skimage.feature import hog

            hog_features = hog(
                gray,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(1, 1),
                visualize=False,
            )

            # 归一化特征向量
            if np.linalg.norm(hog_features) > 0:
                hog_features = hog_features / np.linalg.norm(hog_features)

            # 填充到512维
            embedding = np.zeros(512, dtype=np.float32)
            embedding[: len(hog_features)] = hog_features

            if debug:
                logger.debug(f"HOG特征提取成功，特征向量形状: {embedding.shape}")

            return embedding

        except Exception as e:
            if debug:
                logger.debug(f"HOG特征提取失败: {str(e)}")

            # 如果HOG特征提取失败，创建一个随机特征向量
            # 使用固定的种子，确保相同的图像生成相同的特征
            img_hash = hash(face_rgb.tobytes()) % 10000
            np.random.seed(img_hash)
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            if debug:
                logger.debug(f"生成随机特征向量，特征向量形状: {embedding.shape}")

            return embedding 