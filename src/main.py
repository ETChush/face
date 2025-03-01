"""主程序入口

提供命令行参数解析和主流程控制。
"""

import os
import click
from typing import Optional

from .utils.logging import setup_logging, get_logger
from .utils.file import get_sorted_image_files, ensure_directory
from .processors.face_processor import FaceProcessor
from .video.video_creator import VideoCreator

logger = get_logger(__name__)


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    help="输出目录路径，默认为输入目录下的output子目录",
)
@click.option(
    "--debug/--no-debug", "-d", default=False, help="是否输出调试信息和标记图片"
)
@click.option("--video/--no-video", "-v", default=False, help="是否生成过渡视频")
@click.option("--fps", default=24, help="视频帧率，默认24fps")
@click.option("--transition", "-t", default=0.5, help="图片过渡时间（秒），默认0.5秒")
@click.option("--stay", "-s", default=0.25, help="图片停留时间（秒），默认0.25秒")
@click.option("--similarity-threshold", default=0.3, help="人脸相似度阈值，默认0.3")
@click.option("--size", "-z", help="输出图片尺寸，格式为'宽x高'，如'1920x1080'，默认保持原始大小")
@click.option("--border/--no-border", default=True, help="是否添加白边，默认添加")
def main(
    input_dir: str,
    output_dir: Optional[str] = None,
    debug: bool = False,
    video: bool = False,
    fps: int = 24,
    transition: float = 0.5,
    stay: float = 0.25,
    similarity_threshold: float = 0.3,
    size: Optional[str] = None,
    border: bool = True,
) -> None:
    """人脸对齐和视频生成工具

    处理指定目录下的所有图片，对齐人脸并可选生成过渡视频。

    详细处理流程：
    1. 将输入参数路径中的图片按照文件名排序：
       - 文件名如果以数字开头则按照数字升序
       - 否则按字节序排序

    2. 将第一张图片作为基准图片：
       - 使用YOLOv8对图片中的人脸进行检测，取第一张人脸
       - 使用MediaPipe对选择的人脸进行关键点提取
       - 按照关键点将人脸调整至眼睛水平，嘴巴在下的正常状态
       - 剪切缩放人脸为112x112大小方便ArcFace处理
       - 使用InsightFace获取人脸的嵌入向量作为后续比较的基准
       - 以眼睛中心点为圆心，调整整个图片的角度和大小，在保持眼睛中心的前提下，尽可能多的保留原图部分

    3. 一张一张处理后续图片：
       - 使用YOLOv8对图片中的人脸进行检测
       - 对每张人脸都做如下处理：
         * 使用MediaPipe对选择的人脸进行关键点提取
         * 按照关键点将人脸调整至眼睛水平，嘴巴在下的正常状态
         * 剪切缩放人脸为112x112大小方便ArcFace处理
         * 使用InsightFace获取人脸的嵌入向量
       - 取和基准人脸最相似的人脸为整张图片的主要人脸，但相似度必须大于阈值(默认0.5)
       - 以眼睛中心点为圆心，调整整个图片的角度和大小，按照基准图片眼睛的间距放大缩小，最后和基准图片宽度和高度保持一致

    4. 可选生成过渡视频

    示例：\n
    # 基本用法，只对齐图片\n
    python main.py input_dir\n
    # 生成视频\n
    python main.py input_dir -v\n
    # 调试模式\n
    python main.py input_dir -d\n
    # 指定输出目录\n
    python main.py input_dir -o output_dir\n
    # 自定义视频参数\n
    python main.py input_dir -v --fps 30 -t 1.0 -s 0.5\n
    # 自定义相似度阈值\n
    python main.py input_dir --similarity-threshold 0.6\n
    # 自定义输出尺寸\n
    python main.py input_dir --size 1920x1080\n
    # 不添加白边\n
    python main.py input_dir --no-border
    """
    # 设置日志系统
    setup_logging(debug)

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, "output")
    ensure_directory(output_dir)

    # 如果是调试模式，创建调试输出目录
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        ensure_directory(debug_dir)
        logger.debug(f"创建调试输出目录: {debug_dir}")
    else:
        debug_dir = None

    # 处理尺寸参数
    target_size = None
    if size:
        try:
            width, height = map(int, size.lower().split('x'))
            if width <= 0 or height <= 0:
                raise ValueError("尺寸必须大于0")
            target_size = (height, width)  # OpenCV使用(height, width)格式
        except Exception as e:
            logger.error(f"无效的尺寸格式 '{size}'，应为 'widthxheight'，如 '1920x1080'")
            return
    else:
        target_size = (1080, 1920)
    logger.info(f"目标尺寸: {target_size}")

    # 处理所有图片
    try:
        logger.info("开始处理图片...")
        face_processor = FaceProcessor()
        face_processor.process_images(
            input_dir,
            output_dir,
            debug=debug,
            similarity_threshold=similarity_threshold,
            base_size=target_size,
            add_border=border,
            create_video=video,
            fps=fps,
        )
    except Exception as e:
        logger.error(f"处理图片失败: {str(e)}")
        return
