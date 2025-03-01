"""日志工具模块

提供自定义的日志格式化和配置功能。
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """自定义日志格式化器，为不同级别的日志添加不同的颜色"""
    
    # 终端颜色代码
    COLORS = {
        "DEBUG": "\033[36m",     # 青色
        "INFO": "\033[32m",      # 绿色
        "WARNING": "\033[33m",    # 黄色
        "ERROR": "\033[31m",     # 红色
        "CRITICAL": "\033[35m",   # 紫色
        "RESET": "\033[0m"       # 重置颜色
    }
    
    # 不同级别的前缀
    PREFIXES = {
        "DEBUG": "[调试] ",
        "INFO": "[信息] ",
        "WARNING": "[警告] ",
        "ERROR": "[错误] ",
        "CRITICAL": "[严重] "
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录

        Args:
            record: 日志记录对象

        Returns:
            str: 格式化后的日志消息
        """
        # 获取原始消息
        log_message = super().format(record)
        # 添加颜色和前缀
        levelname = record.levelname
        prefix = self.PREFIXES.get(levelname, "")
        color_code = self.COLORS.get(levelname, self.COLORS["RESET"])
        reset_code = self.COLORS["RESET"]
        # 返回带颜色和前缀的消息
        return f"{color_code}{prefix}{log_message}{reset_code}"


def setup_logging(debug: bool = False) -> None:
    """配置日志系统

    Args:
        debug: 是否启用调试模式
    """
    # 设置日志级别
    log_level = logging.DEBUG if debug else logging.INFO
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 设置自定义格式化器
    formatter = ColoredFormatter("%(message)s")
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加新的处理器
    root_logger.addHandler(console_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logging.Logger: 日志记录器实例
    """
    return logging.getLogger(name) 