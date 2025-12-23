import os
import logging
import sys

class StreamToLogger:
    """
    一个代理类，将 sys.stdout 的输出重定向到 logger
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        # print 每次调用都会发送内容，可能包含换行符
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        # 即使没有内容也要提供 flush 方法，以兼容 sys.stdout 接口
        pass

def setup_logger(filename: str = 'logs/analysis.log', level: str = 'INFO'):
    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()  # root logger
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建处理器
    stream_handler = logging.StreamHandler(sys.__stdout__) # 强制使用原始终端输出，避免死循环
    file_handler = logging.FileHandler(filename, mode='w', encoding='utf-8')

    # 设置格式
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # --- 重定向 print ---
    # 将 sys.stdout 指向我们的代理类
    # 这样所有 print(...) 都会被转换成 logger.info(...)
    sys.stdout = StreamToLogger(logger, logging.INFO)
