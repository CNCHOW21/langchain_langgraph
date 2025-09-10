# @Time    : 2025/8/28 19:15
# @Author  : liuzhou
# @File    : logger.py
# @software: PyCharm
import inspect
import logging
from utils.config import Config

from concurrent_log_handler import ConcurrentRotatingFileHandler


# 这里的关键在于ClassNameFormatter类中重写了format方法，在该方法内，我们使用了inspect模块来跟踪当前执行上下文，
# 并尝试找到发出日志请求的类实例。如果找到了，就将其类型名字作为classname添加到日志记录中；如果没有找到（即非类方法调用），
# 则设置为"None"或者你想要的其他默认值。这样就可以在日志输出中看到类名信息了
class ClassNameFormatter(logging.Formatter):
    def format(self, record):
        # 获取产生日志调用所在类的名称
        frame = inspect.currentframe()
        while frame:
            arginfo = inspect.getargvalues(frame)
            if 'self' in arginfo.args:
                record.classname = type(arginfo.locals['self']).__name__
                break
            frame = frame.f_back
        else:
            # 如果不在任何类中，则设定为None或其他默认值
            record.classname = "None"

        # 使用自定义的格式化字符串
        s = super().format(record)
        return s

# 设置日志基本配置，级别为DEBUG或INFO
logger = logging.getLogger(__name__)
# 设置日志器级别为DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # 日志文件
    Config.LOG_FILE,
    # 日志文件最大允许大小为5MB，达到上限后触发轮转
    maxBytes = Config.MAX_BYTES,
    # 在轮转时，最多保留3个历史日志文件
    backupCount = Config.BACKUP_COUNT
)

# 设置处理器级别处理器
handler.setLevel(logging.DEBUG)
# handler.setFormatter(logging.Formatter(
#     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# ))
formatter = ClassNameFormatter("[%(asctime)s] [%(threadName)s] [%(levelname)s] [%(filename)s:%(lineno)d] : %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# 控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # 控制台日志级别
formatter_console = logging.Formatter("[%(asctime)s] [%(threadName)s] [%(levelname)s] [%(filename)s:%(lineno)d] : %(message)s")
console_handler.setFormatter(formatter_console)
logger.addHandler(console_handler)