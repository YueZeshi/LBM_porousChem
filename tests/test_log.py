import logging

# 获取日志器（推荐用法）
logger = logging.getLogger(__name__)  # 通常用模块名
# 或指定名称
logger = logging.getLogger('my_app')

# 设置级别
logger.setLevel(logging.CRITICAL)

# 输出日志
logger.debug('开始处理数据')
logger.info('用户登录成功')
logger.warning('磁盘空间不足80%')
logger.error('数据库连接失败')