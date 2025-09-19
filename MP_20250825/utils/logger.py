import logging

# 로그 포맷 설정
log_format = "%(asctime)s - %(levelname)s - %(message)s"
# 날짜와 시간 형식 설정
date_format = "%Y-%m-%d %H:%M:%S"
# 로그 설정
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
# 로거 생성
logger = logging.getLogger()
