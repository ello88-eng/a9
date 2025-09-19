from enum import IntEnum


## 비행체 상태
# 체계모드
class SystemMode(IntEnum):
    GROUND = 0x10  # 지상모드
    LAUNCH = 0x20  # 발사모드
    MOVE = 0x30  # 임무-이동
    SURV = 0x31  # 임무-감시
    RECONN = 0x32  # 임무-정찰
    TRACK = 0x33  # 임무-추적
    ATTACK = 0x34  # 임무-타격
    RETURN = 0x40  # 복귀모드


# 비행조종방식
class FlightControlMode(IntEnum):
    AUTO = 0x52  # 자동임무


# SMP 자동임무 모드
class SmpMode(IntEnum):
    STAND_BY = 0x00  # 대기
    SURV = 0x01  # 감시
    RECONN = 0x02  # 정찰
    TRACK = 0x02  # 정찰 (다른 코드명)
    ATTACK = 0x04  # 타격
