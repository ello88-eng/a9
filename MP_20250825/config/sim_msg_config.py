from enum import IntEnum

SIM_HEADER_FMT = "> B H B BBB Q"

SIM_STATUS_INTERVAL = 1.0


class SimulationStatus(IntEnum):
    NONE = 0x00
    START_INIT = 0x01
    INIT_COMPLETED = 0x02
    START_SIM = 0x03
    STOP_SIM = 0x04
    EXIT_SIM = 0x05
    ERROR = 0xFF


class SimulationEvent(IntEnum):
    START = 1
    STOP = 2
    EXIT = 3


SIM_STATUS_TRANSITION = {
    SimulationEvent.START.value: SimulationStatus.START_SIM.value,
    SimulationEvent.STOP.value: SimulationStatus.STOP_SIM.value,
    SimulationEvent.EXIT.value: SimulationStatus.EXIT_SIM.value,
}


# 모의 메시지 ID 정의
class SimulationMessageId(IntEnum):
    SIM_INIT_CMD = 0x01FF01
    SIM_CTRL_CMD = 0x01FF02
    SIM_OBJ_INFO = 0x026501
