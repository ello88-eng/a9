from enum import IntEnum


class Status(IntEnum):
    NONE = 0x00
    START_INIT = 0x01
    INIT_COMPLETED = 0x02
    START_SIM = 0x03
    STOP_SIM = 0x04
    EXIT_SIM = 0x05
    ERROR = 0xFF
