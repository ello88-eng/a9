from config.resolution_config import CRC_16_TABLE


def compute_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        pos = (crc >> 8) ^ byte
        crc = (crc >> 8) ^ CRC_16_TABLE[pos]
    return crc
