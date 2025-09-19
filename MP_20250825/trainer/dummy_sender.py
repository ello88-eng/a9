import pickle
import socket
import time
import sys
import pathlib
import numpy as np
from infos import AvsInfoForRl, TrgInfoForRl
import argparse

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils.logger import logger


def send_dummy_data(udp_ip="127.0.0.1", udp_port=5005, interval=0.5):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        dummy_data = {
            "state": {
                "scenario": np.random.randint(low=0, high=40000, size=(1, 1000)),
                "agent": {id: AvsInfoForRl() for id in range(1, 51)},
                "target": {id: TrgInfoForRl() for id in range(1, 21)},
            }
        }
        data = pickle.dumps(dummy_data)
        sock.sendto(data, (udp_ip, udp_port))
        logger.info(f"Sent dummy data to {udp_ip}:{udp_port}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send dummy data to a specific UDP port."
    )
    parser.add_argument(
        "--udp_ip",
        type=str,
        default="127.0.0.1",
        help="UDP IP address to send data",
    )
    parser.add_argument(
        "--udp_port", type=int, default=5005, help="UDP port to send data"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval between data sends in seconds",
    )
    args = parser.parse_args()

    send_dummy_data(udp_ip=args.udp_ip, udp_port=args.udp_port, interval=args.interval)
