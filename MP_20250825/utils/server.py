import socket
import threading
from typing import Callable, Tuple

from config.main_config import MainArgs
from utils.logger import logger

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class UdpServer:

    def __init__(self, address: Tuple[str, int], buffer_size: int, args: MainArgs) -> None:
        self.address = address
        self.buffer_size = buffer_size
        self.args = args
        self.is_running = False
        self.sock = self._create_socket(self.address)

    def __del__(self) -> None:
        self.sock.close()

    def start(self, callback: Callable, parallel: bool = False) -> None:
        if self.is_running:
            return
        self.is_running = True
        if parallel:
            self.parallel(callback)
        else:
            self.linear(callback)
        self.stop()

    def _create_socket(self, address: Tuple[str, int]) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
        return sock

    def parallel(self, callback: Callable) -> None:
        """202050611 : 신규 쓰레드 생성해서 병렬 처리

        Args:
            callback (Callable): _description_
        """
        while self.is_running:
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
                thread = threading.Thread(target=callback, args=(data, addr, self.args), daemon=True)
                thread.start()
            except Exception as e:
                logger.error(e)
                break

    def linear(self, callback: Callable) -> None:
        """20250611 : 받는 순서대로 처리

        Args:
            callback (Callable): _description_
        """
        while self.is_running:
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
                callback(data, addr, self.args)
            except Exception as e:
                logger.error(e)
                break

    def stop(self) -> None:
        self.is_running = False
