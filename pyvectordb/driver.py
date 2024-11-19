from abc import ABC, abstractmethod
import logging
import socket
from typing import List, Union

from .vector_distance import VectorDistance
from .vector import Vector


class VectorDB(ABC):
    
    def __init__(self, host, port, debug: bool=False):
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - Line: %(lineno)d - %(funcName)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
            ]
        )
        self.__log = logging.getLogger(self.__class__.__name__)
        
        self.__test_connection(host, port)

    @abstractmethod
    def insert_vector(self, vector: Vector) -> None: ...

    @abstractmethod
    def insert_vectors(self, vectors: List[Vector]) -> None: ...

    @abstractmethod
    def read_vector(self, id: str) -> Vector | None: ...

    @abstractmethod
    def update_vector(self, vector: Vector) -> None: ...

    @abstractmethod
    def update_vectors(self, vectors: List[Vector]) -> None: ...

    @abstractmethod
    def delete_vector(self, id: str) -> None: ...
    
    @abstractmethod
    def delete_vectors(self, ids: Union[List[str], List[Vector]]) -> None: ...

    @abstractmethod
    def get_neighbor_vectors(self, vector: Vector, n: int) -> List[VectorDistance]: ...

    def __test_connection(self, host, port):
        timeout = 3.0
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            try:
                s.connect((host, int(port)))
                self.__log.info(f"Connection to {host}:{port} succeeded.")
                
            except (socket.timeout, socket.error) as e:
                err_msg = f"Connection to {host}:{port} failed: {e}"
                self.__log.error(err_msg)
                raise ConnectionError(err_msg)