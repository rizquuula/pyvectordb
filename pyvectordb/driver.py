from abc import ABC, abstractmethod
from typing import List

from .distance_function import DistanceFunction
from .vector_distance import VectorDistance
from .vector import Vector


class VectorDB(ABC):

    @abstractmethod
    def create_vector(self, vector: Vector) -> Vector:
        ...

    @abstractmethod
    def read_vector(self, id: str) -> Vector | None:
        ...

    @abstractmethod
    def update_vector(self, vector: Vector) -> Vector:
        ...

    @abstractmethod
    def delete_vector(self, id: str) -> None:
        ...

    @abstractmethod
    def get_neighbor_vectors(self, vector: Vector, n: int) -> List[VectorDistance]:
        ...
