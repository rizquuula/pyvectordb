from .vector import Vector


class VectorDistance:
    
    def __init__(self, vector: Vector, distance: float) -> None:
        self.vector: Vector=vector
        self.distance = distance
        
    def __str__(self) -> str:
        return f"{self.vector}, distance={self.distance}"
    
    def __repr__(self) -> str:
        return self.__str__()