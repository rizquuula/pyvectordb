import logging
from typing import List


class Vector:
    def __init__(
        self,
        embedding: List[float],
        vector_id: int | None=None,
        text: str | None=None,
    ) -> None:
        self.id = vector_id,
        self.embedding = embedding
        self.text = text

    @staticmethod
    def new_from_dict(data: dict) -> "Vector":
        if data.get("embedding") is None:
            raise ValueError("embedding is required")
        
        if isinstance(data.get("embedding"), list):
            logging.debug(f"invalid embedding: {data.get("embedding")}")
            raise ValueError("invalid embedding: invalid format")
        
        return Vector(
            vector_id=data.get("id"),
            embedding=data.get("embedding"),
            text=data.get("text"),
        )
        
    def __len__(self) -> int:
        return len(self.embedding)

    def __str__(self) -> str:
        if self.embedding is not None and len(self.embedding) > 10:
            embedding = f"{self.embedding[:5]}...{self.embedding[-5:]}"
        else:
            embedding = self.embedding
            
        if self.text and len(self.text) > 10:
            text = f"{self.text[:5]}...{self.text[-5:]}"
        else:
            text = self.text
            
        return f"""Vector[id: {self.id}, embedding: {embedding}, embedding_length: {len(self.embedding)}, text: {text}]"""
    
    def __repr__(self) -> str:
        return self.__str__()