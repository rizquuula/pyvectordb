import logging
from typing import List
from uuid import uuid4


class Vector:
    def __init__(
        self,
        embedding: List[float],
        vector_id: str | None=None,
        description: str | None=None,
    ) -> None:
        if embedding is None or len(embedding) == 0:
            raise ValueError("embedding is required")
        
        self.id = vector_id
        self.embedding = embedding
        self.description = description

    def get_id(self) -> str:
        if self.id is not None:
            return self.id
        
        self.id = str(uuid4())
        return self.id
    
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
            description=data.get("text"),
        )
        
    def __len__(self) -> int:
        return len(self.embedding)

    def __str__(self) -> str:
        if self.embedding is not None and len(self.embedding) > 10:
            embedding = f"{self.embedding[:5]}...{self.embedding[-5:]}"
        else:
            embedding = self.embedding
            
        if self.description and len(self.description) > 10:
            text = f"{self.description[:5]}...{self.description[-5:]}"
        else:
            text = self.description
            
        return f"""Vector[id: {self.id}, embedding: {embedding}, embedding_length: {len(self.embedding)}, text: {text}]"""
    
    def __repr__(self) -> str:
        return self.__str__()