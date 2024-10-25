import logging
from typing import List
from uuid import uuid4


class Vector:
    def __init__(
        self,
        embedding: List[float],
        vector_id: str | None=None,
        metadata: dict | None=None,
        init_id: bool=False,
    ) -> None:
        
        if embedding is None or len(embedding) == 0:
            self.__raise_value_error("embedding")
        self.embedding =  embedding
        
        self.id = vector_id
        self.metadata = metadata
        
        if init_id: self.get_id()
        
    @staticmethod
    def __raise_value_error(param: str):
        raise ValueError(f"{param} is required")
    
    def get_id(self) -> str:
        if self.id is None:
            self.id = str(uuid4())
        return self.id
        
    def __len__(self) -> int:
        return len(self.embedding)

    def __str__(self) -> str:
        if self.embedding is not None and len(self.embedding) > 10:
            embedding = f"{self.embedding[:5]}...{self.embedding[-5:]}"
        else:
            embedding = self.embedding
            
        if self.metadata and len(self.metadata) > 10:
            metadata = f"{self.metadata[:5]}...{self.metadata[-5:]}"
        else:
            metadata = self.metadata
            
        return f"""Vector[id: {self.id}, embedding: {embedding}, embedding_length: {len(self.embedding)}, metadata: {metadata}]"""
    
    def __repr__(self) -> str:
        return self.__str__()