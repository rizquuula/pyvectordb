import logging
import chromadb
from chromadb.config import Settings
from typing import List

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.vector_distance import VectorDistance
from pyvectordb.vector import Vector
from pyvectordb.driver import VectorDB

from .distance import Distance


class ChromaDB(VectorDB):
    
    def __init__(
        self, 
        host: str, 
        port: str,
        auth_provider: str,
        auth_credentials: str,
        collection_name: str,
        distance_function: DistanceFunction=DistanceFunction.L2,
    ) -> None:
        super().__init__()
        
        self.host = host or self.__raise_value_error("host")
        self.port = port or self.__raise_value_error("port")
        self.auth_provider = auth_provider or self.__raise_value_error("auth_provider")
        self.auth_credentials = auth_credentials or self.__raise_value_error("auth_credentials")
        self.collection_name = collection_name or self.__raise_value_error("collection_name")
        self.distance_function = distance_function or self.__raise_value_error("distance_function")
        
        self.client = None
        self.collection = None
        
        self.__init_client()
        self.__health_check()
        self.__init_collection()
    
    @staticmethod
    def __raise_value_error(param: str):
        raise ValueError(f"{param} is required")
    
    def __init_client(self) -> None:
        if self.client is None:
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                ssl=False,
                headers={"X-Chroma-Token": self.auth_credentials},
                settings=Settings(
                    chroma_client_auth_provider=self.auth_provider,
                    chroma_client_auth_credentials=self.auth_credentials,
                )
            )
    
    def __health_check(self) -> None:
        self.client.heartbeat()
        
    def __init_collection(self) -> None:
        if self.collection == None:
            distance_func = self.__get_distance_function(self.distance_function)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": distance_func.value}
            )

    def create_vector(self, vector: Vector) -> Vector:
        self.collection.add(
            ids=[vector.get_id()],
            embeddings=[vector.embedding],
            metadatas=[{"metadata": vector.metadata}],
        )
        return vector

    def read_vector(self, id: str) -> Vector | None:
        result: dict = self.collection.get(
            ids=id,
            include=["metadatas", "documents", "embeddings"]
        )
        return Vector(
            vector_id=result.get("ids")[0],
            embedding=result.get("embeddings")[0],
            metadata=result.get("metadatas")[0].get("metadata")
        )

    def update_vector(self, vector: Vector) -> Vector:
        self.collection.update(
            ids=[vector.get_id()],
            embeddings=[vector.embedding],
            metadatas=[{"metadata": vector.metadata}],
        )
        return vector

    def delete_vector(self, id: str) -> None:
        self.collection.delete(
            ids=[id],
        )

    def get_neighbor_vectors(
        self, 
        vector: Vector, 
        n: int, 
    ) -> List[VectorDistance]:
        result: dict = self.collection.query(
            query_embeddings=[vector.embedding],
            n_results=n,
            include = ["metadatas", "documents", "distances", "embeddings"],
        )
        
        vds = []
        for i, _ in enumerate(result.get("ids")[0]):
            vd = VectorDistance(
                vector=Vector(
                    vector_id=result.get("ids")[0][i],
                    embedding=result.get("embeddings")[0][i],
                    metadata=result.get("metadatas")[0][i]
                ),
                distance=result.get("distances")[0][i]
            )
            vds.append(vd)
                              
        return vds

    def __get_distance_function(self, distance: DistanceFunction) -> Distance:
        if distance == DistanceFunction.L2:
            return Distance.SQUARED_L2
        elif distance == DistanceFunction.MAX_INNER_PRODUCT:
            return Distance.INNER_PRODUCT
        elif distance == DistanceFunction.COSINE:
            return Distance.COSINE_SIMILARITY
        else:
            raise ValueError(f"distance function unavailable on chromadb. [{Distance}]")