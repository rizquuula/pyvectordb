import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, ScoredPoint

from typing import List

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.driver import VectorDB
from pyvectordb.vector_distance import VectorDistance
from pyvectordb.vector import Vector


class QdrantDB(VectorDB):
    
    def __init__(
        self, 
        host: str, 
        api_key: str, 
        port: int, 
        collection: str,
        vector_size: int,
        distance_function: DistanceFunction=DistanceFunction.EUCLIDEAN,
        payload_key: str="description"
    ) -> None:
        super().__init__()
        
        self.host = host or self.__raise_value_error("host")
        self.api_key = api_key or self.__raise_value_error("api_key")
        self.port = port or self.__raise_value_error("port")
        self.collection = collection or self.__raise_value_error("collection")
        self.vector_size = vector_size or self.__raise_value_error("vector_size")
        self.distance_function = distance_function or self.__raise_value_error("distance_function")
        self.payload_key = payload_key or self.__raise_value_error("payload_key")
        
        self.client: QdrantClient = None
        
        self.__init__client()
        self.__init_collection()
    
    @staticmethod
    def __raise_value_error(param: str):
        raise ValueError(f"{param} is required")
    
    def __init__client(self) -> None:
        if self.client is None:
            self.client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key, https=False)
    
    def __init_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size, 
                    distance=self.__get_distance_function(self.distance_function)
                ),
            )
            
    def __get_distance_function(self, distance: DistanceFunction) -> Distance:
        if distance == DistanceFunction.COSINE: 
            distance_func = Distance.COSINE
        elif distance == DistanceFunction.EUCLIDEAN: 
            distance_func = Distance.EUCLID
        elif distance == DistanceFunction.DOT: 
            distance_func = Distance.DOT
        elif distance == DistanceFunction.MANHATTAN: 
            distance_func = Distance.MANHATTAN
        else:
            raise ValueError("distance function unavailable on qdrant")
        
        return distance_func

    def create_vector(self, vector: Vector) -> Vector:
        vector_id = vector.get_id()

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=vector_id,
                    vector=vector.embedding,
                    payload={self.payload_key: vector.description}
                )
            ],
            wait=False
        )
        
        return Vector(
            embedding=vector.embedding,
            vector_id=vector_id,
            description=vector.description,
        )

    def read_vector(self, id: int) -> Vector | None:
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[id],
            with_payload=True,
            with_vectors=True,
        )
        if len(records) == 0:
            return None
        
        record = records[0]
        return Vector(
            embedding=record.vector,
            vector_id=record.id,
            description=record.payload.get(self.payload_key),
        )

    def update_vector(self, vector: Vector) -> Vector:
        if self.read_vector(vector.get_id()) is None:
            raise ValueError("vector not exist")
        
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=vector.get_id(),
                    vector=vector.embedding,
                    payload={self.payload_key: vector.description}
                )
            ],
            wait=False
        )
        return Vector(
            embedding=vector.embedding,
            vector_id=vector.get_id(),
            description=vector.description,
        )

    def delete_vector(self, id: int) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=[id],
            wait=False
        )

    def get_neighbor_vectors(
        self,
        vector: Vector,
        n: int,
        distance_function: DistanceFunction=None,
    ) -> List[VectorDistance]:
        logging.warning("distance_function in qdrant for get_neighbor_vector is not used. Qdrant use distance_function from collection initialization.")
        if distance_function != self.distance_function:
            # restart client
            self.distance_function = distance_function
            self.client = None
            self.__init__client()
            
        scored_points: List[ScoredPoint] = self.client.search(
            collection_name=self.collection,
            query_vector=vector.embedding,
            with_payload=True,
            with_vectors=True,
            limit=n,
        )
        vector_distances  = []
        for point in scored_points:
            vector_distance = VectorDistance(
                vector=Vector(
                    embedding=point.vector,
                    vector_id=point.id,
                    description=point.payload.get(self.payload_key),
                ),
                distance=point.score,
            )
            vector_distances.append(vector_distance)
        return vector_distances
