from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, ScoredPoint

from typing import List, Union

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
        distance_function: DistanceFunction | str=DistanceFunction.EUCLIDEAN,
    ) -> None:
        super().__init__(host, port)
        
        self.host = host or self.__raise_value_error("host")
        self.api_key = api_key or self.__raise_value_error("api_key")
        self.port = port or self.__raise_value_error("port")
        self.collection = collection or self.__raise_value_error("collection")
        self.vector_size = vector_size or self.__raise_value_error("vector_size")
        self.distance_function = distance_function or self.__raise_value_error("distance_function")
        
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
            
    def __get_distance_function(self, distance_function: DistanceFunction | str) -> Distance:
        if isinstance(distance_function, str):
            distance_function = DistanceFunction.from_str(distance_function)
        
        if distance_function == DistanceFunction.COSINE: 
            distance_func = Distance.COSINE
        elif distance_function == DistanceFunction.EUCLIDEAN: 
            distance_func = Distance.EUCLID
        elif distance_function == DistanceFunction.DOT: 
            distance_func = Distance.DOT
        elif distance_function == DistanceFunction.MANHATTAN: 
            distance_func = Distance.MANHATTAN
        else:
            raise ValueError(f"distance function unavailable on qdrant: {[
                "COSINE",
                "EUCLIDEAN",
                "DOT",
                "MANHATTAN",
            ]}")
        
        return distance_func

    def insert_vector(self, vector: Vector) -> None:
        vector_id = vector.get_id()

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=vector_id,
                    vector=vector.embedding,
                    payload={"metadata": vector.metadata}
                )
            ],
            wait=True
        )
        
    def insert_vectors(self, vectors: List[Vector]) -> None:
        if len(vectors) == 0: return
        
        points = [
            PointStruct(
                id=vector.get_id(),
                vector=vector.embedding,
                payload=vector.metadata,
            )
            for vector in vectors
        ]
        
        self.client.upsert(
            collection_name=self.collection,
            points=points,
            wait=False
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
            metadata=record.payload.get("metadata"),
        )

    def update_vector(self, vector: Vector) -> None:
        # we use qdrant upsert, so...
        self.insert_vector(vector)
    
    def update_vectors(self, vectors: List[Vector]) -> None:
        # we use qdrant upsert, so...
        self.insert_vectors(vectors)
        
    def delete_vector(self, id: int) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=[id],
            wait=False
        )
    
    def delete_vectors(self, ids: Union[List[str], List[Vector]]) -> None:
        if len(ids) == 0: return
        
        if isinstance(ids[0], Vector):
            [self.delete_vector(v.id) for v in ids]
        else:
            [self.delete_vector(id_) for id_ in ids]
        
    def get_neighbor_vectors(
        self,
        vector: Vector,
        n: int,
    ) -> List[VectorDistance]:    
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
                    metadata=point.payload.get("metadata"),
                ),
                distance=point.score,
            )
            vector_distances.append(vector_distance)
        return vector_distances


__all__ = [
    "QdrantDB"
]