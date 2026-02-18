from pymilvus import MilvusClient

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.driver import VectorDB
from pyvectordb.vector import Vector
from pyvectordb.vector_distance import VectorDistance

from .distance import Distance


class MilvusDB(VectorDB):
    def __init__(
        self,
        host: str,
        port: int = 19530,
        collection: str = None,
        vector_size: int = None,
        distance_function: DistanceFunction | str = DistanceFunction.COSINE,
        debug: bool = False,
    ) -> None:
        super().__init__(host, port, debug)

        self.host = host or self.__raise_value_error("host")
        self.port = port or self.__raise_value_error("port")
        self.collection = collection or self.__raise_value_error("collection")
        self.vector_size = vector_size or self.__raise_value_error("vector_size")
        self.distance_function = distance_function or self.__raise_value_error("distance_function")

        self.client: MilvusClient = None

        self.__init_client()
        self.__init_collection()

    @staticmethod
    def __raise_value_error(param: str):
        raise ValueError(f"{param} is required")

    def __init_client(self) -> None:
        if self.client is None:
            uri = f"http://{self.host}:{self.port}"
            self.client = MilvusClient(uri=uri)

    def __init_collection(self) -> None:
        if not self.client.has_collection(self.collection):
            metric_type = self.__get_distance_function(self.distance_function)
            self.client.create_collection(
                collection_name=self.collection,
                dimension=self.vector_size,
                metric_type=metric_type,
            )

    def __get_distance_function(self, distance_function: DistanceFunction | str) -> str:
        if isinstance(distance_function, str):
            distance_function = DistanceFunction.from_str(distance_function)

        if distance_function == DistanceFunction.COSINE:
            return Distance.COSINE.value
        elif distance_function == DistanceFunction.EUCLIDEAN:
            return Distance.L2.value
        elif distance_function == DistanceFunction.DOT:
            return Distance.IP.value
        elif distance_function == DistanceFunction.HAMMING:
            return Distance.HAMMING.value
        elif distance_function == DistanceFunction.JACCARD:
            return Distance.JACCARD.value
        else:
            d_ = [
                "COSINE",
                "EUCLIDEAN",
                "DOT",
                "HAMMING",
                "JACCARD",
            ]
            raise ValueError(f"distance function unavailable on milvus: {d_}")

    def insert_vector(self, vector: Vector) -> None:
        vector_id = vector.get_id()

        self.client.insert(
            collection_name=self.collection,
            data=[
                {
                    "id": vector_id,
                    "vector": vector.embedding,
                    "metadata": vector.metadata,
                }
            ],
        )

    def insert_vectors(self, vectors: list[Vector]) -> None:
        if len(vectors) == 0:
            return

        data = []
        for vector in vectors:
            data.append(
                {
                    "id": vector.get_id(),
                    "vector": vector.embedding,
                    "metadata": vector.metadata,
                }
            )

        self.client.insert(collection_name=self.collection, data=data)

    def read_vector(self, id: str) -> Vector | None:
        results = self.client.query(
            collection_name=self.collection,
            ids=[id],
            output_fields=["vector", "metadata"],
        )

        if len(results) == 0:
            return None

        result = results[0]
        return Vector(
            embedding=result.get("vector"),
            vector_id=result.get("id"),
            metadata=result.get("metadata"),
        )

    def update_vector(self, vector: Vector) -> None:
        # Milvus uses upsert for both insert and update
        self.client.upsert(
            collection_name=self.collection,
            data=[
                {
                    "id": vector.get_id(),
                    "vector": vector.embedding,
                    "metadata": vector.metadata,
                }
            ],
        )

    def update_vectors(self, vectors: list[Vector]) -> None:
        if len(vectors) == 0:
            return

        data = []
        for vector in vectors:
            data.append(
                {
                    "id": vector.get_id(),
                    "vector": vector.embedding,
                    "metadata": vector.metadata,
                }
            )

        self.client.upsert(collection_name=self.collection, data=data)

    def delete_vector(self, id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            ids=[id],
        )

    def delete_vectors(self, ids: list[str] | list[Vector]) -> None:
        if len(ids) == 0:
            return

        if isinstance(ids[0], Vector):
            ids = [v.get_id() for v in ids]

        self.client.delete(
            collection_name=self.collection,
            ids=ids,
        )

    def get_neighbor_vectors(
        self,
        vector: Vector,
        n: int,
    ) -> list[VectorDistance]:
        results = self.client.search(
            collection_name=self.collection,
            data=[vector.embedding],
            limit=n,
            output_fields=["id", "vector", "metadata"],
        )

        vector_distances = []
        for hit in results[0]:
            vector_distance = VectorDistance(
                vector=Vector(
                    embedding=hit.get("entity", {}).get("vector"),
                    vector_id=hit.get("id"),
                    metadata=hit.get("entity", {}).get("metadata"),
                ),
                distance=hit.get("distance", 0.0),
            )
            vector_distances.append(vector_distance)

        return vector_distances


__all__ = ["MilvusDB"]
