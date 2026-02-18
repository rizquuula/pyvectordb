import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import MetadataQuery

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.driver import VectorDB
from pyvectordb.vector import Vector
from pyvectordb.vector_distance import VectorDistance

from .distance import Distance


class WeaviateDB(VectorDB):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        api_key: str = None,
        collection: str = None,
        vector_size: int = None,
        distance_function: DistanceFunction | str = DistanceFunction.COSINE,
        debug: bool = False,
    ) -> None:
        super().__init__(host, port, debug)

        self.host = host or self.__raise_value_error("host")
        self.port = port or self.__raise_value_error("port")
        self.grpc_port = grpc_port
        self.api_key = api_key
        self.collection_name = collection or self.__raise_value_error("collection")
        self.vector_size = vector_size or self.__raise_value_error("vector_size")
        self.distance_function = distance_function or self.__raise_value_error("distance_function")

        self.client = None
        self.collection = None

        self.__init_client()
        self.__init_collection()

    @staticmethod
    def __raise_value_error(param: str):
        raise ValueError(f"{param} is required")

    def __init_client(self) -> None:
        if self.client is None:
            if self.api_key:
                # Connect to Weaviate Cloud
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.host,
                    auth_credentials=weaviate.classes.init.Auth.api_key(self.api_key),
                )
            else:
                # Connect to local/custom Weaviate instance
                self.client = weaviate.connect_to_custom(
                    http_host=self.host,
                    http_port=self.port,
                    http_secure=False,
                    grpc_host=self.host,
                    grpc_port=self.grpc_port,
                    grpc_secure=False,
                )

    def __init_collection(self) -> None:
        if self.collection is None:
            # Check if collection exists
            try:
                self.collection = self.client.collections.get(self.collection_name)
                # Verify the collection exists by trying to fetch
                self.collection.collections.get()
            except Exception:
                # Create collection if it doesn't exist
                distance_metric = self.__get_distance_function(self.distance_function)
                self.client.collections.create(
                    name=self.collection_name,
                    vectorizer_config=wvc.Configure.Vectorizer.none(),
                    vector_index_config=wvc.Configure.VectorIndex.flat(distance_metric=distance_metric),
                    properties=[wvc.Property(name="metadata", data_type=wvc.DataType.OBJECT)],
                )
                self.collection = self.client.collections.get(self.collection_name)

    def __get_distance_function(self, distance_function: DistanceFunction | str) -> str:
        if isinstance(distance_function, str):
            distance_function = DistanceFunction.from_str(distance_function)

        if distance_function == DistanceFunction.COSINE:
            return Distance.COSINE.value
        elif distance_function == DistanceFunction.EUCLIDEAN:
            return Distance.L2_SQUARED.value
        elif distance_function == DistanceFunction.DOT:
            return Distance.DOT.value
        elif distance_function == DistanceFunction.MANHATTAN:
            return Distance.MANHATTAN.value
        elif distance_function == DistanceFunction.HAMMING:
            return Distance.HAMMING.value
        else:
            d_ = [
                "COSINE",
                "EUCLIDEAN",
                "DOT",
                "MANHATTAN",
                "HAMMING",
            ]
            raise ValueError(f"distance function unavailable on weaviate: {d_}")

    def insert_vector(self, vector: Vector) -> None:
        vector_id = vector.get_id()

        self.collection.data.insert(
            uuid=vector_id,
            properties={"metadata": vector.metadata},
            vector=vector.embedding,
        )

    def insert_vectors(self, vectors: list[Vector]) -> None:
        if len(vectors) == 0:
            return

        data = []
        for vector in vectors:
            data.append(
                {
                    "uuid": vector.get_id(),
                    "properties": {"metadata": vector.metadata},
                    "vector": vector.embedding,
                }
            )

        # Use insert_many for batch insert
        objects = []
        for item in data:
            from weaviate.classes.data import DataObject

            obj = DataObject(
                uuid=item["uuid"],
                properties=item["properties"],
                vector=item["vector"],
            )
            objects.append(obj)

        self.collection.data.insert_many(objects)

    def read_vector(self, id: str) -> Vector | None:
        try:
            result = self.collection.query.fetch_object_by_id(
                uuid=id,
                include_vector=True,
            )

            if result is None:
                return None

            return Vector(
                embedding=result.vector,
                vector_id=result.uuid,
                metadata=result.properties.get("metadata") if result.properties else None,
            )
        except Exception:
            return None

    def update_vector(self, vector: Vector) -> None:
        vector_id = vector.get_id()

        self.collection.data.update(
            uuid=vector_id,
            properties={"metadata": vector.metadata},
            vector=vector.embedding,
        )

    def update_vectors(self, vectors: list[Vector]) -> None:
        # Weaviate doesn't have a direct batch update, so we update one by one
        for vector in vectors:
            self.update_vector(vector)

    def delete_vector(self, id: str) -> None:
        self.collection.data.delete_by_id(uuid=id)

    def delete_vectors(self, ids: list[str] | list[Vector]) -> None:
        if len(ids) == 0:
            return

        if isinstance(ids[0], Vector):
            ids = [v.get_id() for v in ids]

        for id_ in ids:
            self.delete_vector(id_)

    def get_neighbor_vectors(
        self,
        vector: Vector,
        n: int,
    ) -> list[VectorDistance]:
        results = self.collection.query.near_vector(
            near_vector=vector.embedding,
            limit=n,
            return_metadata=MetadataQuery(distance=True),
        )

        vector_distances = []
        for obj in results.objects:
            vector_distance = VectorDistance(
                vector=Vector(
                    embedding=obj.vector,
                    vector_id=obj.uuid,
                    metadata=obj.properties.get("metadata") if obj.properties else None,
                ),
                distance=obj.metadata.distance if obj.metadata else 0.0,
            )
            vector_distances.append(vector_distance)

        return vector_distances


__all__ = ["WeaviateDB"]
