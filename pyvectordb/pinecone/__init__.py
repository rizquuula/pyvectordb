from pinecone import AwsRegion, CloudProvider, Metric, Pinecone, ServerlessSpec

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.driver import VectorDB
from pyvectordb.vector import Vector
from pyvectordb.vector_distance import VectorDistance


class PineconeDB(VectorDB):
    def __init__(
        self,
        api_key: str,
        host: str = None,
        index_name: str = None,
        dimension: int = None,
        cloud: str = "aws",
        region: str = "us-east-1",
        distance_function: DistanceFunction | str = DistanceFunction.COSINE,
        debug: bool = False,
    ) -> None:
        # Pinecone is a managed service, so we use a dummy port for the base class
        super().__init__(host or "pinecone.io", 443, debug)

        self.api_key = api_key or self.__raise_value_error("api_key")
        self.host = host
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self.distance_function = distance_function or self.__raise_value_error("distance_function")

        self.client: Pinecone = None
        self.index = None

        self.__init_client()
        self.__init_index()

    @staticmethod
    def __raise_value_error(param: str):
        raise ValueError(f"{param} is required")

    def __init_client(self) -> None:
        if self.client is None:
            self.client = Pinecone(api_key=self.api_key)

    def __init_index(self) -> None:
        # If index_name is provided but host is not, try to get host from existing index
        if self.index_name and not self.host:
            try:
                index_description = self.client.describe_index(self.index_name)
                self.host = index_description.host
            except Exception:
                # Index doesn't exist, will create if dimension is provided
                pass

        # Create index if it doesn't exist and dimension is provided
        if self.index_name and self.dimension and not self.host:
            self.__create_index()

        # Initialize index client
        if self.host:
            self.index = self.client.Index(host=self.host)
        elif self.index_name:
            # Try connecting by name (SDK will fetch host automatically)
            self.index = self.client.Index(name=self.index_name)

    def __create_index(self) -> None:
        metric = self.__get_distance_function(self.distance_function)

        self.client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=self.__get_cloud_provider(self.cloud), region=self.__get_aws_region(self.region)),
        )

        # Get the host after creation
        index_description = self.client.describe_index(self.index_name)
        self.host = index_description.host
        self.index = self.client.Index(host=self.host)

    def __get_cloud_provider(self, cloud: str) -> CloudProvider:
        cloud = cloud.lower()
        if cloud == "aws":
            return CloudProvider.AWS
        elif cloud == "gcp":
            return CloudProvider.GCP
        elif cloud == "azure":
            return CloudProvider.AZURE
        else:
            return CloudProvider.AWS

    def __get_aws_region(self, region: str) -> AwsRegion:
        try:
            return AwsRegion(region)
        except ValueError:
            # Return the region string as-is, Pinecone accepts it
            return region

    def __get_distance_function(self, distance_function: DistanceFunction | str) -> Metric:
        if isinstance(distance_function, str):
            distance_function = DistanceFunction.from_str(distance_function)

        if distance_function == DistanceFunction.COSINE:
            return Metric.COSINE
        elif distance_function == DistanceFunction.EUCLIDEAN:
            return Metric.EUCLIDEAN
        elif distance_function == DistanceFunction.DOT:
            return Metric.DOTPRODUCT
        else:
            d_ = [
                "COSINE",
                "EUCLIDEAN",
                "DOT",
            ]
            raise ValueError(f"distance function unavailable on pinecone: {d_}")

    def insert_vector(self, vector: Vector) -> None:
        self.index.upsert(vectors=[(vector.get_id(), vector.embedding, vector.metadata)])

    def insert_vectors(self, vectors: list[Vector]) -> None:
        if len(vectors) == 0:
            return

        vectors_data = [(vector.get_id(), vector.embedding, vector.metadata) for vector in vectors]

        self.index.upsert(vectors=vectors_data)

    def read_vector(self, id: str) -> Vector | None:
        fetch_response = self.index.fetch(ids=[id])

        if id not in fetch_response.vectors:
            return None

        vector_data = fetch_response.vectors[id]
        return Vector(
            embedding=vector_data.values,
            vector_id=vector_data.id,
            metadata=vector_data.metadata,
        )

    def update_vector(self, vector: Vector) -> None:
        # Pinecone uses upsert for both insert and update
        self.insert_vector(vector)

    def update_vectors(self, vectors: list[Vector]) -> None:
        # Pinecone uses upsert for both insert and update
        self.insert_vectors(vectors)

    def delete_vector(self, id: str) -> None:
        self.index.delete(ids=[id])

    def delete_vectors(self, ids: list[str] | list[Vector]) -> None:
        if len(ids) == 0:
            return

        if isinstance(ids[0], Vector):
            ids = [v.get_id() for v in ids]

        self.index.delete(ids=ids)

    def get_neighbor_vectors(
        self,
        vector: Vector,
        n: int,
    ) -> list[VectorDistance]:
        query_response = self.index.query(
            vector=vector.embedding,
            top_k=n,
            include_metadata=True,
            include_values=True,
        )

        vector_distances = []
        for match in query_response.matches:
            vector_distance = VectorDistance(
                vector=Vector(
                    embedding=match.values,
                    vector_id=match.id,
                    metadata=match.metadata,
                ),
                distance=match.score,
            )
            vector_distances.append(vector_distance)

        return vector_distances


__all__ = ["PineconeDB"]
