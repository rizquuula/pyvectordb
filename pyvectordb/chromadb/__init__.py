import chromadb
from chromadb.config import Settings

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.driver import VectorDB
from pyvectordb.vector import Vector
from pyvectordb.vector_distance import VectorDistance

from .distance import Distance


class ChromaDB(VectorDB):
    def __init__(
        self,
        host: str,
        port: str,
        auth_provider: str = None,
        auth_credentials: str = None,
        collection_name: str = None,
        distance_function: DistanceFunction | str = DistanceFunction.L2,
        debug: bool = False,
    ) -> None:
        super().__init__(host, port, debug)

        self.host = host or self.__raise_value_error("host")
        self.port = port or self.__raise_value_error("port")
        self.auth_provider = auth_provider
        self.auth_credentials = auth_credentials
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
            # ChromaDB v1.x API - using HttpClient with settings
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )

            if self.auth_provider and self.auth_credentials:
                settings.chroma_client_auth_provider = self.auth_provider
                settings.chroma_client_auth_credentials = self.auth_credentials

            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=settings,
            )

    def __health_check(self) -> None:
        self.client.heartbeat()

    def __init_collection(self) -> None:
        if self.collection is None:
            distance_func = self.__get_distance_function(self.distance_function)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": distance_func.value}
            )

    def insert_vector(self, vector: Vector) -> None:
        self.collection.add(
            ids=[vector.get_id()],
            embeddings=[vector.embedding],
            metadatas=[vector.metadata],
        )

    def insert_vectors(self, vectors: list[Vector]) -> None:
        if len(vectors) == 0:
            return

        ids, embeddings, metadatas = [], [], []
        for v in vectors:
            ids.append(v.get_id())
            embeddings.append(v.embedding)
            metadatas.append(v.metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def read_vector(self, id: str) -> Vector | None:
        result: dict = self.collection.get(ids=id, include=["metadatas", "documents", "embeddings"])

        if len(result.get("ids")) == 0:
            return None

        vector_id = result.get("ids")[0]
        embedding = result.get("embeddings")[0]
        metadata = result.get("metadatas")[0]

        return Vector(
            embedding=embedding,
            vector_id=vector_id,
            metadata=metadata,
        )

    def update_vector(self, vector: Vector) -> None:
        self.collection.update(
            ids=[vector.get_id()],
            embeddings=[vector.embedding],
            metadatas=[vector.metadata],
        )

    def update_vectors(self, vectors: list[Vector]) -> None:
        if len(vectors) == 0:
            return

        ids, embeddings, metadatas = [], [], []
        for v in vectors:
            ids.append(v.id)
            embeddings.append(v.embedding)
            metadatas.append(v.metadata)

        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_vector(self, id_: str) -> None:
        self.collection.delete(
            ids=[id_],
        )

    def delete_vectors(self, ids: list[str] | list[Vector]) -> None:
        if len(ids) == 0:
            return

        if isinstance(ids[0], Vector):
            ids = [v.id for v in ids]

        self.collection.delete(
            ids=ids,
        )

    def get_neighbor_vectors(
        self,
        vector: Vector,
        n: int,
    ) -> list[VectorDistance]:
        result: dict = self.collection.query(
            query_embeddings=[vector.embedding],
            n_results=n,
            include=["metadatas", "documents", "distances", "embeddings"],
        )

        vds = []
        for i, _ in enumerate(result.get("ids")[0]):
            vd = VectorDistance(
                vector=Vector(
                    vector_id=result.get("ids")[0][i],
                    embedding=result.get("embeddings")[0][i],
                    metadata=result.get("metadatas")[0][i],
                ),
                distance=result.get("distances")[0][i],
            )
            vds.append(vd)

        return vds

    def __get_distance_function(self, distance_function: DistanceFunction | str) -> Distance:
        if isinstance(distance_function, str):
            distance_function = DistanceFunction.from_str(distance_function)

        if distance_function == DistanceFunction.L2:
            return Distance.SQUARED_L2
        elif distance_function == DistanceFunction.MAX_INNER_PRODUCT:
            return Distance.INNER_PRODUCT
        elif distance_function == DistanceFunction.COSINE:
            return Distance.COSINE_SIMILARITY
        else:
            d_ = [
                "L2",
                "MAX_INNER_PRODUCT",
                "COSINE",
            ]
            raise ValueError(f"distance function unavailable on chromadb. {d_}")


__all__ = ["ChromaDB"]
