from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Any, Generator, List, Tuple, Union

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.vector_distance import VectorDistance
from pyvectordb.driver import VectorDB
from pyvectordb.vector import Vector

from .model import VectorORM, get_vector_orm


class PgvectorDB(VectorDB):
    
    def __init__(
        self,
        host: str,
        port: str,
        user: str,
        password: str,
        db_name: str,
        collection: str,
        distance_function: DistanceFunction | str=DistanceFunction.L2,
    ) -> None:
        super().__init__(host, port)
        
        self.db_user = user or self.__raise_value_error("db_user")
        self.db_password = password or self.__raise_value_error("db_password")
        self.db_host = host or self.__raise_value_error("db_host")
        self.db_port = port or self.__raise_value_error("db_port")
        self.db_name = db_name or self.__raise_value_error("db_name")  
        self.collection = collection or self.__raise_value_error("collection")  
        self.distance_function = distance_function or self.__raise_value_error("distance_function")
        
        self.__engine = None
        
        self.__init_engine()
        self.conn = next(self.__get_db_session())
        
        self.__init_collection()
        self.__vector_orm: VectorORM = get_vector_orm(self.collection)

    @staticmethod
    def __raise_value_error(param: str):
        raise ValueError(f"{param} is required")
    
    def __init_engine(self) -> None:
        if self.__engine == None:
            self.__engine = create_engine(
                f'postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}', 
                echo=True,
                connect_args={}
            )
    
    def __get_db_session(self) -> Generator[Session, None, None]:
        SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.__engine
        )

        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
            
    def __init_collection(self) -> None:
        query = f"""
CREATE TABLE IF NOT EXISTS {self.collection} (
    id text PRIMARY KEY,
    embedding vector,
    metadata text,
    created_at timestamptz DEFAULT now()
);
"""
        self.conn.execute(text(query))
        self.conn.commit()
        

    def insert_vector(self, vector: Vector) -> Vector:
        v = self.__vector_orm(
            id=vector.get_id(),
            embedding=vector.embedding,
            metadata_=vector.metadata_to_string(),
        )
        
        self.conn.add(v)
        self.conn.commit()

    def insert_vectors(self, vectors: List[Vector]) -> None:
        if len(vectors) == 0: return
        
        v_orms = []
        for vector in vectors:
            v_orms.append(
                self.__vector_orm(
                    id=vector.get_id(),
                    embedding=vector.embedding,
                    metadata_=vector.metadata_to_string(),
                )
            )
            
        self.conn.add_all(v_orms)
        self.conn.commit()
        
    def read_vector(self, id: int) -> Vector | None:
        v_orm = self.__read_vector_orm(id)
        
        if v_orm is None: 
            return None
        
        vector = Vector(
            embedding=v_orm.embedding,
            vector_id=v_orm.id,
            metadata=v_orm.metadata_,
        )
        return vector
    
    def update_vector(self, vector: Vector) -> Vector:
        v = self.__read_vector_orm(vector.id)
        if v is None: raise ValueError("vector not found in database")
        
        v.embedding = vector.embedding
        v.metadata_ = vector.metadata_to_string()
        
        self.conn.add(v)
        self.conn.commit()

    def update_vectors(self, vectors: List[Vector]) -> None:
        if len(vectors) == 0: return
        
        v_orms = []
        for vector in vectors:
            v = self.__read_vector_orm(vector.id)
            if v is None: raise ValueError(f"vector {vector.id} not found in database")
            
            v.embedding = vector.embedding
            v.metadata_ = vector.metadata_to_string()
            v_orms.append(v)
            
        self.conn.add_all(v_orms)
        self.conn.commit()

    def delete_vector(self, id: int) -> None:
        v = self.__read_vector_orm(id)
        self.conn.delete(v)
        self.conn.commit()
    
    def delete_vectors(self, ids: Union[List[str], List[Vector]]) -> None:
        if len(ids) == 0: return
        
        if isinstance(ids[0], Vector):
            [self.delete_vector(v.id) for v in ids]
        else:
            [self.delete_vector(id_) for id_ in ids]
        
    def get_neighbor_vectors(
        self, 
        vector: Vector, 
        n: int=5, 
    ) -> List[VectorDistance]:
        vectordistances = []
        distance_func = self.__get_distance_function(self.distance_function)

        q = self.conn.execute(
            select(
                self.__vector_orm,
                distance_func(vector.embedding).label("distance")
            )
            .order_by(distance_func(vector.embedding))
            .limit(n)
        )
        results: Tuple[List[VectorORM]] = q.all()
        
        for r in results:
            vector = Vector(
                embedding=r[0].embedding,
                vector_id=r[0].id,
                metadata=r[0].metadata_,
            )
            distance = r[1]
            vectordistances.append(
                VectorDistance(vector, distance)
            )
        return vectordistances
    
    def __get_distance_function(self, distance_function: DistanceFunction | str) -> Any:
        if isinstance(distance_function, str):
            distance_function = DistanceFunction.from_str(distance_function)
        
        if distance_function==DistanceFunction.L2:
            return self.__vector_orm.embedding.l2_distance
        elif distance_function==DistanceFunction.MAX_INNER_PRODUCT:
            return self.__vector_orm.embedding.max_inner_product
        elif distance_function==DistanceFunction.COSINE:
            return self.__vector_orm.embedding.cosine_distance
        elif distance_function==DistanceFunction.L1:
            return self.__vector_orm.embedding.l1_distance
        elif distance_function==DistanceFunction.HAMMING:
            return self.__vector_orm.embedding.hamming_distance
        elif distance_function==DistanceFunction.JACCARD:
            return self.__vector_orm.embedding.jaccard_distance
        else:
            raise ValueError(f"distance function unavailable on pgvector: : {[
                'L2',
                'MAX_INNER_PRODUCT',
                'COSINE',
                'L1',
                'HAMMING',
                'JACCARD',
            ]}")
        
    def __read_vector_orm(self, id: int) -> VectorORM | None:
        v = self.conn.execute(
            select(self.__vector_orm)
            .where(self.__vector_orm.id == id)
        ).one_or_none()
        
        if v is None: return None
        
        return v[0]
    
    
__all__ = [
    "PgvectorDB"
]