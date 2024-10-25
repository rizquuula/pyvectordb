from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Any, Generator, List

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.vector_distance import VectorDistance
from pyvectordb.driver import VectorDB
from pyvectordb.vector import Vector

from .model import get_vector_orm


class PgvectorDB(VectorDB):
    
    def __init__(
        self,
        db_user: str,
        db_password: str,
        db_host: str,
        db_port: str,
        db_name: str,
        collection: str,
        distance_function: DistanceFunction=DistanceFunction.L2,
    ) -> None:
        super().__init__()
        
        self.db_user = db_user or self.__raise_value_error("db_user")
        self.db_password = db_password or self.__raise_value_error("db_password")
        self.db_host = db_host or self.__raise_value_error("db_host")
        self.db_port = db_port or self.__raise_value_error("db_port")
        self.db_name = db_name or self.__raise_value_error("db_name")  
        self.collection = collection or self.__raise_value_error("collection")  
        self.distance_function = distance_function or self.__raise_value_error("distance_function")
        
        self.__engine = None
        
        self.__init_engine()
        self.conn = next(self.__get_db_session())
        
        self.__init_collection()
        self.__vector_orm = get_vector_orm(self.collection)

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
        

    def create_vector(self, vector: Vector) -> Vector:
        conn = self.conn
        
        v = self.__vector_orm(
            id=vector.get_id(),
            embedding=vector.embedding,
            metadata_=vector.metadata,
        )
        
        conn.add(v)
        conn.commit()
        conn.refresh(v)

        return Vector(
            embedding=v.embedding,
            vector_id=v.id,
            metadata=v.metadata_,
        )
        
    def read_vector(self, id: int) -> Vector | None:
        v = self.__read_vector_orm(id)
        
        if v is None: 
            return None
        
        return Vector(
            embedding=v.embedding,
            vector_id=v.id,
            metadata=v.metadata_,
        )   
    
    def update_vector(self, vector: Vector) -> Vector:
        conn = self.conn
        
        v = self.__read_vector_orm(vector.id)
        if v is None: raise ValueError("vector not found in database")
        
        v.embedding = vector.embedding
        v._metadata = vector.metadata
        
        conn.add(v)
        conn.commit()
        
        return Vector(
            embedding=v.embedding,
            vector_id=v.id,
            metadata=v.metadata_,
        )

    def delete_vector(self, id: int) -> None:
        conn = self.conn
        v = self.__read_vector_orm(id)
        conn.delete(v)
        conn.commit()
    
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
        results = q.all()
        
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
    
    def __get_distance_function(self, selected_distance_function: DistanceFunction) -> Any:
        if selected_distance_function==DistanceFunction.L2:
            return self.__vector_orm.embedding.l2_distance
        elif selected_distance_function==DistanceFunction.MAX_INNER_PRODUCT:
            return self.__vector_orm.embedding.max_inner_product
        elif selected_distance_function==DistanceFunction.COSINE:
            return self.__vector_orm.embedding.cosine_distance
        elif selected_distance_function==DistanceFunction.L1:
            return self.__vector_orm.embedding.l1_distance
        elif selected_distance_function==DistanceFunction.HAMMING:
            return self.__vector_orm.embedding.hamming_distance
        elif selected_distance_function==DistanceFunction.JACCARD:
            return self.__vector_orm.embedding.jaccard_distance
        else:
            raise ValueError("distance function unavailable on pgvector")
        
    def __read_vector_orm(self, id: int) -> object | None:
        v = self.conn.execute(
            select(self.__vector_orm)
            .where(self.__vector_orm.id == id)
        ).one_or_none()
        if v is None:
            return v
        
        return v[0]