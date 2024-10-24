from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, Session
from typing import Any, Generator, List

from pyvectordb.distance_function import DistanceFunction
from pyvectordb.vector_distance import VectorDistance
from pyvectordb.driver import VectorDB
from pyvectordb.vector import Vector

from .model import VectorORM


class PgvectorDB(VectorDB):
    
    def __init__(
        self,
        db_user: str,
        db_password: str,
        db_host: str,
        db_port: str,
        db_name: str,
    ) -> None:
        super().__init__()
        
        self.__is_required(db_user, "db_user")
        self.__is_required(db_password, "db_password")
        self.__is_required(db_host, "db_host")
        self.__is_required(db_port, "db_port")
        self.__is_required(db_name, "db_name")
        
        self.__engine = create_engine(
            f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}', 
            echo=True,
            connect_args={}
            )
        
        self.conn = next(self.__get_db_session())

    def __is_required(self, text: str, type: str) -> None:
        if text is None or text == "":
            raise ValueError(f"{type} is required")
    
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

    def create_vector(self, vector: Vector) -> Vector:
        conn = self.conn
        v = VectorORM(
            embedding=vector.embedding,
            text=vector.description,
        )
        conn.add(v)
        conn.commit()
        conn.refresh(v)

        return Vector.new_from_dict(v.to_dict())
        

    def read_vector(self, id: int) -> Vector | None:
        v = self.__read_vector_orm(id)
        
        if v is None: 
            return None
        
        return Vector.new_from_dict(v.to_dict())       
    
    def update_vector(self, vector: Vector) -> Vector:
        conn = self.conn
        
        v = self.__read_vector_orm(vector.id)
        if v is None: raise ValueError("vector not found in database")
        
        v.embedding = vector.embedding
        v.text = vector.description
        
        conn.add(v)
        conn.commit()
        
        return Vector.new_from_dict(v.to_dict())

    def delete_vector(self, id: int) -> None:
        conn = self.conn
        v = self.__read_vector_orm(id)
        conn.delete(v)
        conn.commit()
    
    def get_neighbor_vectors(
        self, 
        vector: Vector, 
        n: int=5, 
        distance_function: DistanceFunction=DistanceFunction.COSINE,
    ) -> List[VectorDistance]:
        vectordistances = []
        distance_func = self.__get_distance_function(distance_function)

        q = self.conn.execute(
            select(
                VectorORM,
                distance_func(vector.embedding).label("distance")
            )
            .order_by(distance_func(vector.embedding))
            .limit(n)
        )
        results = q.all()
        
        for r in results:
            vector = Vector.new_from_dict(r[0].to_dict())
            distance = r[1]
            vectordistances.append(
                VectorDistance(vector, distance)
            )
        return vectordistances
    
    def __get_distance_function(self, selected_distance_function: DistanceFunction) -> Any:
        if selected_distance_function==DistanceFunction.L2:
            distance_func = VectorORM.embedding.l2_distance
        elif selected_distance_function==DistanceFunction.MAX_INNER_PRODUCT:
            distance_func = VectorORM.embedding.max_inner_product
        elif selected_distance_function==DistanceFunction.COSINE:
            distance_func = VectorORM.embedding.cosine_distance
        elif selected_distance_function==DistanceFunction.L1:
            distance_func = VectorORM.embedding.l1_distance
        elif selected_distance_function==DistanceFunction.HAMMING:
            distance_func = VectorORM.embedding.hamming_distance
        elif selected_distance_function==DistanceFunction.JACCARD:
            distance_func = VectorORM.embedding.jaccard_distance
        else:
            raise ValueError("distance function unavailable on pgvector")
        
        return distance_func

    def __read_vector_orm(self, id: int) -> VectorORM | None:
        v = self.conn.execute(
            select(VectorORM)
            .where(VectorORM.id == id)
        ).one_or_none()
        if v is None:
            return v
        
        return v[0]