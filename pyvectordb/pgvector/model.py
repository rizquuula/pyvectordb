from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import Column, String, DateTime, Table


class Base(DeclarativeBase):
    __abstract__ = True
    def to_dict(self):
        return {field.name:getattr(self, field.name) for field in self.__table__.c}


class VectorORM(Base):
    __abstract__ = True
    
    id: Mapped[str] = Column(String, primary_key=True)
    embedding: Mapped[Vector] = mapped_column(Vector())
    metadata_: Mapped[str] = Column(String, nullable=False, name="metadata")
    created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.now())
        

def get_vector_orm(tablename: str) -> "VectorORM":
    class VectorORMreal(VectorORM):
        __tablename__ = tablename
    return VectorORMreal


# def get_vector_orm(table_name: str, engine) -> DefaultVectorORM:
#     v = DefaultVectorORM
#     v.__tablename__ = table_name
    
#     v.__table__ = Table(table_name, Base.metadata, autoload_with=engine)
#     return v