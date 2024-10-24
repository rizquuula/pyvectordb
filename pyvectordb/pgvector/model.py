from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import Column, Integer, String, DateTime


class Base(DeclarativeBase):
    __abstract__ = True
    def to_dict(self):
        return {field.name:getattr(self, field.name) for field in self.__table__.c}
    
class VectorORM(Base):
    __tablename__ = "embeddings"
    
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    embedding: Mapped[Vector] = mapped_column(Vector())
    text: Mapped[str] = Column(String, nullable=False)      
    created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=datetime.now())
