# PyVectorDB

Simple Wrapper for vector database in Python with minimal support for CRUD and retrieve.

## Installation 

    pip install pyvectordb

## Usage Example

### PGVector

PGvector is an extension for PostgreSQL that allows the storage, indexing, and querying of vector embeddings. It is designed to support vector similarity search, which is useful in machine learning applications like natural language processing, image recognition, and recommendation systems. By storing vector embeddings as a data type, PGvector enables efficient similarity searches using distance metrics such as cosine similarity, Euclidean distance, inner product, etc.

```py
from dotenv import load_dotenv
load_dotenv()

import os
from pyvectordb import Vector
from pyvectordb.pgvector.pgvector import PgvectorDB
from pyvectordb.distance_function import DistanceFunction

v1 = Vector(
    embedding=[2., 2., 1.],
    metadata={"text": "hellow from pyvectordb"}
)
v2 = Vector(
    embedding=[2., 2., 2.],
    metadata={"text": "hi"}
)
v3 = Vector(
    embedding=[2., 2., 3.],
    metadata={"text": "good morning!"}
)

vector_db = PgvectorDB(
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    db_name=os.getenv("PG_NAME"),
    collection=os.getenv("PG_COLLECTION"),
    distance_function=DistanceFunction.L2,
)

# insert new vector
vector_db.insert_vector(v1)
vector_db.insert_vectors([v2, v3])

# read v1
v_from_db = vector_db.read_vector(v1.get_id())

# update v1 embedding
new_embedding = [2., 2., 4.]
v_from_db.embedding = new_embedding
vector_db.update_vector(v_from_db)

# read updated embedding and check
v_from_db_updated = vector_db.read_vector(v1.get_id())
assert all(v_from_db_updated.embedding == new_embedding)

# re-update v1 embedding to the v1, check
vector_db.update_vectors([v1, v2, v3])
re_updated_embedding = vector_db.read_vector(v1.get_id()).embedding
assert all(re_updated_embedding == v1.embedding)

for x in vector_db.get_neighbor_vectors(v1, 3):
    print(f"{x}")

vector_db.delete_vector(v1.get_id())
vector_db.delete_vectors([v2, v3])
```

### Qdrant

Qdrant “is a vector similarity search engine that provides a production-ready service with a convenient API to store, search, and manage points (i.e. vectors) with an additional payload.” You can think of the payloads as additional pieces of information that can help you hone in on your search and also receive useful information that you can give to your users.

Using Qdrant in pyvectordb is simple, you only need to change the client to `QdrantDB`

```py
from pyvectordb import QdrantDB

vector_db = QdrantDB(
    host=os.getenv("Q_HOST"),
    api_key=os.getenv("Q_API_KEY"),
    port=os.getenv("Q_PORT"),
    collection=os.getenv("Q_COLLECTION"),
    vector_size=int(os.getenv("Q_SIZE")),
    distance_function=DistanceFunction.COSINE,
)
```

### Chroma DB

Chroma is the AI-native open-source vector database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.

```py
from pyvectordb import ChromaDB

vector_db = ChromaDB(
    host=os.getenv("CH_HOST"),
    port=os.getenv("CH_PORT"),
    auth_provider=os.getenv("CH_AUTH_PROVIDER"),
    auth_credentials=os.getenv("CH_AUTH_CREDENTIALS"),
    collection_name=os.getenv("CH_COLLECTION_NAME"),
    distance_function=DistanceFunction.L2,
)
```

## Available functions

These are available functions in this simple tool

```py
def insert_vector(self, vector: Vector) -> None: ...
def insert_vectors(self, vectors: List[Vector]) -> None: ...
def read_vector(self, id: str) -> Vector | None: ...
def update_vector(self, vector: Vector) -> None: ...
def update_vectors(self, vectors: List[Vector]) -> None: ...
def delete_vector(self, id: str) -> None: ...
def delete_vectors(self, ids: Union[List[str], List[Vector]]) -> None: ...
def get_neighbor_vectors(self, vector: Vector, n: int) -> List[VectorDistance]: ...
```

## Support or Anything

Reach me out on email razifrizqullah@gmail.com