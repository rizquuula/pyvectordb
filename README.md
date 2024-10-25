# PyVectorDB

Simple Wrapper for vector database in Python with minimal support for CRUD and retrieve.

## Installation 

    pip install pyvectordb

## Usage Example

### PGVector

PGvector is an extension for PostgreSQL that allows the storage, indexing, and querying of vector embeddings. It is designed to support vector similarity search, which is useful in machine learning applications like natural language processing, image recognition, and recommendation systems. By storing vector embeddings as a data type, PGvector enables efficient similarity searches using distance metrics such as cosine similarity, Euclidean distance, inner product, etc.

```py
import os
from pyvectordb import PgvectorDB, Vector
from pyvectordb.distance_function import DistanceFunction

v = Vector(
    embedding=[2., 2., 1.]
)

print("VECTOR", v)

pgv = PgvectorDB(
    db_user=os.getenv("PG_USER"),
    db_password=os.getenv("PG_PASSWORD"),
    db_host=os.getenv("PG_HOST"),
    db_port=os.getenv("PG_PORT"),
    db_name=os.getenv("PG_NAME"),
    collection=os.getenv("PG_COLLECTION"),
    distance_function=DistanceFunction.L2,
)

new_v = pgv.create_vector(v)
print("CREATE_VECTOR", new_v)

new_v = pgv.read_vector(new_v.id)
print("READ_VECTOR", new_v)

new_v = pgv.update_vector(new_v)
print("UPDATE_VECTOR", new_v)

for x in pgv.get_neighbor_vectors(v, 5):
    print(f"{x}")

pgv.delete_vector(new_v.id)
print("DELETE_VECTOR")
```

### Qdrant

Qdrant “is a vector similarity search engine that provides a production-ready service with a convenient API to store, search, and manage points (i.e. vectors) with an additional payload.” You can think of the payloads as additional pieces of information that can help you hone in on your search and also receive useful information that you can give to your users.

Using Qdrant in pyvectordb is simple, you only need to change the client to `QdrantDB`

```py
from pyvectordb import QdrantDB

qv = QdrantDB(
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

ch = ChromaDB(
    host=os.getenv("CH_HOST"),
    port=os.getenv("CH_PORT"),
    auth_provider=os.getenv("CH_AUTH_PROVIDER"),
    auth_credentials=os.getenv("CH_AUTH_CREDENTIALS"),
    collection_name=os.getenv("CH_COLLECTION_NAME"),
    distance_function=DistanceFunction.L2,
)
```

## Support or Anything

Reach me out on email razifrizqullah@gmail.com