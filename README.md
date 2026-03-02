# PyVectorDB

[![GitHub license](https://img.shields.io/github/license/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb/network)
![GitHub watchers](https://img.shields.io/github/watchers/rizquuula/pyvectordb)
[![GitHub issues](https://img.shields.io/github/issues/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb/pulls)
[![Contributors](https://img.shields.io/github/contributors/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb/graphs/contributors)
![GitHub last commit](https://img.shields.io/github/last-commit/rizquuula/pyvectordb)
![Commit activity](https://img.shields.io/github/commit-activity/y/rizquuula/pyvectordb)
[![GitHub repo size](https://img.shields.io/github/repo-size/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb)
[![GitHub languages](https://img.shields.io/github/languages/top/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb)
[![GitHub languages count](https://img.shields.io/github/languages/count/rizquuula/pyvectordb)](https://github.com/rizquuula/pyvectordb)
[![PyPI Version](https://img.shields.io/pypi/v/pyvectordb)](https://pypi.org/project/pyvectordb/)
[![Python Version](https://img.shields.io/pypi/python-version/pyvectordb)](https://pypi.org/project/pyvectordb/)

**Simple** Python wrapper for CRUD operations and vector similarity search across multiple vector databases.

## Features

- 🚀 **Unified API** - Single interface for multiple vector databases
- 🔄 **Multi-database support** - PGVector, Qdrant, ChromaDB, Milvus, Weaviate, Pinecone
- 📦 **Lightweight** - Install only the dependencies you need
- 🛠️ **Full CRUD** - Insert, read, update, delete, and similarity search

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Databases](#supported-databases)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- Python 3.9+

## Installation

### Install all dependencies (not recommended - requires a lot of disk space)

```sh
pip install pyvectordb[all]
```

### Install specific database support (recommended)

```sh
pip install pyvectordb[pgvector]   # PostgreSQL with pgvector
pip install pyvectordb[qdrant]     # Qdrant
pip install pyvectordb[chromadb]   # ChromaDB
pip install pyvectordb[milvus]     # Milvus
pip install pyvectordb[weaviate]   # Weaviate
pip install pyvectordb[pinecone]   # Pinecone
```

## Quick Start

```py
from dotenv import load_dotenv
load_dotenv()

import os
from pyvectordb import Vector
from pyvectordb.pgvector.pgvector import PgvectorDB
from pyvectordb.distance_function import DistanceFunction

# Create vectors with embeddings and metadata
v1 = Vector(
    embedding=[2., 2., 1.],
    metadata={"text": "hello from pyvectordb"}
)
v2 = Vector(
    embedding=[2., 2., 2.],
    metadata={"text": "hi"}
)

# Initialize database connection
vector_db = PgvectorDB(
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    db_name=os.getenv("PG_NAME"),
    collection=os.getenv("PG_COLLECTION"),
    distance_function=DistanceFunction.L2,
)

# CRUD operations
vector_db.insert_vector(v1)
vector_db.insert_vectors([v2])

# Read
v_from_db = vector_db.read_vector(v1.get_id())

# Update
v_from_db.embedding = [2., 2., 4.]
vector_db.update_vector(v_from_db)

# Delete
vector_db.delete_vector(v1.get_id())

# Similarity search
neighbors = vector_db.search(v1, k=3)
```

## Supported Databases

| Database | Install Extra | Description |
|----------|---------------|-------------|
| [PGVector](https://github.com/pgvector/pgvector) | `[pgvector]` | PostgreSQL vector extension |
| [Qdrant](https://qdrant.tech/) | `[qdrant]` | Vector similarity search engine |
| [ChromaDB](https://www.trychroma.com/) | `[chromadb]` | AI-native open-source vector database |
| [Milvus](https://milvus.io/) | `[milvus]` | Open-source vector database |
| [Weaviate](https://weaviate.io/) | `[weaviate]` | Cloud-native vector database |
| [Pinecone](https://www.pinecone.io/) | `[pinecone]` | Managed vector database |

### Database-Specific Usage

<details>
<summary>PGVector</summary>

```py
from pyvectordb.pgvector.pgvector import PgvectorDB
from pyvectordb.distance_function import DistanceFunction

db = PgvectorDB(
    user="postgres",
    password="password",
    host="localhost",
    port=5432,
    db_name="vectordb",
    collection="my_collection",
    distance_function=DistanceFunction.L2,
)
```
</details>

<details>
<summary>Qdrant</summary>

```py
from pyvectordb import QdrantDB
from pyvectordb.distance_function import DistanceFunction

db = QdrantDB(
    host="localhost",
    api_key="your-api-key",
    port=6333,
    collection="my_collection",
    vector_size=1536,
    distance_function=DistanceFunction.COSINE,
)
```
</details>

<details>
<summary>ChromaDB</summary>

```py
from pyvectordb import ChromaDB
from pyvectordb.distance_function import DistanceFunction

db = ChromaDB(
    host="localhost",
    port=8000,
    collection_name="my_collection",
    distance_function=DistanceFunction.L2,
)
```
</details>

<details>
<summary>Milvus</summary>

```py
from pyvectordb import MilvusDB
from pyvectordb.distance_function import DistanceFunction

db = MilvusDB(
    host="localhost",
    port=19530,
    collection="my_collection",
    vector_size=1536,
    distance_function=DistanceFunction.COSINE,
)
```
</details>

<details>
<summary>Weaviate</summary>

```py
from pyvectordb import WeaviateDB
from pyvectordb.distance_function import DistanceFunction

db = WeaviateDB(
    host="localhost",
    port=8080,
    grpc_port=50051,
    api_key="your-api-key",
    collection="my_collection",
    vector_size=1536,
    distance_function=DistanceFunction.COSINE,
)
```
</details>

<details>
<summary>Pinecone</summary>

```py
from pyvectordb import PineconeDB
from pyvectordb.distance_function import DistanceFunction

db = PineconeDB(
    api_key="your-api-key",
    environment="us-east-1",
    collection="my_collection",
    vector_size=1536,
    distance_function=DistanceFunction.COSINE,
)
```
</details>

## API Reference

All database implementations support the following unified interface:

| Method | Description |
|--------|-------------|
| `insert_vector(vector)` | Insert a single vector |
| `insert_vectors(vectors)` | Insert multiple vectors |
| `read_vector(id)` | Get vector by ID |
| `update_vector(vector)` | Update a single vector |
| `update_vectors(vectors)` | Update multiple vectors |
| `delete_vector(id)` | Delete vector by ID |
| `delete_vectors(ids)` | Delete multiple vectors by ID |
| `search(vector, k)` | Find k nearest neighbors |

## Contributing

Contributions are welcome! Please feel free to submit a [Pull Request](https://github.com/rizquuula/pyvectordb/pulls) or open an [Issue](https://github.com/rizquuula/pyvectordb/issues).

### Development Setup

```sh
# Clone the repository
git clone https://github.com/rizquuula/pyvectordb.git
cd pyvectordb

# Install development dependencies
pip install -e ".[all]"

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 **Email:** [razifrizqullah@gmail.com](mailto:razifrizqullah@gmail.com)
- 💬 **GitHub Issues:** [Submit an Issue](https://github.com/rizquuula/pyvectordb/issues)
- 💼 **LinkedIn:** [razifrizqullah](https://www.linkedin.com/in/razifrizqullah/)

---

If you find this project helpful, please consider:
- ⭐ Starring the repository
- 🍴 Forking and contributing
- 🗨 Sharing your feedback

Thank you for your support!
