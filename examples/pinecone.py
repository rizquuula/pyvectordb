from dotenv import load_dotenv
load_dotenv()

import os
from pyvectordb import Vector
from pyvectordb.pinecone import PineconeDB
from pyvectordb.distance_function import DistanceFunction

v1 = Vector(
    embedding=[2., 2., 1.],
    metadata={"text": "hello from pyvectordb"}
)
v2 = Vector(
    embedding=[2., 2., 2.],
    metadata={"text": "hi"}
)
v3 = Vector(
    embedding=[2., 2., 3.],
    metadata={"text": "good morning!"}
)

# Initialize PineconeDB
# Option 1: Connect to existing index using host
# vector_db = PineconeDB(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     host=os.getenv("PINECONE_HOST"),
# )

# Option 2: Connect to existing index using index name
# vector_db = PineconeDB(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     index_name=os.getenv("PINECONE_INDEX_NAME"),
# )

# Option 3: Create new index (requires dimension)
vector_db = PineconeDB(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    dimension=3,  # Must match your embedding dimension
    cloud="aws",
    region="us-east-1",
    distance_function=DistanceFunction.COSINE,
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
assert list(v_from_db_updated.embedding) == list(new_embedding), "updated embedding not equal"

# re-update v1 embedding to the v1, check
vector_db.update_vectors([v1, v2, v3])
re_updated_embedding = vector_db.read_vector(v1.get_id()).embedding
assert list(re_updated_embedding) == list(v1.embedding), "re-updated embedding not equal"

for x in vector_db.get_neighbor_vectors(v1, 3):
    print(f"{x}")

vector_db.delete_vector(v1.get_id())
vector_db.delete_vectors([v2, v3])