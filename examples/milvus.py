import os

from dotenv import load_dotenv

from pyvectordb import Vector
from pyvectordb.distance_function import DistanceFunction
from pyvectordb.milvus import MilvusDB

load_dotenv()

v1 = Vector(embedding=[2.0, 2.0, 1.0], metadata={"text": "hello from pyvectordb"})
v2 = Vector(embedding=[2.0, 2.0, 2.0], metadata={"text": "hi"})
v3 = Vector(embedding=[2.0, 2.0, 3.0], metadata={"text": "good morning!"})

vector_db = MilvusDB(
    host=os.getenv("MILVUS_HOST"),
    port=int(os.getenv("MILVUS_PORT", 19530)),
    collection=os.getenv("MILVUS_COLLECTION"),
    vector_size=int(os.getenv("MILVUS_VECTOR_SIZE")),
    distance_function=DistanceFunction.COSINE,
)

# insert new vector
vector_db.insert_vector(v1)
vector_db.insert_vectors([v2, v3])

# read v1
v_from_db = vector_db.read_vector(v1.get_id())

# update v1 embedding
new_embedding = [2.0, 2.0, 4.0]
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
