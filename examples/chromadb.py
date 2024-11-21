from dotenv import load_dotenv
load_dotenv()

import os
from pyvectordb import Vector
from pyvectordb.chromadb import ChromaDB
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

vector_db = ChromaDB(
    host=os.getenv("CH_HOST"),
    port=os.getenv("CH_PORT"),
    auth_provider=os.getenv("CH_AUTH_PROVIDER"),
    auth_credentials=os.getenv("CH_AUTH_CREDENTIALS"),
    collection_name=os.getenv("CH_COLLECTION_NAME"),
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
assert v_from_db_updated.embedding == new_embedding, "updated embedding not equal"

# re-update v1 embedding to the v1, check
vector_db.update_vectors([v1, v2, v3])
re_updated_embedding = vector_db.read_vector(v1.get_id()).embedding
assert re_updated_embedding == v1.embedding, "re-updated embedding not equal"

for x in vector_db.get_neighbor_vectors(v1, 3):
    print(f"{x}")

vector_db.delete_vector(v1.get_id())
vector_db.delete_vectors([v2, v3])
