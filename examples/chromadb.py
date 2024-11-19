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

# full flow test
vector_db.insert_vector(v1)
vector_db.insert_vectors([v2, v3])
new_v = vector_db.read_vector(v1.get_id())
vector_db.update_vector(new_v)
vector_db.update_vectors([v1, v2, v3])

for x in vector_db.get_neighbor_vectors(v1, 3):
    print(f"{x}")

vector_db.delete_vector(v1.get_id())
vector_db.delete_vectors([v2, v3])
