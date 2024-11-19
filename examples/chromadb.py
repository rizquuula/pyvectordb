from dotenv import load_dotenv
load_dotenv()

import os
from pyvectordb import Vector
from pyvectordb.chromadb import ChromaDB
from pyvectordb.distance_function import DistanceFunction

v = Vector(
    embedding=[2., 2., 1.],
    metadata={"text": "hellow from pyvectordb"}
)

ch = ChromaDB(
    host=os.getenv("CH_HOST"),
    port=os.getenv("CH_PORT"),
    auth_provider=os.getenv("CH_AUTH_PROVIDER"),
    auth_credentials=os.getenv("CH_AUTH_CREDENTIALS"),
    collection_name=os.getenv("CH_COLLECTION_NAME"),
    distance_function=DistanceFunction.L2,
)

# full flow test
ch.insert_vector(v)
new_v = ch.read_vector(v.get_id())
ch.update_vector(new_v)

for x in ch.get_neighbor_vectors(v, 3):
    print(f"{x}")

ch.delete_vector(v.get_id())
