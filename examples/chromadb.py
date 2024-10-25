import os
from pyvectordb import ChromaDB, Vector
from pyvectordb.distance_function import DistanceFunction

v = Vector(
    embedding=[2., 2., 1.]
)

print("VECTOR", v)

ch = ChromaDB(
    host=os.getenv("CH_HOST"),
    port=os.getenv("CH_PORT"),
    auth_provider=os.getenv("CH_AUTH_PROVIDER"),
    auth_credentials=os.getenv("CH_AUTH_CREDENTIALS"),
    collection_name=os.getenv("CH_COLLECTION_NAME"),
    distance_function=DistanceFunction.L2,
)

new_v = ch.create_vector(v)
print("CREATE_VECTOR", new_v)

new_v = ch.read_vector(new_v.id)
print("READ_VECTOR", new_v)

new_v = ch.update_vector(new_v)
print("UPDATE_VECTOR", new_v)

for x in ch.get_neighbor_vectors(v, 3):
    print(f"{x}")

ch.delete_vector(new_v.id)
print("DELETE_VECTOR")

