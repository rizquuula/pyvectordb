import os
from pyvectordb import Vector
from pyvectordb.qdrant import QdrantDB
from pyvectordb.distance_function import DistanceFunction

v = Vector(
    embedding=[2., 2., 1.]
)

print("VECTOR", v)

qv = QdrantDB(
    host=os.getenv("Q_HOST"),
    api_key=os.getenv("Q_API_KEY"),
    port=os.getenv("Q_PORT"),
    collection=os.getenv("Q_COLLECTION"),
    vector_size=int(os.getenv("Q_SIZE")),
    distance_function=DistanceFunction.COSINE,
)

new_v = qv.create_vector(v)
print("CREATE_VECTOR", new_v)

new_v = qv.read_vector(new_v.id)
print("READ_VECTOR", new_v)

new_v = qv.update_vector(new_v)
print("UPDATE_VECTOR", new_v)

for x in qv.get_neighbor_vectors(v, 5):
    print(f"{x}")

qv.delete_vector(new_v.id)
print("DELETE_VECTOR")
