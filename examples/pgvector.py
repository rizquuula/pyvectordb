import os
from pyvectordb import Vector
from pyvectordb.pgvector import PgvectorDB
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

new_v = pgv.insert_vector(v)
print("CREATE_VECTOR", new_v)

new_v = pgv.read_vector(new_v.id)
print("READ_VECTOR", new_v)

new_v = pgv.update_vector(new_v)
print("UPDATE_VECTOR", new_v)

for x in pgv.get_neighbor_vectors(v, 5):
    print(f"{x}")

pgv.delete_vector(new_v.id)
print("DELETE_VECTOR")