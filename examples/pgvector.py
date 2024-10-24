import os
from pyvectordb import PostgresVector, Vector
from pyvectordb.distance_function import DistanceFunction

v = Vector(
    embedding=[2, 2, 1]
)

print("VECTOR", v)

pgv = PostgresVector(
    db_user=os.getenv("DB_USER"),
    db_password=os.getenv("DB_PASSWORD"),
    db_host=os.getenv("DB_HOST"),
    db_port=os.getenv("DB_PORT"),
    db_name=os.getenv("DB_NAME"),
)

new_v = pgv.create_vector(v)
print("CREATE_VECTOR", new_v)

new_v = pgv.read_vector(new_v.id)
print("READ_VECTOR", new_v)

new_v = pgv.update_vector(new_v)
print("UPDATE_VECTOR", new_v)

for x in pgv.get_neighbor_vectors(v, 5, DistanceFunction.L2_DISTANCE):
    print(f"{x}")

pgv.delete_vector(new_v.id)
print("DELETE_VECTOR")

