from enum import Enum


class DistanceFunction(Enum):
    """Distance function to measure how var the vector to surrounding vector in vector db"""
    COSINE_DISTANCE = "cosine_distance"
    HAMMING_DISTANCE = "hamming_distance"
    JACCARD_DISTANCE = "jaccard_distance"
    L1_DISTANCE = "l1_distance"
    L2_DISTANCE = "l2_distance"
    MAX_INNER_PRODUCT = "max_inner_product"
    