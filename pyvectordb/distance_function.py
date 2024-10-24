from enum import Enum


class DistanceFunction(Enum):
    """Distance function to measure how var the vector to surrounding vector in vector db"""
    COSINE = "cosine"
    HAMMING = "hamming"
    JACCARD = "jaccard"
    L1 = "l1"
    L2 = "l2"
    MAX_INNER_PRODUCT = "max_inner_product"
    EUCLIDEAN = "euclid"
    DOT = "dot"
    MANHATTAN = "manhattan"
