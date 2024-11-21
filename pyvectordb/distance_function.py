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
    
    @staticmethod
    def from_str(text: str) -> "DistanceFunction":
        text = text.lower()
        
        if text == "cosine": return DistanceFunction.COSINE 
        if text == "hamming": return DistanceFunction.HAMMING 
        if text == "jaccard": return DistanceFunction.JACCARD 
        if text == "l1": return DistanceFunction.L1 
        if text == "l2": return DistanceFunction.L2 
        if text == "max_inner_product": return DistanceFunction.MAX_INNER_PRODUCT 
        if text == "euclid": return DistanceFunction.EUCLIDEAN 
        if text == "euclidean": return DistanceFunction.EUCLIDEAN 
        if text == "dot": return DistanceFunction.DOT 
        if text == "manhattan": return DistanceFunction.MANHATTAN 
        
        raise ValueError("invalid string for distance function")
