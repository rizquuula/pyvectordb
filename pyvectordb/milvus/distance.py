from enum import Enum


class Distance(Enum):
    """Milvus distance metrics"""

    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"
