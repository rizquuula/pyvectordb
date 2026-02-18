from enum import Enum


class Distance(Enum):
    """Weaviate distance metrics"""

    COSINE = "cosine"
    DOT = "dot"
    L2_SQUARED = "l2-squared"
    HAMMING = "hamming"
    MANHATTAN = "manhattan"
