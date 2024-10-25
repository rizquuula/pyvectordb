from enum import Enum


class Distance(Enum):
    SQUARED_L2=	"l2"
    INNER_PRODUCT=	"ip"
    COSINE_SIMILARITY=	"cosine"
