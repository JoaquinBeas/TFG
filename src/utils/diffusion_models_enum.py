from enum import Enum

class MNISTModelType(Enum):
    SIMPLE_CNN     = "simple_cnn"
    COMPLEX_CNN    = "complex_cnn"
    DECISION_TREE  = "decision_tree"
    RESNET_PREACT  = "resnet_preact"
