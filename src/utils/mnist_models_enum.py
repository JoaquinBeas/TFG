from enum import Enum

class MNISTModelType(Enum):
    SIMPLE_CNN     = "simple_cnn"
    COMPRESSED_CNN    = "compressed_cnn"
    DECISION_TREE  = "decision_tree"
    RESNET_PREACT  = "resnet_preact"
