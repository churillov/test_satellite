"""Common functions."""
import numpy as np
from torch import Tensor
from typing import List, Dict
from faiss import IndexFlatL2


def get_class_to_label(class_names: List) -> Dict[str, int]:
    """
    The function creates a dictionary. The key is the class name. The value is the label.
    Args:
        class_names: List names class

    Returns:
        Return dict. The key is the class name. The value is the label.
    """
    class2label = {}
    for ind, class_name in enumerate(class_names):
        class2label[class_name] = ind
    return class2label


def get_label_to_class_dict(class_dict: Dict[str, int]) -> Dict[int, str]:
    """
    The function creates a dictionary. The key is the label. The value is the class name.
    Args:
        class_dict: Dict, the key is the class name. The value is the label.

    Returns:
        Return dict. The key is the label. The value is the class name.
    """
    label2class = {}
    for key in class_dict:
        label2class[class_dict[key]] = key
    return label2class


def get_class_name(index: IndexFlatL2, emb: Tensor, labels: Dict[str, str], k: int = 1) -> int:
    """
    The function predicts a label using a database of embeddings.
    Args:
        index: Vector Database.
        emb: Image embedding.
        labels: The dictionary contains tags to get the closest similar images.
                The key is the index, the value is the tag.
        k: Number of neighbors.

    Returns:
        Nearest neighbor label.
    """
    distances, indexes = index.search(emb, k)

    predict = []
    for i in indexes[0]:
        predict.append(int(labels[f'{i}']))

    vals, counts = np.unique(predict, return_counts=True)
    return vals[np.argmax(counts)]


def get_index_similarity(index: IndexFlatL2, emb: Tensor, k: int = 1) -> List:
    """
    The function finds the indices of the nearest neighbors.
    Args:
        index: Vector Database.
        emb: Image embedding.
        k: Number of neighbors.

    Returns:
        Returns a list of indices of the nearest neighbors.
    """
    _, indexes = index.search(emb, k)
    return indexes[0]
