from typing import List
from abc import ABC
from threading import Lock


class Identifiable(ABC):
    """
    Identifiable is a utility base class used to identify
    all instances of a class and its subclasses.
    """

    _id_counter: int = 0
    _id_counter_lock: Lock = Lock()
    _id: List[int]

    def __init__(self):
        with self._id_counter_lock:
            self._id = [self._id_counter]
            self._id_counter += 1

    @property
    def id(self):
        return self._id[0]
