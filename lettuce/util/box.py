from typing import Generic, TypeVar, List

BoxItem = TypeVar('BoxItem')


class Box(Generic[BoxItem]):
    """
    Box is a utility structure used to avoid implicit copies of a variable.
    When a Box is passed, the item the Box holds is not copied but a reference to it is used.
    """

    def __init__(self, item: BoxItem):
        self._item: List[BoxItem] = list(item)

    def get(self) -> BoxItem:
        return self._item[0]

    def set(self, item: BoxItem):
        self._item[0] = item

    def copy(self):
        return Box(self.get())
