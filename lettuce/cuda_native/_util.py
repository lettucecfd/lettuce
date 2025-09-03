from typing import Union, List

import mmh3


def lettuce_hash(value: Union[str, List[str]]) -> str:
    if isinstance(value, list):
        value = ' '.join(value)
    return mmh3.hash_bytes(value).hex()[:8]
