import hashlib
import threading

from collections import OrderedDict

def make_key(*parts: str) -> str:

    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data : OrderedDict[str,str] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self,key:str):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self,key:str, value:str) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._data), "capacity" : self.capacity, "hits" : self.hits, "misses": self.misses}

    