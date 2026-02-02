import time
from dataclasses import dataclass

@dataclass
class Timer:
    start: float = 0.0
    end: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self.end - self.start
