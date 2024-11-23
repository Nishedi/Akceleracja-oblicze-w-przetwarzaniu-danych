from abc import ABC, abstractmethod
import numpy as np


class PrimalityTestStrategy(ABC):
    @abstractmethod
    def is_prime(self, n: int, k: int, bases: np.ndarray) -> bool:
        pass
