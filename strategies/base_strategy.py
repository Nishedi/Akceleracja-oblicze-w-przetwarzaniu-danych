from abc import ABC, abstractmethod


class PrimalityTestStrategy(ABC):
    @abstractmethod
    def is_prime(self, n: int, k: int) -> bool:
        pass
