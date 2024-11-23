from strategies.base_strategy import PrimalityTestStrategy
import numpy as np


class CPUPrimalityTestStrategy(PrimalityTestStrategy):
    @staticmethod
    def _miller_test(d, n, base):
        x = pow(base, d, n)
        if x == 1 or x == n - 1:
            return True
        while d != n - 1:
            x = (x * x) % n
            d *= 2
            if x == 1:
                return False
            if x == n - 1:
                return True
        return False

    def is_prime(self, n: int, k: int, bases: np.ndarray) -> bool:
        if n == 2 or n == 3:
            return True
        if n <= 1 or n % 2 == 0:
            return False

        d = n - 1
        while d % 2 == 0:
            d //= 2

        for base in bases:
            if not self._miller_test(d, n, int(base)):
                return False
        return True
