import numpy as np


class RandomBaseGenerator:

    @staticmethod
    def generate_bases(n: int, k: int) -> np.ndarray:
        if n <= 4:
            return np.array([], dtype=np.uint64)

        if k <= 0:
            return np.array([], dtype=np.uint64)

        """Generuje `k` losowych podstaw z zakresu [2, n-2] w postaci tablicy NumPy."""
        return np.random.randint(2, n - 2, size=k, dtype=np.uint64)
