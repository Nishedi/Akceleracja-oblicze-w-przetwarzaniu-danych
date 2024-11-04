import time
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import random
from pycuda.compiler import SourceModule
from strategies.base_strategy import PrimalityTestStrategy

class GPUPrimalityTestStrategy(PrimalityTestStrategy):
    def __init__(self):
        self.mod = SourceModule("""
        __device__ long long modular_pow(long long base, long long exp, long long mod) {
            long long result = 1;
            base = base % mod;
            while (exp > 0) {
                if (exp % 2 == 1) {
                    result = (result * base) % mod;
                }
                exp = exp >> 1;
                base = (base * base) % mod;
            }
            return result;
        }

        __global__ void miller_rabin_test(long long *bases, int *results, long long d, long long n, int k) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= k) return;

            long long a = bases[idx];
            long long x = modular_pow(a, d, n);
            if (x == 1 || x == n - 1) {
                results[idx] = 1;
                return;
            }

            while (d != n - 1) {
                x = (x * x) % n;
                d *= 2;
                if (x == 1) {
                    results[idx] = 0;
                    return;
                }
                if (x == n - 1) {
                    results[idx] = 1;
                    return;
                }
            }

            results[idx] = 0;
        }
        """)

    def is_prime(self, n: int, k: int) -> bool:
        if n == 2 or n == 3:
            return True
        if n <= 1 or n % 2 == 0:
            return False

        # Obliczanie wartości d
        d = n - 1
        while d % 2 == 0:
            d //= 2

        # Generowanie losowych podstaw
        bases = np.array([random.randint(2, n - 2) for _ in range(k)], dtype=np.int64)
        results = np.zeros(k, dtype=np.int32)

        # Przenoszenie danych na GPU
        gpu_bases = gpuarray.to_gpu(bases)
        gpu_results = gpuarray.to_gpu(results)

        # Uruchamianie kernela
        block_size = 512
        grid_size = (k + block_size - 1) // block_size
        miller_rabin_test = self.mod.get_function("miller_rabin_test")
        miller_rabin_test(gpu_bases, gpu_results, np.int64(d), np.int64(n), np.int32(k), block=(block_size, 1, 1), grid=(grid_size, 1))

        # Pobranie wyników z GPU
        results = gpu_results.get()

        # Sprawdzenie, czy wszystkie testy zakończyły się sukcesem
        return all(results)
