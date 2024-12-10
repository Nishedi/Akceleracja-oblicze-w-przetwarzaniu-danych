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
        __device__ unsigned long long modular_pow(unsigned long long base, unsigned long long exp, unsigned long long mod) {
            unsigned long long result = 1;
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

        __global__ void miller_rabin_test(unsigned long long *bases, int *results, unsigned long long d, unsigned long long n, int k) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= k) return;

            unsigned long long a = bases[idx];
            unsigned long long x = modular_pow(a, d, n);
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

    def is_prime(self, n: int, k: int, bases: np.ndarray) -> bool:
        if n == 2 or n == 3:
            return True
        if n <= 1 or n % 2 == 0:
            return False

        d = n - 1
        while d % 2 == 0:
            d //= 2

        # Przenoszenie danych na GPU
        gpu_bases = gpuarray.to_gpu(bases.astype(np.int64))
        results = np.zeros(k, dtype=np.int32)
        gpu_results = gpuarray.to_gpu(results)

        block_size = 256

        # Uruchamianie kernela
        block_size = max_threads_per_block
        grid_size = (k + block_size - 1) // block_size
        miller_rabin_test = self.mod.get_function("miller_rabin_test")
        miller_rabin_test(gpu_bases, gpu_results, np.uint64(d), np.uint64(n), np.int32(k), block=(block_size, 1, 1),
                          grid=(grid_size, 1))
        drv.Context.synchronize()  # Czekanie na zakończenie kernela

        # Pobranie wyników z GPU
        results = gpu_results.get()

        # Sprawdzenie, czy wszystkie testy zakończyły się sukcesem
        return all(results)
