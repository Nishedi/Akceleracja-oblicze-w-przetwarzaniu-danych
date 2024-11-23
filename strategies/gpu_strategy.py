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

        # Obliczanie wartości d
        d = n - 1
        while d % 2 == 0:
            d //= 2

        # Przenoszenie danych na GPU
        start_data_transfer = time.time()
        gpu_bases = gpuarray.to_gpu(bases.astype(np.int64))
        results = np.zeros(k, dtype=np.int32)
        gpu_results = gpuarray.to_gpu(results)
        end_data_transfer = time.time()

        # Uruchamianie kernela
        start_kernel = time.time()
        block_size = 512
        grid_size = (k + block_size - 1) // block_size
        miller_rabin_test = self.mod.get_function("miller_rabin_test")
        miller_rabin_test(gpu_bases, gpu_results, np.uint64(d), np.uint64(n), np.int32(k), block=(block_size, 1, 1),
                          grid=(grid_size, 1))
        drv.Context.synchronize()  # Czekanie na zakończenie kernela
        end_kernel = time.time()

        # Pobranie wyników z GPU
        start_results_retrieval = time.time()
        results = gpu_results.get()
        end_results_retrieval = time.time()

        # Wyniki czasowe
        # print(f"Czas przenoszenia danych na GPU: {end_data_transfer - start_data_transfer:.6f} s")
        # print(f"Czas uruchomienia kernela: {end_kernel - start_kernel:.6f} s")
        # print(f"Czas pobierania wyników z GPU: {end_results_retrieval - start_results_retrieval:.6f} s")

        # Sprawdzenie, czy wszystkie testy zakończyły się sukcesem
        return all(results)
