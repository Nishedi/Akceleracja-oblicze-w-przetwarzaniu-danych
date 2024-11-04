import numpy as np
import time
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import concurrent.futures

def generate_chunk(size):
    return np.random.rand(size).astype(np.float32)

chunk_size = 10000000
total_size = 2*990000000
print("Generowanie tablicy, troche to trwa...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    chunks = list(executor.map(generate_chunk, [chunk_size] * (total_size // chunk_size)))
end_time = time.time()
array = np.concatenate(chunks)
print("Długość tablicy:", len(array), "Czas generowania:", end_time - start_time, "sekund")

# Pomiar czasu dla obliczeń w Pythonie
start_time = time.time()
sum_result = np.sum(array)
end_time = time.time()

print("Suma elementów (Python):", sum_result)
print("Czas obliczeń (Python):", end_time - start_time, "sekund")

mod = SourceModule("""
__global__ void sum_reduction(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}
""")

# Przenosimy dane na GPU
gpu_array = gpuarray.to_gpu(array)

# Ustawienia dotyczące bloku i siatki
block_size = 1024
grid_size = (len(array) + block_size - 1) // block_size

# Przygotowanie bufora wyjściowego na GPU
gpu_partial_sums = gpuarray.zeros(grid_size, dtype=np.float32)

# Uruchamianie kernela
sum_reduction = mod.get_function("sum_reduction")
shared_mem_size = block_size * array.itemsize
start_time = time.time()
sum_reduction(gpu_array, gpu_partial_sums, np.int32(len(array)), block=(block_size, 1, 1), grid=(grid_size, 1),
              shared=shared_mem_size)
partial_sums = gpu_partial_sums.get()
sum_result_gpu = np.sum(partial_sums)
end_time = time.time()

print("Suma elementów (PyCUDA z redukcją):", sum_result_gpu)
print("Czas obliczeń (PyCUDA):", end_time - start_time, "sekund")
print("Różnica między wynikami:", abs(sum_result - sum_result_gpu))
